//===- LoopStreamingSwitcher.cpp - Per-loop SSVE/NEON switching ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass uses a trained GBM model to decide per-loop whether Streaming
// SVE (SSVE) or NEON is faster. Loops predicted to benefit from SSVE are
// outlined into separate functions marked with aarch64_pstate_sm_body.
//
// The pass runs BEFORE the Loop Vectorizer so that the LV sees the streaming
// attribute on outlined functions and uses scalable SVE vectorization.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Scalar/LoopStreamingSwitcher.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/LoopAccessAnalysis.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/LoopVectorFeatures.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/CodeExtractor.h"
#include <cmath>

using namespace llvm;

#define DEBUG_TYPE "loop-streaming-switcher"

STATISTIC(NumOutlined, "Number of loops outlined for streaming SVE");

static cl::opt<bool> AutoStreamingMode(
    "loop-vectorize-auto-streaming", cl::init(false), cl::Hidden,
    cl::desc("Automatically select SSVE vs NEON per-loop using a trained "
             "GBM model. Outlines SSVE-beneficial loops into streaming "
             "functions before the Loop Vectorizer runs."));

static cl::opt<bool> SwitcherVerbose(
    "loop-streaming-switcher-verbose", cl::init(false), cl::Hidden,
    cl::desc("Print per-loop SSVE/NEON decisions to stderr."));

static cl::opt<int> OutlineLoopIndex(
    "outline-loop-index", cl::init(-1), cl::Hidden,
    cl::desc("Outline only the Nth qualifying inner loop (0-indexed). "
             "Bypasses GBM. -1 (default) uses GBM as normal."));

static cl::list<int> OutlineLoopIndices(
    "outline-loop-indices", cl::Hidden, cl::CommaSeparated,
    cl::desc("Comma-separated list of qualifying loop indices to outline. "
             "Bypasses GBM. Overrides -outline-loop-index if non-empty."));

//===----------------------------------------------------------------------===//
// GBM Classifier (auto-generated)
//===----------------------------------------------------------------------===//

namespace {

#include "LoopStreamingSwitcherGBM.inc"

//===----------------------------------------------------------------------===//
// Loop Outlining
//===----------------------------------------------------------------------===//

static Function *outlineLoopForStreaming(Loop *L, DominatorTree &DT,
                                         AssumptionCache *AC,
                                         Function &ParentFunc) {
  if (!L->isLoopSimplifyForm()) {
    LLVM_DEBUG(dbgs() << "LSS: Skipping loop not in LoopSimplify form: "
                      << L->getLocStr() << "\n");
    return nullptr;
  }

  CodeExtractorAnalysisCache CEAC(ParentFunc);
  CodeExtractor Extractor(L->getBlocks(), &DT, /*AggregateArgs=*/false,
                          /*BFI=*/nullptr, /*BPI=*/nullptr, AC);

  Function *NewFunc = Extractor.extractCodeRegion(CEAC);
  if (!NewFunc) {
    LLVM_DEBUG(dbgs() << "LSS: CodeExtractor failed for loop: "
                      << L->getLocStr() << "\n");
    return nullptr;
  }

  NewFunc->addFnAttr("aarch64_pstate_sm_body");

  if (ParentFunc.hasFnAttribute("target-features")) {
    StringRef Features =
        ParentFunc.getFnAttribute("target-features").getValueAsString();
    if (Features.contains("+sme")) {
      NewFunc->addFnAttr("target-features", Features);
    } else {
      std::string NewFeatures = (Features + ",+sme").str();
      NewFunc->addFnAttr("target-features", NewFeatures);
    }
  } else {
    NewFunc->addFnAttr("target-features", "+sme");
  }

  if (ParentFunc.hasFnAttribute("target-cpu"))
    NewFunc->addFnAttr(ParentFunc.getFnAttribute("target-cpu"));

  ++NumOutlined;
  return NewFunc;
}

} // anonymous namespace

//===----------------------------------------------------------------------===//
// Pass Implementation
//===----------------------------------------------------------------------===//

PreservedAnalyses LoopStreamingSwitcherPass::run(Function &F,
                                                  FunctionAnalysisManager &AM) {
  if (!AutoStreamingMode && OutlineLoopIndex < 0 && OutlineLoopIndices.empty())
    return PreservedAnalyses::all();

  if (F.hasFnAttribute("aarch64_pstate_sm_enabled") ||
      F.hasFnAttribute("aarch64_pstate_sm_body"))
    return PreservedAnalyses::all();

  if (F.hasOptNone() || F.empty())
    return PreservedAnalyses::all();

  auto &LI = AM.getResult<LoopAnalysis>(F);
  if (LI.empty())
    return PreservedAnalyses::all();

  auto &SE = AM.getResult<ScalarEvolutionAnalysis>(F);
  auto &DT = AM.getResult<DominatorTreeAnalysis>(F);
  auto *AC = AM.getCachedResult<AssumptionAnalysis>(F);
  auto &LAIs = AM.getResult<LoopAccessAnalysis>(F);
  const DataLayout &DL = F.getDataLayout();

  SmallVector<Loop *, 8> InnermostLoops;
  for (Loop *L : LI) {
    SmallVector<Loop *, 8> Stack;
    Stack.push_back(L);
    while (!Stack.empty()) {
      Loop *Current = Stack.pop_back_val();
      if (Current->getSubLoops().empty())
        InnermostLoops.push_back(Current);
      for (Loop *Sub : Current->getSubLoops())
        Stack.push_back(Sub);
    }
  }

  // Build set of forced indices for list mode.
  DenseSet<int> ForcedIndices;
  for (int Idx : OutlineLoopIndices)
    ForcedIndices.insert(Idx);

  // First pass: decide which loops to outline, collecting pointers.
  int QualifyingIdx = 0;
  SmallVector<Loop *, 8> ToOutline;

  for (Loop *L : InnermostLoops) {
    if (!isQualifyingLoop(L, SE, LAIs, DL)) {
      LLVM_DEBUG(dbgs() << "LSS: Loop not qualifying: "
                        << L->getLocStr() << "\n");
      if (SwitcherVerbose)
        errs() << "LSS: NEON — " << F.getName() << " " << L->getLocStr() << "\n";
      continue;
    }

    bool ShouldOutline = false;

    if (!ForcedIndices.empty()) {
      // List mode: outline loops whose qualifying index is in the set.
      ShouldOutline = ForcedIndices.count(QualifyingIdx);
    } else if (OutlineLoopIndex >= 0) {
      // Forced mode: outline only the Nth qualifying loop, bypass GBM.
      ShouldOutline = (QualifyingIdx == OutlineLoopIndex);
    } else {
      // GBM mode: let the model decide.
      const LoopAccessInfo &LAI = LAIs.getInfo(*L);
      LoopVectorFeatures Features = extractLoopVectorFeatures(L, SE, LAI, DL);
      ShouldOutline = shouldUseStreamingSVE(Features);
    }

    ++QualifyingIdx;

    if (!ShouldOutline) {
      LLVM_DEBUG(dbgs() << "LSS: Loop stays NEON: " << L->getLocStr() << "\n");
      if (SwitcherVerbose)
        errs() << "LSS: NEON — " << F.getName() << " " << L->getLocStr() << "\n";
      continue;
    }

    LLVM_DEBUG(dbgs() << "LSS: Will outline loop: " << L->getLocStr() << "\n");
    if (SwitcherVerbose)
      errs() << "LSS: SSVE — " << F.getName() << " " << L->getLocStr() << "\n";

    ToOutline.push_back(L);
  }

  if (ToOutline.empty())
    return PreservedAnalyses::all();

  // Second pass: outline collected loops in reverse order so that earlier
  // loop pointers remain valid (later blocks are extracted first).
  for (int I = (int)ToOutline.size() - 1; I >= 0; --I) {
    Loop *L = ToOutline[I];
    Function *NewFunc = outlineLoopForStreaming(L, DT, AC, F);
    if (NewFunc) {
      LLVM_DEBUG(dbgs() << "LSS: Outlined loop into streaming function: "
                        << NewFunc->getName() << "\n");
    }
  }

  return PreservedAnalyses::none();
}
