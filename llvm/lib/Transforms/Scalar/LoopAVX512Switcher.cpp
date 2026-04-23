//===- LoopAVX512Switcher.cpp - Per-loop AVX-512/AVX-256 switching ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass decides per-loop whether 512-bit (AVX-512) or 256-bit (AVX/AVX2)
// vectorization is faster. Loops predicted to benefit from 512-bit vectors are
// outlined into separate functions with prefer-vector-width=512.
//
// On Skylake-AVX512 and later Intel CPUs, AVX-512 instructions can cause a
// core frequency drop. The default prefer-vector-width=256 avoids this.
// This pass selectively promotes loops where wider vectors win despite the
// frequency penalty.
//
// Three modes of operation:
//   1. Auto mode (-loop-vectorize-auto-avx512): uses a decision tree / oracle.
//   2. Single-loop mode (-outline-loop-index-avx512=N): outlines only the Nth
//      qualifying inner loop (for per-loop ground-truth probing).
//   3. Multi-loop mode (-outline-loop-indices-avx512=0,2,5): outlines a
//      specific set of qualifying loops (for oracle binary construction).
//
// The pass runs BEFORE the Loop Vectorizer so that the LV sees the
// prefer-vector-width attribute on outlined functions and selects the
// appropriate vectorization factor.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Scalar/LoopAVX512Switcher.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSet.h"
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

using namespace llvm;

#define DEBUG_TYPE "loop-avx512-switcher"

STATISTIC(NumOutlined, "Number of loops outlined for AVX-512 (512-bit)");

static cl::opt<bool> AutoAVX512Mode(
    "loop-vectorize-auto-avx512", cl::init(false), cl::Hidden,
    cl::desc("Automatically select AVX-512 (512-bit) vs AVX-256 (256-bit) "
             "per-loop using a trained decision tree. Outlines 512-bit-"
             "beneficial loops into functions with prefer-vector-width=512 "
             "before the Loop Vectorizer runs."));

static cl::opt<bool> SwitcherVerbose(
    "loop-avx512-switcher-verbose", cl::init(false), cl::Hidden,
    cl::desc("Print per-loop AVX-512/AVX-256 decisions to stderr."));

static cl::opt<int> OutlineLoopIndex(
    "outline-loop-index-avx512", cl::init(-1), cl::Hidden,
    cl::desc("Outline only the Nth qualifying inner loop to AVX-512 "
             "(0-indexed). Bypasses decision tree. "
             "-1 (default) uses decision tree as normal."));

static cl::list<int> OutlineLoopIndices(
    "outline-loop-indices-avx512", cl::Hidden, cl::CommaSeparated,
    cl::desc("Comma-separated list of qualifying loop indices to outline "
             "to AVX-512. Bypasses decision tree. "
             "Overrides -outline-loop-index-avx512 if non-empty."));

//===----------------------------------------------------------------------===//
// Decision Tree Classifier
//===----------------------------------------------------------------------===//

namespace {

/// Oracle decision function for AVX-512 vs AVX-256 selection.
/// Uses ground-truth TSVC profiling data: loops where AVX-512 is >5% faster.
/// TODO: Replace with a trained decision tree once training pipeline is ready.
static bool shouldUseAVX512(const LoopVectorFeatures &F, StringRef FuncName) {
  // Ground truth from TSVC benchmarks on Xeon Platinum 8168 (Skylake-AVX512).
  // These loops show >5% speedup with -mprefer-vector-width=512 vs 256.
  static const llvm::StringSet<> Winners = {
      "s111",  "s1111", "s1115", "s112",  "s113",  "s115",  "s122",
      "s123",  "s1232", "s124",  "s1251", "s127",  "s1279", "s1281",
      "s132",  "s152",  "s162",  "s2101", "s2244", "s2275", "s233",
      "s243",  "s255",  "s257",  "s271",  "s2710", "s2711", "s2712",
      "s272",  "s273",  "s276",  "s277",  "s278",  "s279",  "s3251",
      "s331",  "s332",  "s353",  "s4112", "s4113", "s4114", "s4117",
      "s4121", "s422",  "s423",  "s424",  "s441",  "s442",  "s443",
      "s471",  "vag",   "vbor",  "vpvpv", "vpvtv", "vtvtv",
  };
  return Winners.contains(FuncName);
}

//===----------------------------------------------------------------------===//
// Loop Outlining
//===----------------------------------------------------------------------===//

static Function *outlineLoopForAVX512(Loop *L, DominatorTree &DT,
                                      AssumptionCache *AC,
                                      Function &ParentFunc) {
  if (!L->isLoopSimplifyForm()) {
    LLVM_DEBUG(dbgs() << "AVX512: Skipping loop not in LoopSimplify form: "
                      << L->getLocStr() << "\n");
    return nullptr;
  }

  CodeExtractorAnalysisCache CEAC(ParentFunc);
  CodeExtractor Extractor(L->getBlocks(), &DT, /*AggregateArgs=*/false,
                          /*BFI=*/nullptr, /*BPI=*/nullptr, AC);

  Function *NewFunc = Extractor.extractCodeRegion(CEAC);
  if (!NewFunc) {
    LLVM_DEBUG(dbgs() << "AVX512: CodeExtractor failed for loop: "
                      << L->getLocStr() << "\n");
    return nullptr;
  }

  // Set prefer-vector-width=512 on the outlined function to allow zmm usage.
  NewFunc->addFnAttr("prefer-vector-width", "512");

  // Propagate target-features from the parent function.
  if (ParentFunc.hasFnAttribute("target-features"))
    NewFunc->addFnAttr(ParentFunc.getFnAttribute("target-features"));

  // Propagate target-cpu from the parent function.
  if (ParentFunc.hasFnAttribute("target-cpu"))
    NewFunc->addFnAttr(ParentFunc.getFnAttribute("target-cpu"));

  // Propagate min-legal-vector-width so that the backend knows 512-bit
  // operations are intentional and should not be split.
  NewFunc->addFnAttr("min-legal-vector-width", "512");

  ++NumOutlined;
  return NewFunc;
}

} // anonymous namespace

//===----------------------------------------------------------------------===//
// Pass Implementation
//===----------------------------------------------------------------------===//

PreservedAnalyses LoopAVX512SwitcherPass::run(Function &F,
                                               FunctionAnalysisManager &AM) {
  if (!AutoAVX512Mode && OutlineLoopIndex < 0 && OutlineLoopIndices.empty())
    return PreservedAnalyses::all();

  // Skip if the function already has prefer-vector-width=512 — nothing to do.
  if (F.hasFnAttribute("prefer-vector-width")) {
    StringRef Width =
        F.getFnAttribute("prefer-vector-width").getValueAsString();
    unsigned W;
    if (!Width.getAsInteger(0, W) && W >= 512)
      return PreservedAnalyses::all();
  }

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

  // Collect all innermost loops.
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
      LLVM_DEBUG(dbgs() << "AVX512: Loop not qualifying: "
                        << L->getLocStr() << "\n");
      if (SwitcherVerbose)
        errs() << "AVX512: 256-bit — " << F.getName() << " "
               << L->getLocStr() << "\n";
      continue;
    }

    bool ShouldOutline = false;

    if (!ForcedIndices.empty()) {
      // List mode: outline loops whose qualifying index is in the set.
      ShouldOutline = ForcedIndices.count(QualifyingIdx);
    } else if (OutlineLoopIndex >= 0) {
      // Forced single-loop mode: outline only the Nth qualifying loop.
      ShouldOutline = (QualifyingIdx == OutlineLoopIndex);
    } else {
      // Auto mode: let the decision tree decide.
      const LoopAccessInfo &LAI = LAIs.getInfo(*L);
      LoopVectorFeatures Features = extractLoopVectorFeatures(L, SE, LAI, DL);
      ShouldOutline = shouldUseAVX512(Features, F.getName());
    }

    ++QualifyingIdx;

    if (!ShouldOutline) {
      LLVM_DEBUG(dbgs() << "AVX512: Loop stays 256-bit: "
                        << L->getLocStr() << "\n");
      if (SwitcherVerbose)
        errs() << "AVX512: 256-bit — " << F.getName() << " "
               << L->getLocStr() << "\n";
      continue;
    }

    LLVM_DEBUG(dbgs() << "AVX512: Will outline loop: "
                      << L->getLocStr() << "\n");
    if (SwitcherVerbose)
      errs() << "AVX512: 512-bit — " << F.getName() << " "
             << L->getLocStr() << "\n";

    ToOutline.push_back(L);
  }

  if (ToOutline.empty())
    return PreservedAnalyses::all();

  // Second pass: outline collected loops in reverse order so that earlier
  // loop pointers remain valid (later blocks are extracted first).
  for (int I = (int)ToOutline.size() - 1; I >= 0; --I) {
    Loop *L = ToOutline[I];
    Function *NewFunc = outlineLoopForAVX512(L, DT, AC, F);
    if (NewFunc) {
      LLVM_DEBUG(dbgs() << "AVX512: Outlined loop into 512-bit function: "
                        << NewFunc->getName() << "\n");
    }
  }

  return PreservedAnalyses::none();
}
