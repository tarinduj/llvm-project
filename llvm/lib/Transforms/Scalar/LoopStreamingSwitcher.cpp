//===- LoopStreamingSwitcher.cpp - Per-loop SSVE/NEON switching ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass uses a trained decision tree to decide per-loop whether Streaming
// SVE (SSVE) or NEON is faster. Loops predicted to benefit from SSVE are
// outlined into separate functions marked with aarch64_pstate_sm_body.
//
// The pass runs BEFORE the Loop Vectorizer so that the LV sees the streaming
// attribute on outlined functions and uses scalable SVE vectorization.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Scalar/LoopStreamingSwitcher.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/LoopAccessAnalysis.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Transforms/Utils/CodeExtractor.h"

using namespace llvm;

#define DEBUG_TYPE "loop-streaming-switcher"

STATISTIC(NumOutlined, "Number of loops outlined for streaming SVE");

static cl::opt<bool> AutoStreamingMode(
    "loop-vectorize-auto-streaming", cl::init(false), cl::Hidden,
    cl::desc("Automatically select SSVE vs NEON per-loop using a trained "
             "decision tree. Outlines SSVE-beneficial loops into streaming "
             "functions before the Loop Vectorizer runs."));

//===----------------------------------------------------------------------===//
// Decision Tree Features and Classifier
//===----------------------------------------------------------------------===//

namespace {

/// Features needed by the SSVE/NEON decision tree classifier.
struct StreamingDecisionFeatures {
  int64_t rec_num_backward_deps = 0;
  int64_t rec_num_forward_deps = 0;
  int64_t stride_max_stride_bytes = 0;
  int64_t stride_min_stride_bytes = 0;
  int64_t stride_num_accesses = 0;
  int64_t stride_num_non_unit_stride = 0;
  int64_t lb_final_iv_value = 0;
  double ci_ops_per_memory = 0.0;
  int64_t ci_num_total_ops = 0;
  int64_t ci_num_fp_arith_ops = 0;
  int64_t ci_num_int_arith_ops = 0;
  int64_t nested_parent_trip_count = 0;
  int64_t ap_min_addrec_depth = 0;
  double cost_ratio = 1.0; // Default: no vectorization benefit (conservative)
  int64_t cost_scalar = 0;
  int64_t arith_num_sub = 0;
  int64_t arith_num_add = 0;
  int64_t trip_count_value = 0;
  int64_t dtype_num_f16_accesses = 0;
  int64_t dtype_num_bf16_accesses = 0;
  int64_t dtype_num_f32_accesses = 0;
  int64_t dtype_num_f64_accesses = 0;
  int64_t dtype_num_i8_accesses = 0;
  int64_t dtype_num_i16_accesses = 0;
  int64_t dtype_num_i32_accesses = 0;
  int64_t dtype_num_i64_accesses = 0;
  int64_t dtype_num_ptr_accesses = 0;
  int64_t dtype_max_element_size_bytes = 0;
  int64_t dtype_min_element_size_bytes = 0;
};

/// Trained decision tree: returns true if SSVE is predicted faster than NEON.
/// Tree trained on 287 synthetic benchmarks, 87% CV accuracy, depth 10.
/// Features: 61 LLVM loop features; top splits on stride, recurrence, bounds.
static bool shouldUseStreamingSVE(const StreamingDecisionFeatures &F) {
  if (F.rec_num_backward_deps <= 0) {
    if (F.stride_max_stride_bytes <= 24) {
      if (F.lb_final_iv_value <= 254)
        return false; // NEON
      // lb_final_iv_value > 254
      if (F.ci_ops_per_memory <= 0.63) {
        if (F.nested_parent_trip_count <= 255)
          return true; // SSVE
        // nested_parent_trip_count > 255
        if (F.cost_scalar <= 24)
          return F.ci_num_total_ops <= 14; // SSVE if <=14, NEON if >14
        return true; // cost_scalar > 24 -> SSVE
      }
      // ci_ops_per_memory > 0.63
      if (F.ap_min_addrec_depth <= 1)
        return false; // NEON
      // ap_min_addrec_depth > 1
      if (F.ci_num_fp_arith_ops <= 1)
        return true; // SSVE
      // ci_num_fp_arith_ops > 1
      if (F.nested_parent_trip_count <= 254)
        return F.nested_parent_trip_count <= 250; // SSVE <=250, NEON 250-254
      return true; // >254 -> SSVE
    }
    // stride_max_stride_bytes > 24
    if (F.arith_num_sub <= 0) {
      if (F.ci_ops_per_memory <= 0.36)
        return F.stride_num_accesses <= 72; // SSVE if <=72, NEON if >72
      // ci_ops_per_memory > 0.36
      if (F.stride_num_non_unit_stride <= 3)
        return true; // SSVE
      // stride_num_non_unit_stride > 3
      if (F.ci_num_total_ops <= 15)
        return F.nested_parent_trip_count <= 254; // SSVE <=254, NEON >254
      // ci_num_total_ops > 15 -> SSVE (all deep sub-branches)
      return true;
    }
    // arith_num_sub > 0
    if (F.lb_final_iv_value <= 253)
      return false; // NEON
    return true; // SSVE
  }
  // rec_num_backward_deps > 0
  if (F.cost_ratio <= 0.83)
    return false; // NEON
  if (F.stride_num_non_unit_stride <= 3)
    return false; // NEON
  return true; // SSVE
}

//===----------------------------------------------------------------------===//
// Feature Extraction
//===----------------------------------------------------------------------===//

/// Extract features needed by the decision tree from a loop.
/// This runs BEFORE the Loop Vectorizer, so cost model features (cost_scalar,
/// cost_ratio) are approximated: cost_scalar ≈ ci_num_total_ops,
/// cost_ratio defaults to 1.0 (conservative).
static StreamingDecisionFeatures
extractDecisionFeatures(Loop *L, ScalarEvolution &SE,
                        const LoopAccessInfo &LAI, const DataLayout &DL) {
  StreamingDecisionFeatures F;

  // Trip count
  unsigned SmallTC = SE.getSmallConstantTripCount(L);
  F.trip_count_value = static_cast<int64_t>(SmallTC);

  // Stride info from LoopAccessInfo
  const auto &SymbolicStrides = LAI.getSymbolicStrides();
  const MemoryDepChecker &DepChecker = LAI.getDepChecker();
  const auto &MemInsts = DepChecker.getMemoryInstructions();

  // We need a PSE for getPtrStride
  PredicatedScalarEvolution PSE(SE, *L);

  int64_t MaxStrideBytes = 0, MinStrideBytes = INT64_MAX;
  unsigned NumNonUnitStride = 0;
  unsigned NumF16Accesses = 0, NumBF16Accesses = 0;
  unsigned NumF32Accesses = 0, NumF64Accesses = 0;
  unsigned NumI8Accesses = 0, NumI16Accesses = 0;
  unsigned NumI32Accesses = 0, NumI64Accesses = 0;
  unsigned NumPtrAccesses = 0;
  uint64_t MaxElementSizeBytes = 0, MinElementSizeBytes = UINT64_MAX;

  for (Instruction *I : MemInsts) {
    Type *AccessTy = getLoadStoreType(I);
    uint64_t ElementSizeBytes = DL.getTypeStoreSize(AccessTy);
    Value *Ptr = getLoadStorePointerOperand(I);

    // Per-width dtype tracking
    if (AccessTy->isHalfTy())
      ++NumF16Accesses;
    else if (AccessTy->isBFloatTy())
      ++NumBF16Accesses;
    else if (AccessTy->isFloatTy())
      ++NumF32Accesses;
    else if (AccessTy->isDoubleTy())
      ++NumF64Accesses;
    else if (AccessTy->isIntegerTy(8))
      ++NumI8Accesses;
    else if (AccessTy->isIntegerTy(16))
      ++NumI16Accesses;
    else if (AccessTy->isIntegerTy(32))
      ++NumI32Accesses;
    else if (AccessTy->isIntegerTy(64))
      ++NumI64Accesses;
    else if (AccessTy->isPointerTy())
      ++NumPtrAccesses;
    MaxElementSizeBytes = std::max(MaxElementSizeBytes, ElementSizeBytes);
    MinElementSizeBytes = std::min(MinElementSizeBytes, ElementSizeBytes);

    std::optional<int64_t> StrideOpt =
        getPtrStride(PSE, AccessTy, Ptr, L, SymbolicStrides, false, false);

    if (StrideOpt.has_value()) {
      int64_t StrideBytes =
          std::abs(*StrideOpt) * static_cast<int64_t>(ElementSizeBytes);
      if (StrideBytes != static_cast<int64_t>(ElementSizeBytes))
        ++NumNonUnitStride;
      if (StrideBytes > 0) {
        MaxStrideBytes = std::max(MaxStrideBytes, StrideBytes);
        MinStrideBytes = std::min(MinStrideBytes, StrideBytes);
      }
    }
  }

  F.stride_max_stride_bytes = MaxStrideBytes;
  F.stride_min_stride_bytes =
      (MinStrideBytes == INT64_MAX) ? 0 : MinStrideBytes;
  F.stride_num_accesses = static_cast<int64_t>(MemInsts.size());
  F.stride_num_non_unit_stride = static_cast<int64_t>(NumNonUnitStride);
  F.dtype_num_f16_accesses = static_cast<int64_t>(NumF16Accesses);
  F.dtype_num_bf16_accesses = static_cast<int64_t>(NumBF16Accesses);
  F.dtype_num_f32_accesses = static_cast<int64_t>(NumF32Accesses);
  F.dtype_num_f64_accesses = static_cast<int64_t>(NumF64Accesses);
  F.dtype_num_i8_accesses = static_cast<int64_t>(NumI8Accesses);
  F.dtype_num_i16_accesses = static_cast<int64_t>(NumI16Accesses);
  F.dtype_num_i32_accesses = static_cast<int64_t>(NumI32Accesses);
  F.dtype_num_i64_accesses = static_cast<int64_t>(NumI64Accesses);
  F.dtype_num_ptr_accesses = static_cast<int64_t>(NumPtrAccesses);
  F.dtype_max_element_size_bytes = static_cast<int64_t>(MaxElementSizeBytes);
  F.dtype_min_element_size_bytes =
      static_cast<int64_t>(MinElementSizeBytes == UINT64_MAX
                               ? 0
                               : MinElementSizeBytes);

  // Compute intensity and arithmetic patterns
  unsigned NumIntArithOps = 0, NumFPArithOps = 0, NumMemOps = 0;
  unsigned NumLogicOps = 0, NumCompareOps = 0, NumTotalOps = 0;
  unsigned NumAdd = 0, NumSub = 0;

  for (BasicBlock *BB : L->blocks()) {
    for (Instruction &I : *BB) {
      ++NumTotalOps;
      switch (I.getOpcode()) {
      case Instruction::Add:
        ++NumIntArithOps;
        ++NumAdd;
        break;
      case Instruction::Sub:
        ++NumIntArithOps;
        ++NumSub;
        break;
      case Instruction::Mul:
      case Instruction::UDiv:
      case Instruction::SDiv:
      case Instruction::URem:
      case Instruction::SRem:
      case Instruction::Shl:
      case Instruction::LShr:
      case Instruction::AShr:
        ++NumIntArithOps;
        break;
      case Instruction::FAdd:
        ++NumFPArithOps;
        ++NumAdd;
        break;
      case Instruction::FSub:
        ++NumFPArithOps;
        ++NumSub;
        break;
      case Instruction::FMul:
      case Instruction::FDiv:
      case Instruction::FRem:
      case Instruction::FNeg:
        ++NumFPArithOps;
        break;
      case Instruction::And:
      case Instruction::Or:
      case Instruction::Xor:
        ++NumLogicOps;
        break;
      case Instruction::Load:
      case Instruction::Store:
        ++NumMemOps;
        break;
      case Instruction::ICmp:
      case Instruction::FCmp:
        ++NumCompareOps;
        break;
      default:
        break;
      }
    }
  }

  F.ci_num_total_ops = static_cast<int64_t>(NumTotalOps);
  F.ci_num_fp_arith_ops = static_cast<int64_t>(NumFPArithOps);
  F.ci_num_int_arith_ops = static_cast<int64_t>(NumIntArithOps);
  unsigned TotalComputeOps =
      NumIntArithOps + NumFPArithOps + NumLogicOps + NumCompareOps;
  F.ci_ops_per_memory =
      NumMemOps > 0
          ? static_cast<double>(TotalComputeOps) /
                static_cast<double>(NumMemOps)
          : 0.0;
  F.arith_num_sub = static_cast<int64_t>(NumSub);
  F.arith_num_add = static_cast<int64_t>(NumAdd);

  // Approximate cost_scalar with instruction count (LV cost model unavailable)
  F.cost_scalar = F.ci_num_total_ops;
  // cost_ratio defaults to 1.0 (no vectorization benefit assumed)

  // Nested loop info
  if (Loop *Parent = L->getParentLoop()) {
    unsigned ParentTC = SE.getSmallConstantTripCount(Parent);
    F.nested_parent_trip_count = static_cast<int64_t>(ParentTC);
  }

  // Loop bounds
  if (auto Bounds = L->getBounds(SE)) {
    if (const SCEVConstant *FinalConst =
            dyn_cast<SCEVConstant>(SE.getSCEV(&Bounds->getFinalIVValue())))
      F.lb_final_iv_value = FinalConst->getAPInt().getSExtValue();
  }

  // Recurrence info from dependency checker
  const auto *Deps = DepChecker.getDependences();
  if (Deps) {
    for (const auto &Dep : *Deps) {
      switch (Dep.Type) {
      case MemoryDepChecker::Dependence::Backward:
      case MemoryDepChecker::Dependence::BackwardVectorizable:
      case MemoryDepChecker::Dependence::BackwardVectorizableButPreventsForwarding:
        ++F.rec_num_backward_deps;
        break;
      case MemoryDepChecker::Dependence::Forward:
      case MemoryDepChecker::Dependence::ForwardButPreventsForwarding:
        ++F.rec_num_forward_deps;
        break;
      default:
        break;
      }
    }
  }

  // Access pattern info (SCEV AddRec depth)
  unsigned MinAddRecDepth = UINT_MAX;
  for (Instruction *I : MemInsts) {
    Value *Ptr = getLoadStorePointerOperand(I);
    if (!Ptr)
      continue;
    const SCEV *PtrSCEV = SE.getSCEV(Ptr);
    unsigned Depth = 0;
    const SCEV *Current = PtrSCEV;
    while (const SCEVAddRecExpr *AR = dyn_cast<SCEVAddRecExpr>(Current)) {
      ++Depth;
      Current = AR->getStart();
    }
    if (Depth > 0)
      MinAddRecDepth = std::min(MinAddRecDepth, Depth);
  }
  F.ap_min_addrec_depth =
      static_cast<int64_t>(MinAddRecDepth == UINT_MAX ? 0 : MinAddRecDepth);

  return F;
}

//===----------------------------------------------------------------------===//
// Loop Outlining
//===----------------------------------------------------------------------===//

/// Outline a loop into a new function marked with aarch64_pstate_sm_body.
/// Returns the new function, or nullptr on failure.
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

  // Mark the outlined function as locally streaming
  NewFunc->addFnAttr("aarch64_pstate_sm_body");

  // Copy target-features from parent and ensure +sme is present
  if (ParentFunc.hasFnAttribute("target-features")) {
    StringRef Features =
        ParentFunc.getFnAttribute("target-features").getValueAsString();
    if (Features.contains("+sme")) {
      NewFunc->addFnAttr("target-features", Features);
    } else {
      // Append +sme to existing features
      std::string NewFeatures = (Features + ",+sme").str();
      NewFunc->addFnAttr("target-features", NewFeatures);
    }
  } else {
    NewFunc->addFnAttr("target-features", "+sme");
  }

  // Copy target-cpu if present
  if (ParentFunc.hasFnAttribute("target-cpu"))
    NewFunc->addFnAttr(ParentFunc.getFnAttribute("target-cpu"));

  // Don't try to update LoopInfo — we'll invalidate all analyses.

  ++NumOutlined;
  return NewFunc;
}

} // anonymous namespace

//===----------------------------------------------------------------------===//
// Pass Implementation
//===----------------------------------------------------------------------===//

PreservedAnalyses LoopStreamingSwitcherPass::run(Function &F,
                                                  FunctionAnalysisManager &AM) {
  if (!AutoStreamingMode)
    return PreservedAnalyses::all();

  // Skip functions already in streaming mode
  if (F.hasFnAttribute("aarch64_pstate_sm_enabled") ||
      F.hasFnAttribute("aarch64_pstate_sm_body"))
    return PreservedAnalyses::all();

  // Skip optnone functions
  if (F.hasOptNone())
    return PreservedAnalyses::all();

  if (F.empty())
    return PreservedAnalyses::all();

  auto &LI = AM.getResult<LoopAnalysis>(F);
  if (LI.empty())
    return PreservedAnalyses::all();

  auto &SE = AM.getResult<ScalarEvolutionAnalysis>(F);
  auto &DT = AM.getResult<DominatorTreeAnalysis>(F);
  auto *AC = AM.getCachedResult<AssumptionAnalysis>(F);
  auto &LAIs = AM.getResult<LoopAccessAnalysis>(F);
  const DataLayout &DL = F.getDataLayout();

  // Find the first innermost loop that should use SSVE and outline it.
  // We only outline ONE loop per pass invocation, then return
  // PreservedAnalyses::none() to force all analyses to be recomputed.
  // The pass manager will re-run this pass on the modified function,
  // picking up the next candidate loop with fresh analyses.
  //
  // This avoids the problem of stale Loop*/SE/DT pointers after
  // CodeExtractor restructures the function.
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

  for (Loop *L : InnermostLoops) {
    // Get LoopAccessInfo for this loop
    const LoopAccessInfo &LAI = LAIs.getInfo(*L);

    // Extract features and evaluate decision tree
    StreamingDecisionFeatures Features =
        extractDecisionFeatures(L, SE, LAI, DL);

    if (!shouldUseStreamingSVE(Features)) {
      LLVM_DEBUG(dbgs() << "LSS: Loop stays NEON: " << L->getLocStr() << "\n");
      continue;
    }

    LLVM_DEBUG(dbgs() << "LSS: Decision tree predicts SSVE for loop: "
                      << L->getLocStr() << "\n");

    // Outline the loop into a streaming function
    Function *NewFunc = outlineLoopForStreaming(L, DT, AC, F);
    if (NewFunc) {
      LLVM_DEBUG(dbgs() << "LSS: Outlined loop into streaming function: "
                        << NewFunc->getName() << "\n");
      // Invalidate ALL analyses — CodeExtractor restructures the function.
      // The pass manager will re-run this pass to find more candidates.
      return PreservedAnalyses::none();
    }
  }

  return PreservedAnalyses::all();
}
