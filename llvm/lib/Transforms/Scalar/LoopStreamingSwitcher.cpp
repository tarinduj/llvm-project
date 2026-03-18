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
#include "llvm/ADT/SmallPtrSet.h"
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
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/CodeExtractor.h"

using namespace llvm;

#define DEBUG_TYPE "loop-streaming-switcher"

STATISTIC(NumOutlined, "Number of loops outlined for streaming SVE");

static cl::opt<bool> AutoStreamingMode(
    "loop-vectorize-auto-streaming", cl::init(false), cl::Hidden,
    cl::desc("Automatically select SSVE vs NEON per-loop using a trained "
             "decision tree. Outlines SSVE-beneficial loops into streaming "
             "functions before the Loop Vectorizer runs."));

static cl::opt<std::string> SwitcherFeatureOutput(
    "loop-streaming-switcher-feature-output", cl::init(""), cl::Hidden,
    cl::desc("Output file for pre-LV loop features (JSON). "
             "Dumps features from the LoopStreamingSwitcher pass."));

//===----------------------------------------------------------------------===//
// Decision Tree Features and Classifier
//===----------------------------------------------------------------------===//

namespace {

/// Features needed by the SSVE/NEON decision tree classifier.
/// Mirrors the feature set from LoopVectorize.cpp JSON output,
/// excluding post-LV cost model features (cost_vector, vf_width,
/// vectorized, etc.) which are unavailable before the Loop Vectorizer.
struct StreamingDecisionFeatures {
  // Core loop features
  int64_t loop_depth = 0;
  int64_t num_blocks = 0;
  int64_t trip_count_value = 0;
  bool trip_count_is_constant = false;

  // Memory
  int64_t num_loads = 0;
  int64_t num_stores = 0;
  int64_t num_symbolic_strides = 0;
  int64_t max_safe_width_bits = 0;
  bool safe_for_any_width = false;

  // Inductions / reductions
  int64_t induction_count = 0;
  bool has_primary_induction = false;
  int64_t reduction_count = 0;

  // Compute intensity
  int64_t ci_num_int_arith_ops = 0;
  int64_t ci_num_fp_arith_ops = 0;
  int64_t ci_num_logic_ops = 0;
  int64_t ci_num_memory_ops = 0;
  int64_t ci_num_compare_ops = 0;
  int64_t ci_num_conversion_ops = 0;
  int64_t ci_num_call_ops = 0;
  int64_t ci_num_total_ops = 0;
  double ci_ops_per_memory = 0.0;

  // Dependencies
  int64_t dep_num_dependences = 0;
  bool dep_has_loop_carried_deps = false;
  bool dep_has_backward_deps = false;
  bool dep_has_forward_deps = false;
  bool dep_has_unknown_deps = false;
  bool dep_has_indirect_unsafe = false;
  bool dep_safe_for_vectorization = false;
  bool dep_safe_for_any_width = false;

  // Stride summary
  int64_t stride_num_unit_stride = 0;
  int64_t stride_num_non_unit_stride = 0;
  int64_t stride_num_accesses = 0;
  int64_t stride_num_unique_base_ptrs = 0;
  bool stride_has_non_affine = false;
  int64_t stride_min_stride_bytes = 0;
  int64_t stride_max_stride_bytes = 0;

  // Nested loop info
  bool nested_has_parent = false;
  int64_t nested_loop_depth = 0;
  int64_t nested_num_sub_loops = 0;
  int64_t nested_parent_trip_count = 0;
  bool nested_parent_trip_count_is_constant = false;

  // Access pattern
  bool ap_has_mixed_depth = false;
  int64_t ap_max_addrec_depth = 0;
  int64_t ap_min_addrec_depth = 0;
  int64_t ap_num_unique_arrays = 0;

  // Arithmetic pattern
  bool arith_has_multiply_add = false;
  int64_t arith_num_add = 0;
  int64_t arith_num_sub = 0;
  int64_t arith_num_mul = 0;
  int64_t arith_num_div = 0;
  int64_t arith_num_fma_patterns = 0;

  // Loop bounds
  int64_t lb_initial_iv_value = 0;
  int64_t lb_final_iv_value = 0;
  int64_t lb_step_value = 1;

  // Recurrence info
  bool rec_has_recurrence = false;
  int64_t rec_num_backward_deps = 0;
  int64_t rec_num_forward_deps = 0;

  // Dtype summary
  int64_t dtype_max_element_size_bytes = 0;
  int64_t dtype_min_element_size_bytes = 0;
  int64_t dtype_num_f32_accesses = 0;
  int64_t dtype_num_f64_accesses = 0;
  int64_t dtype_num_i8_accesses = 0;
  int64_t dtype_num_i16_accesses = 0;
  int64_t dtype_num_i32_accesses = 0;
  int64_t dtype_num_i64_accesses = 0;
  int64_t dtype_num_f16_accesses = 0;
  int64_t dtype_num_bf16_accesses = 0;
  int64_t dtype_num_ptr_accesses = 0;

  // Cost model features — unavailable pre-LV, use defaults
  int64_t cost_scalar = 0;   // Approximated as ci_num_total_ops
  int64_t cost_vector = 0;   // Approximated as cost_scalar
  int64_t vf_width = 1;      // Default: no vectorization
  bool vf_is_scalable = false;
  int64_t interleave_count = 1;
  bool vectorized = false;
  double cost_ratio = 1.0;   // Default: no benefit
};

/// Auto-generated from decision tree (depth=10, 21 leaves, 17 features)
/// Features used: ci_num_compare_ops, ci_num_int_arith_ops, ci_num_memory_ops,
/// ci_num_total_ops, ci_ops_per_memory, cost_vector, dep_num_dependences,
/// dtype_num_i16_accesses, dtype_num_i8_accesses, has_primary_induction,
/// lb_final_iv_value, nested_parent_trip_count, stride_num_non_unit_stride,
/// stride_num_unit_stride, trip_count_value, vectorized, vf_width
/// Classes: 0 = NEON, 1 = SSVE
static bool shouldUseStreamingSVE(const StreamingDecisionFeatures &F) {
  if (F.stride_num_unit_stride <= 32) {
    if (F.ci_num_memory_ops <= 76) {
      if (F.ci_ops_per_memory <= 0.387500) {
        return true; // SSVE (87%)
      } else {
        if (F.trip_count_value <= 2043) {
          if (F.ci_num_compare_ops <= 1) {
            if (F.nested_parent_trip_count <= 250000) {
              if (F.trip_count_value <= 106) {
                return false; // NEON (62%)
              } else {
                if (F.dtype_num_i16_accesses <= 2) {
                  if (F.stride_num_non_unit_stride <= 0) {
                    return false; // NEON (100%)
                  } else {
                    if (F.cost_vector <= 14) {
                      return false; // NEON (75%)
                    } else {
                      return false; // NEON (100%)
                    }
                  }
                } else {
                  return false; // NEON (67%)
                }
              }
            } else {
              return false; // NEON (100%)
            }
          } else {
            return false; // NEON (51%)
          }
        } else {
          if (F.vectorized <= 0) {
            if (F.ci_num_total_ops <= 15) {
              return true; // SSVE (69%)
            } else {
              if (F.lb_final_iv_value <= 23997) {
                return false; // NEON (100%)
              } else {
                return false; // NEON (78%)
              }
            }
          } else {
            return true; // SSVE (84%)
          }
        }
      }
    } else {
      return false; // NEON (100%)
    }
  } else {
    if (F.vf_width <= 1) {
      return false; // NEON (100%)
    } else {
      if (F.has_primary_induction <= 0) {
        if (F.ci_num_int_arith_ops <= 47) {
          return false; // NEON (100%)
        } else {
          return true; // SSVE (73%)
        }
      } else {
        if (F.dtype_num_i8_accesses <= 17) {
          if (F.ci_num_total_ops <= 173) {
            return true; // SSVE (76%)
          } else {
            if (F.dep_num_dependences <= 8) {
              return true; // SSVE (92%)
            } else {
              return true; // SSVE (100%)
            }
          }
        } else {
          return false; // NEON (63%)
        }
      }
    }
  }
}

//===----------------------------------------------------------------------===//
// Feature Extraction
//===----------------------------------------------------------------------===//

/// Extract features needed by the decision tree from a loop.
/// This runs BEFORE the Loop Vectorizer, so cost model features (cost_vector,
/// vf_width, vectorized) are set to defaults. All other features mirror
/// what LoopVectorize.cpp extracts into JSON.
static StreamingDecisionFeatures
extractDecisionFeatures(Loop *L, ScalarEvolution &SE,
                        const LoopAccessInfo &LAI, const DataLayout &DL) {
  StreamingDecisionFeatures F;

  // --- Core loop features ---
  F.loop_depth = static_cast<int64_t>(L->getLoopDepth());
  F.num_blocks = static_cast<int64_t>(L->getNumBlocks());
  unsigned SmallTC = SE.getSmallConstantTripCount(L);
  F.trip_count_value = static_cast<int64_t>(SmallTC);
  F.trip_count_is_constant = (SmallTC > 0);

  // --- Memory ---
  F.num_loads = static_cast<int64_t>(LAI.getNumLoads());
  F.num_stores = static_cast<int64_t>(LAI.getNumStores());
  const auto &SymbolicStrides = LAI.getSymbolicStrides();
  F.num_symbolic_strides = static_cast<int64_t>(SymbolicStrides.size());
  const MemoryDepChecker &DepChecker = LAI.getDepChecker();
  F.max_safe_width_bits =
      static_cast<int64_t>(DepChecker.getMaxSafeVectorWidthInBits());
  F.safe_for_any_width = DepChecker.isSafeForAnyVectorWidth();

  // --- Inductions / reductions (simplified, no LVL available) ---
  // Matches LV's primary induction: integer type, step=1, start=0,
  // and must be the widest integer induction type.
  BasicBlock *Header = L->getHeader();
  unsigned NumInductions = 0;
  bool HasPrimaryInduction = false;
  unsigned NumReductions = 0;
  PHINode *PrimaryCandidate = nullptr;
  unsigned WidestIndBits = 0;

  for (PHINode &Phi : Header->phis()) {
    const SCEV *PhiSCEV = SE.getSCEV(&Phi);
    if (const SCEVAddRecExpr *AR = dyn_cast<SCEVAddRecExpr>(PhiSCEV)) {
      if (AR->getLoop() == L) {
        ++NumInductions;
        // Check for primary: integer, step=1, start=0
        Type *PhiTy = Phi.getType();
        if (PhiTy->isIntegerTy() && AR->isAffine()) {
          if (const SCEVConstant *Start =
                  dyn_cast<SCEVConstant>(AR->getStart())) {
            if (const SCEVConstant *Step =
                    dyn_cast<SCEVConstant>(AR->getStepRecurrence(SE))) {
              if (Start->getAPInt().isZero() && Step->getAPInt().isOne()) {
                unsigned Bits = PhiTy->getIntegerBitWidth();
                if (Bits >= WidestIndBits) {
                  WidestIndBits = Bits;
                  PrimaryCandidate = &Phi;
                }
              }
            }
          }
        }
      }
    } else {
      // Non-AddRec PHI in header — not an induction
      ++NumReductions;
    }
  }
  // LV clears primary if it's not the widest int induction type.
  // We approximate: if we found a candidate, accept it.
  HasPrimaryInduction = (PrimaryCandidate != nullptr);
  F.induction_count = static_cast<int64_t>(NumInductions);
  F.has_primary_induction = HasPrimaryInduction;
  F.reduction_count = static_cast<int64_t>(NumReductions);

  // --- Stride, dtype, and base pointer info ---
  const auto &MemInsts = DepChecker.getMemoryInstructions();
  PredicatedScalarEvolution PSE(SE, *L);

  int64_t MaxStrideBytes = 0, MinStrideBytes = INT64_MAX;
  unsigned NumUnitStride = 0, NumNonUnitStride = 0, NumNonAffine = 0;
  unsigned NumF16Accesses = 0, NumBF16Accesses = 0;
  unsigned NumF32Accesses = 0, NumF64Accesses = 0;
  unsigned NumI8Accesses = 0, NumI16Accesses = 0;
  unsigned NumI32Accesses = 0, NumI64Accesses = 0;
  unsigned NumPtrAccesses = 0;
  uint64_t MaxElementSizeBytes = 0, MinElementSizeBytes = UINT64_MAX;
  DenseMap<const Value *, int> BasePtrToId;
  int NextBasePtrId = 0;

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

    // Base pointer tracking
    const SCEV *PtrSCEV = SE.getSCEV(Ptr);
    const SCEV *BaseSCEV = SE.getPointerBase(PtrSCEV);
    if (const SCEVUnknown *BaseUnknown = dyn_cast<SCEVUnknown>(BaseSCEV)) {
      const Value *BasePtr = BaseUnknown->getValue();
      if (BasePtrToId.find(BasePtr) == BasePtrToId.end())
        BasePtrToId[BasePtr] = NextBasePtrId++;
    }

    // Stride analysis
    std::optional<int64_t> StrideOpt =
        getPtrStride(PSE, AccessTy, Ptr, L, SymbolicStrides, false, false);

    if (StrideOpt.has_value()) {
      int64_t StrideBytes =
          std::abs(*StrideOpt) * static_cast<int64_t>(ElementSizeBytes);
      if (StrideBytes == static_cast<int64_t>(ElementSizeBytes))
        ++NumUnitStride;
      else
        ++NumNonUnitStride;
      if (StrideBytes > 0) {
        MaxStrideBytes = std::max(MaxStrideBytes, StrideBytes);
        MinStrideBytes = std::min(MinStrideBytes, StrideBytes);
      }
    } else {
      ++NumNonAffine;
    }
  }

  F.stride_num_unit_stride = static_cast<int64_t>(NumUnitStride);
  F.stride_num_non_unit_stride = static_cast<int64_t>(NumNonUnitStride);
  F.stride_num_accesses = static_cast<int64_t>(MemInsts.size());
  F.stride_num_unique_base_ptrs = static_cast<int64_t>(NextBasePtrId);
  F.stride_has_non_affine = NumNonAffine > 0;
  F.stride_max_stride_bytes = MaxStrideBytes;
  F.stride_min_stride_bytes =
      (MinStrideBytes == INT64_MAX) ? 0 : MinStrideBytes;
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

  // --- Compute intensity and arithmetic patterns ---
  unsigned NumIntArithOps = 0, NumFPArithOps = 0, NumMemOps = 0;
  unsigned NumLogicOps = 0, NumCompareOps = 0, NumConversionOps = 0;
  unsigned NumCallOps = 0, NumTotalOps = 0;
  unsigned NumAdd = 0, NumSub = 0, NumMul = 0, NumDiv = 0;
  unsigned NumFMA = 0;
  bool HasMultiplyAdd = false;

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
        ++NumIntArithOps;
        ++NumMul;
        break;
      case Instruction::UDiv:
      case Instruction::SDiv:
        ++NumIntArithOps;
        ++NumDiv;
        break;
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
        // Check for FMA: fadd(fmul(...), ...)
        if (isa<BinaryOperator>(I)) {
          if (isa<BinaryOperator>(I.getOperand(0)) &&
              cast<BinaryOperator>(I.getOperand(0))->getOpcode() ==
                  Instruction::FMul) {
            ++NumFMA;
            HasMultiplyAdd = true;
          } else if (isa<BinaryOperator>(I.getOperand(1)) &&
                     cast<BinaryOperator>(I.getOperand(1))->getOpcode() ==
                         Instruction::FMul) {
            ++NumFMA;
            HasMultiplyAdd = true;
          }
        }
        break;
      case Instruction::FSub:
        ++NumFPArithOps;
        ++NumSub;
        break;
      case Instruction::FMul:
        ++NumFPArithOps;
        ++NumMul;
        break;
      case Instruction::FDiv:
      case Instruction::FRem:
        ++NumFPArithOps;
        ++NumDiv;
        break;
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
      case Instruction::Trunc:
      case Instruction::ZExt:
      case Instruction::SExt:
      case Instruction::FPTrunc:
      case Instruction::FPExt:
      case Instruction::UIToFP:
      case Instruction::SIToFP:
      case Instruction::FPToUI:
      case Instruction::FPToSI:
      case Instruction::IntToPtr:
      case Instruction::PtrToInt:
      case Instruction::BitCast:
        ++NumConversionOps;
        break;
      case Instruction::Call:
        ++NumCallOps;
        break;
      default:
        break;
      }
    }
  }

  F.ci_num_int_arith_ops = static_cast<int64_t>(NumIntArithOps);
  F.ci_num_fp_arith_ops = static_cast<int64_t>(NumFPArithOps);
  F.ci_num_logic_ops = static_cast<int64_t>(NumLogicOps);
  F.ci_num_memory_ops = static_cast<int64_t>(NumMemOps);
  F.ci_num_compare_ops = static_cast<int64_t>(NumCompareOps);
  F.ci_num_conversion_ops = static_cast<int64_t>(NumConversionOps);
  F.ci_num_call_ops = static_cast<int64_t>(NumCallOps);
  F.ci_num_total_ops = static_cast<int64_t>(NumTotalOps);
  unsigned TotalComputeOps =
      NumIntArithOps + NumFPArithOps + NumLogicOps + NumCompareOps;
  F.ci_ops_per_memory =
      NumMemOps > 0
          ? static_cast<double>(TotalComputeOps) /
                static_cast<double>(NumMemOps)
          : 0.0;
  F.arith_num_add = static_cast<int64_t>(NumAdd);
  F.arith_num_sub = static_cast<int64_t>(NumSub);
  F.arith_num_mul = static_cast<int64_t>(NumMul);
  F.arith_num_div = static_cast<int64_t>(NumDiv);
  F.arith_num_fma_patterns = static_cast<int64_t>(NumFMA);
  F.arith_has_multiply_add = HasMultiplyAdd;

  // Approximate cost model features
  F.cost_scalar = F.ci_num_total_ops;
  F.cost_vector = F.ci_num_total_ops; // Assume no vectorization benefit
  // vf_width=1, vectorized=false, cost_ratio=1.0 already set as defaults

  // --- Nested loop info ---
  F.nested_loop_depth = static_cast<int64_t>(L->getLoopDepth());
  F.nested_num_sub_loops = static_cast<int64_t>(L->getSubLoops().size());
  if (Loop *Parent = L->getParentLoop()) {
    F.nested_has_parent = true;
    unsigned ParentTC = SE.getSmallConstantTripCount(Parent);
    F.nested_parent_trip_count = static_cast<int64_t>(ParentTC);
    F.nested_parent_trip_count_is_constant = (ParentTC > 0);
  }

  // --- Loop bounds ---
  if (auto Bounds = L->getBounds(SE)) {
    if (const SCEVConstant *InitConst =
            dyn_cast<SCEVConstant>(SE.getSCEV(&Bounds->getInitialIVValue())))
      F.lb_initial_iv_value = InitConst->getAPInt().getSExtValue();
    if (const SCEVConstant *FinalConst =
            dyn_cast<SCEVConstant>(SE.getSCEV(&Bounds->getFinalIVValue())))
      F.lb_final_iv_value = FinalConst->getAPInt().getSExtValue();
    if (const SCEVConstant *StepConst =
            dyn_cast<SCEVConstant>(SE.getSCEV(Bounds->getStepValue())))
      F.lb_step_value = StepConst->getAPInt().getSExtValue();
  }

  // --- Dependencies ---
  const auto *Deps = DepChecker.getDependences();
  F.dep_safe_for_vectorization = DepChecker.isSafeForVectorization();
  F.dep_safe_for_any_width = DepChecker.isSafeForAnyVectorWidth();
  if (Deps) {
    F.dep_num_dependences = static_cast<int64_t>(Deps->size());
    for (const auto &Dep : *Deps) {
      switch (Dep.Type) {
      case MemoryDepChecker::Dependence::Backward:
      case MemoryDepChecker::Dependence::BackwardVectorizable:
      case MemoryDepChecker::Dependence::BackwardVectorizableButPreventsForwarding:
        F.dep_has_backward_deps = true;
        F.dep_has_loop_carried_deps = true;
        ++F.rec_num_backward_deps;
        break;
      case MemoryDepChecker::Dependence::Forward:
      case MemoryDepChecker::Dependence::ForwardButPreventsForwarding:
        F.dep_has_forward_deps = true;
        ++F.rec_num_forward_deps;
        break;
      case MemoryDepChecker::Dependence::Unknown:
        F.dep_has_unknown_deps = true;
        F.dep_has_loop_carried_deps = true;
        break;
      case MemoryDepChecker::Dependence::IndirectUnsafe:
        F.dep_has_indirect_unsafe = true;
        F.dep_has_loop_carried_deps = true;
        break;
      default:
        break;
      }
    }
  }
  F.rec_has_recurrence =
      F.rec_num_backward_deps > 0 || F.rec_num_forward_deps > 0;

  // --- Access pattern info (SCEV AddRec depth) ---
  unsigned MaxAddRecDepth = 0, MinAddRecDepth = UINT_MAX;
  SmallPtrSet<const Value *, 8> BasePointers;

  for (Instruction *I : MemInsts) {
    Value *Ptr = getLoadStorePointerOperand(I);
    if (!Ptr)
      continue;
    const SCEV *PtrSCEV = SE.getSCEV(Ptr);

    // Base pointer for unique array count
    const SCEV *Base = SE.getPointerBase(PtrSCEV);
    if (const SCEVUnknown *BaseUnknown = dyn_cast<SCEVUnknown>(Base))
      BasePointers.insert(BaseUnknown->getValue());

    // AddRec nesting depth
    unsigned Depth = 0;
    const SCEV *Current = PtrSCEV;
    while (const SCEVAddRecExpr *AR = dyn_cast<SCEVAddRecExpr>(Current)) {
      ++Depth;
      Current = AR->getStart();
    }
    if (Depth > 0) {
      MaxAddRecDepth = std::max(MaxAddRecDepth, Depth);
      MinAddRecDepth = std::min(MinAddRecDepth, Depth);
    }
  }
  F.ap_max_addrec_depth = static_cast<int64_t>(MaxAddRecDepth);
  F.ap_min_addrec_depth =
      static_cast<int64_t>(MinAddRecDepth == UINT_MAX ? 0 : MinAddRecDepth);
  F.ap_has_mixed_depth =
      (MaxAddRecDepth != MinAddRecDepth) && (MinAddRecDepth != UINT_MAX);
  F.ap_num_unique_arrays = static_cast<int64_t>(BasePointers.size());

  return F;
}

/// Convert StreamingDecisionFeatures to a JSON object for dumping.
static json::Object featuresToJson(const StreamingDecisionFeatures &F) {
  json::Object J;
  J["loop_depth"] = F.loop_depth;
  J["num_blocks"] = F.num_blocks;
  J["trip_count_value"] = F.trip_count_value;
  J["trip_count_is_constant"] = F.trip_count_is_constant;
  J["num_loads"] = F.num_loads;
  J["num_stores"] = F.num_stores;
  J["num_symbolic_strides"] = F.num_symbolic_strides;
  J["max_safe_width_bits"] = F.max_safe_width_bits;
  J["safe_for_any_width"] = F.safe_for_any_width;
  J["induction_count"] = F.induction_count;
  J["has_primary_induction"] = F.has_primary_induction;
  J["ci_num_int_arith_ops"] = F.ci_num_int_arith_ops;
  J["ci_num_fp_arith_ops"] = F.ci_num_fp_arith_ops;
  J["ci_num_logic_ops"] = F.ci_num_logic_ops;
  J["ci_num_memory_ops"] = F.ci_num_memory_ops;
  J["ci_num_compare_ops"] = F.ci_num_compare_ops;
  J["ci_num_conversion_ops"] = F.ci_num_conversion_ops;
  J["ci_num_call_ops"] = F.ci_num_call_ops;
  J["ci_num_total_ops"] = F.ci_num_total_ops;
  J["ci_ops_per_memory"] = F.ci_ops_per_memory;
  J["dep_num_dependences"] = F.dep_num_dependences;
  J["dep_has_loop_carried_deps"] = F.dep_has_loop_carried_deps;
  J["dep_has_backward_deps"] = F.dep_has_backward_deps;
  J["dep_has_forward_deps"] = F.dep_has_forward_deps;
  J["dep_has_unknown_deps"] = F.dep_has_unknown_deps;
  J["dep_has_indirect_unsafe"] = F.dep_has_indirect_unsafe;
  J["dep_safe_for_vectorization"] = F.dep_safe_for_vectorization;
  J["dep_safe_for_any_width"] = F.dep_safe_for_any_width;
  J["stride_num_unit_stride"] = F.stride_num_unit_stride;
  J["stride_num_non_unit_stride"] = F.stride_num_non_unit_stride;
  J["stride_num_accesses"] = F.stride_num_accesses;
  J["stride_num_unique_base_ptrs"] = F.stride_num_unique_base_ptrs;
  J["stride_has_non_affine"] = F.stride_has_non_affine;
  J["stride_min_stride_bytes"] = F.stride_min_stride_bytes;
  J["stride_max_stride_bytes"] = F.stride_max_stride_bytes;
  J["nested_has_parent"] = F.nested_has_parent;
  J["nested_loop_depth"] = F.nested_loop_depth;
  J["nested_num_sub_loops"] = F.nested_num_sub_loops;
  J["nested_parent_trip_count"] = F.nested_parent_trip_count;
  J["nested_parent_trip_count_is_constant"] =
      F.nested_parent_trip_count_is_constant;
  J["ap_has_mixed_depth"] = F.ap_has_mixed_depth;
  J["ap_max_addrec_depth"] = F.ap_max_addrec_depth;
  J["ap_min_addrec_depth"] = F.ap_min_addrec_depth;
  J["ap_num_unique_arrays"] = F.ap_num_unique_arrays;
  J["arith_has_multiply_add"] = F.arith_has_multiply_add;
  J["arith_num_add"] = F.arith_num_add;
  J["arith_num_sub"] = F.arith_num_sub;
  J["arith_num_mul"] = F.arith_num_mul;
  J["arith_num_div"] = F.arith_num_div;
  J["arith_num_fma_patterns"] = F.arith_num_fma_patterns;
  J["lb_initial_iv_value"] = F.lb_initial_iv_value;
  J["lb_final_iv_value"] = F.lb_final_iv_value;
  J["lb_step_value"] = F.lb_step_value;
  J["rec_num_backward_deps"] = F.rec_num_backward_deps;
  J["rec_num_forward_deps"] = F.rec_num_forward_deps;
  J["dtype_max_element_size_bytes"] = F.dtype_max_element_size_bytes;
  J["dtype_min_element_size_bytes"] = F.dtype_min_element_size_bytes;
  J["dtype_num_f32_accesses"] = F.dtype_num_f32_accesses;
  J["dtype_num_f64_accesses"] = F.dtype_num_f64_accesses;
  J["dtype_num_i8_accesses"] = F.dtype_num_i8_accesses;
  J["dtype_num_i16_accesses"] = F.dtype_num_i16_accesses;
  J["dtype_num_i32_accesses"] = F.dtype_num_i32_accesses;
  J["dtype_num_i64_accesses"] = F.dtype_num_i64_accesses;
  J["dtype_num_f16_accesses"] = F.dtype_num_f16_accesses;
  J["dtype_num_bf16_accesses"] = F.dtype_num_bf16_accesses;
  J["dtype_num_ptr_accesses"] = F.dtype_num_ptr_accesses;
  return J;
}

/// Global feature accumulator for JSON output
static json::Object *GlobalSwitcherFeatures = nullptr;

static void writeSwitcherFeatures() {
  if (!GlobalSwitcherFeatures || SwitcherFeatureOutput.empty())
    return;
  std::error_code EC;
  raw_fd_ostream OS(SwitcherFeatureOutput, EC);
  if (EC) {
    errs() << "Error opening switcher feature output: " << EC.message() << "\n";
    return;
  }
  json::Object Root;
  Root["functions"] = std::move(*GlobalSwitcherFeatures);
  OS << json::Value(std::move(Root)) << "\n";
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
  // Initialize feature dumper on first use
  bool DumpFeatures = !SwitcherFeatureOutput.empty();
  if (DumpFeatures && !GlobalSwitcherFeatures) {
    GlobalSwitcherFeatures = new json::Object();
    static bool RegisteredCleanup = false;
    if (!RegisteredCleanup) {
      std::atexit(writeSwitcherFeatures);
      RegisteredCleanup = true;
    }
  }

  if (!AutoStreamingMode && !DumpFeatures)
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

    // Dump features if requested — accumulate loops per function
    if (DumpFeatures && GlobalSwitcherFeatures) {
      std::string FuncName = F.getName().str();
      json::Object LoopObj = featuresToJson(Features);
      LoopObj["location"] = L->getLocStr();

      // Append to existing function entry, or create new one
      if (json::Value *Existing = GlobalSwitcherFeatures->get(FuncName)) {
        if (json::Object *ExObj = Existing->getAsObject()) {
          if (json::Array *Loops = ExObj->getArray("loops")) {
            Loops->push_back(std::move(LoopObj));
          }
        }
      } else {
        json::Object FuncData;
        json::Array LoopArray;
        LoopArray.push_back(std::move(LoopObj));
        FuncData["loops"] = std::move(LoopArray);
        (*GlobalSwitcherFeatures)[FuncName] = std::move(FuncData);
      }
    }

    if (!AutoStreamingMode)
      continue;

    // Skip loops with reductions — scalable SVE cannot expand reductions
    // and will crash the backend with "Expanding reductions for scalable
    // vectors is undefined".
    if (Features.reduction_count > 0) {
      LLVM_DEBUG(dbgs() << "LSS: Skipping loop with reductions: "
                        << L->getLocStr() << "\n");
      continue;
    }

    // Skip loops with non-intrinsic calls — real function calls (memcpy,
    // printf, user functions) may not be streaming-compatible.
    // LLVM intrinsics (fma, sqrt, etc.) are fine in streaming mode.
    {
      bool HasNonIntrinsicCall = false;
      for (BasicBlock *BB : L->blocks()) {
        for (Instruction &I : *BB) {
          if (auto *CI = dyn_cast<CallInst>(&I)) {
            Function *Callee = CI->getCalledFunction();
            if (!Callee || !Callee->isIntrinsic()) {
              HasNonIntrinsicCall = true;
              break;
            }
          }
        }
        if (HasNonIntrinsicCall)
          break;
      }
      if (HasNonIntrinsicCall) {
        LLVM_DEBUG(dbgs() << "LSS: Skipping loop with non-intrinsic calls: "
                          << L->getLocStr() << "\n");
        continue;
      }
    }

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
