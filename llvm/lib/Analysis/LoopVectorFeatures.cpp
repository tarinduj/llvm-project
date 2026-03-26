//===- LoopVectorFeatures.cpp - Loop features for vectorization decisions --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/LoopVectorFeatures.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/LoopAccessAnalysis.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

static cl::opt<std::string> FeatureOutput(
    "loop-vector-feature-output", cl::init(""), cl::Hidden,
    cl::desc("Output file for pre-LV loop features (JSON)."));

//===----------------------------------------------------------------------===//
// Feature Extraction
//===----------------------------------------------------------------------===//

LoopVectorFeatures llvm::extractLoopVectorFeatures(Loop *L,
                                                    ScalarEvolution &SE,
                                                    const LoopAccessInfo &LAI,
                                                    const DataLayout &DL) {
  LoopVectorFeatures F;

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

  // --- Inductions / reductions ---
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
      ++NumReductions;
    }
  }
  HasPrimaryInduction = (PrimaryCandidate != nullptr);
  F.induction_count = static_cast<int64_t>(NumInductions);
  F.has_primary_induction = HasPrimaryInduction;
  F.reduction_count = static_cast<int64_t>(NumReductions);

  // --- Stride, dtype, and base pointer info ---
  const auto &MemInsts = DepChecker.getMemoryInstructions();
  PredicatedScalarEvolution PSE(SE, *L);

  int64_t MaxStrideBytes = 0, MinStrideBytes = INT64_MAX;
  unsigned NumUnitStride = 0, NumNonUnitStride = 0, NumNonAffine = 0;
  unsigned NumFloatAccesses = 0, NumIntAccesses = 0;
  unsigned Num8BitAccesses = 0, Num16BitAccesses = 0;
  unsigned Num32BitAccesses = 0, Num64BitAccesses = 0;
  uint64_t MaxElementSizeBytes = 0, MinElementSizeBytes = UINT64_MAX;
  DenseMap<const Value *, int> BasePtrToId;
  int NextBasePtrId = 0;

  for (Instruction *I : MemInsts) {
    Type *AccessTy = getLoadStoreType(I);
    uint64_t ElementSizeBytes = DL.getTypeStoreSize(AccessTy);
    Value *Ptr = getLoadStorePointerOperand(I);

    // Float vs Int classification
    if (AccessTy->isFloatingPointTy())
      ++NumFloatAccesses;
    else
      ++NumIntAccesses;

    // Per-bitwidth classification
    unsigned BitWidth = DL.getTypeSizeInBits(AccessTy);
    if (BitWidth <= 8)
      ++Num8BitAccesses;
    else if (BitWidth <= 16)
      ++Num16BitAccesses;
    else if (BitWidth <= 32)
      ++Num32BitAccesses;
    else
      ++Num64BitAccesses;

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
  F.stride_num_unique_base_ptrs = static_cast<int64_t>(NextBasePtrId);
  F.stride_has_non_affine = NumNonAffine > 0;
  F.stride_max_stride_bytes = MaxStrideBytes;
  F.stride_min_stride_bytes =
      (MinStrideBytes == INT64_MAX) ? 0 : MinStrideBytes;
  F.dtype_num_float_accesses = static_cast<int64_t>(NumFloatAccesses);
  F.dtype_num_int_accesses = static_cast<int64_t>(NumIntAccesses);
  F.dtype_num_8bit_accesses = static_cast<int64_t>(Num8BitAccesses);
  F.dtype_num_16bit_accesses = static_cast<int64_t>(Num16BitAccesses);
  F.dtype_num_32bit_accesses = static_cast<int64_t>(Num32BitAccesses);
  F.dtype_num_64bit_accesses = static_cast<int64_t>(Num64BitAccesses);
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

  // --- Nested loop info ---
  F.nested_num_sub_loops = static_cast<int64_t>(L->getSubLoops().size());
  if (Loop *Parent = L->getParentLoop()) {
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
    bool HasBackward = false, HasForward = false;
    for (const auto &Dep : *Deps) {
      switch (Dep.Type) {
      case MemoryDepChecker::Dependence::Backward:
      case MemoryDepChecker::Dependence::BackwardVectorizable:
      case MemoryDepChecker::Dependence::BackwardVectorizableButPreventsForwarding:
        HasBackward = true;
        F.dep_has_loop_carried_deps = true;
        break;
      case MemoryDepChecker::Dependence::Forward:
      case MemoryDepChecker::Dependence::ForwardButPreventsForwarding:
        HasForward = true;
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
    F.dep_has_backward_deps = HasBackward;
    F.dep_has_forward_deps = HasForward;
  }
  F.rec_has_recurrence = F.dep_has_backward_deps || F.dep_has_forward_deps;

  // --- Access pattern info (SCEV AddRec depth) ---
  unsigned MaxAddRecDepth = 0, MinAddRecDepth = UINT_MAX;
  SmallPtrSet<const Value *, 8> BasePointers;

  for (Instruction *I : MemInsts) {
    Value *Ptr = getLoadStorePointerOperand(I);
    if (!Ptr)
      continue;
    const SCEV *PtrSCEV = SE.getSCEV(Ptr);

    const SCEV *Base = SE.getPointerBase(PtrSCEV);
    if (const SCEVUnknown *BaseUnknown = dyn_cast<SCEVUnknown>(Base))
      BasePointers.insert(BaseUnknown->getValue());

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

//===----------------------------------------------------------------------===//
// JSON Conversion
//===----------------------------------------------------------------------===//

json::Object llvm::featuresToJson(const LoopVectorFeatures &F) {
  json::Object J;
  J["loop_depth"] = F.loop_depth;
  J["num_blocks"] = F.num_blocks;
  J["trip_count_value"] = F.trip_count_value;
  J["trip_count_is_constant"] = F.trip_count_is_constant;
  J["num_loads"] = F.num_loads;
  J["num_stores"] = F.num_stores;
  J["num_symbolic_strides"] = F.num_symbolic_strides;
  J["max_safe_width_bits"] = F.max_safe_width_bits;
  J["induction_count"] = F.induction_count;
  J["has_primary_induction"] = F.has_primary_induction;
  J["ci_num_int_arith_ops"] = F.ci_num_int_arith_ops;
  J["ci_num_fp_arith_ops"] = F.ci_num_fp_arith_ops;
  J["ci_num_logic_ops"] = F.ci_num_logic_ops;
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
  J["stride_num_unique_base_ptrs"] = F.stride_num_unique_base_ptrs;
  J["stride_has_non_affine"] = F.stride_has_non_affine;
  J["stride_min_stride_bytes"] = F.stride_min_stride_bytes;
  J["stride_max_stride_bytes"] = F.stride_max_stride_bytes;
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
  J["rec_has_recurrence"] = F.rec_has_recurrence;
  J["dtype_max_element_size_bytes"] = F.dtype_max_element_size_bytes;
  J["dtype_min_element_size_bytes"] = F.dtype_min_element_size_bytes;
  J["dtype_num_float_accesses"] = F.dtype_num_float_accesses;
  J["dtype_num_int_accesses"] = F.dtype_num_int_accesses;
  J["dtype_num_8bit_accesses"] = F.dtype_num_8bit_accesses;
  J["dtype_num_16bit_accesses"] = F.dtype_num_16bit_accesses;
  J["dtype_num_32bit_accesses"] = F.dtype_num_32bit_accesses;
  J["dtype_num_64bit_accesses"] = F.dtype_num_64bit_accesses;
  return J;
}

//===----------------------------------------------------------------------===//
// Qualifying Loop Check
//===----------------------------------------------------------------------===//

bool llvm::isQualifyingLoop(Loop *L, ScalarEvolution &SE,
                            LoopAccessInfoManager &LAIs, const DataLayout &DL) {
  if (!L->isLoopSimplifyForm())
    return false;

  // Check for non-intrinsic calls BEFORE LAI/feature extraction — loops with
  // calls (memcpy, external functions) have complex pointer expressions that
  // can crash SCEV during LAI construction or feature extraction.
  for (BasicBlock *BB : L->blocks()) {
    for (Instruction &I : *BB) {
      if (auto *CI = dyn_cast<CallInst>(&I)) {
        Function *Callee = CI->getCalledFunction();
        if (!Callee || !Callee->isIntrinsic())
          return false;
      }
    }
  }

  // Safe to compute LAI now that we know there are no non-intrinsic calls.
  const LoopAccessInfo &LAI = LAIs.getInfo(*L);

  if (!LAI.canVectorizeMemory())
    return false;

  LoopVectorFeatures Features = extractLoopVectorFeatures(L, SE, LAI, DL);
  if (Features.reduction_count > 0)
    return false;

  return true;
}

//===----------------------------------------------------------------------===//
// Feature Dump Pass
//===----------------------------------------------------------------------===//

static json::Object *GlobalFeatures = nullptr;

static void writeFeatures() {
  if (!GlobalFeatures || FeatureOutput.empty())
    return;
  std::error_code EC;
  raw_fd_ostream OS(FeatureOutput, EC);
  if (EC) {
    errs() << "Error opening feature output: " << EC.message() << "\n";
    return;
  }
  json::Object Root;
  Root["functions"] = std::move(*GlobalFeatures);
  OS << json::Value(std::move(Root)) << "\n";
}

PreservedAnalyses LoopVectorFeatureDumpPass::run(Function &F,
                                                  FunctionAnalysisManager &AM) {
  if (FeatureOutput.empty())
    return PreservedAnalyses::all();

  if (!GlobalFeatures) {
    GlobalFeatures = new json::Object();
    static bool RegisteredCleanup = false;
    if (!RegisteredCleanup) {
      std::atexit(writeFeatures);
      RegisteredCleanup = true;
    }
  }

  if (F.hasFnAttribute("aarch64_pstate_sm_enabled") ||
      F.hasFnAttribute("aarch64_pstate_sm_body"))
    return PreservedAnalyses::all();

  if (F.hasOptNone() || F.empty())
    return PreservedAnalyses::all();

  auto &LI = AM.getResult<LoopAnalysis>(F);
  if (LI.empty())
    return PreservedAnalyses::all();

  auto &SE = AM.getResult<ScalarEvolutionAnalysis>(F);
  auto &LAIs = AM.getResult<LoopAccessAnalysis>(F);
  const DataLayout &DL = F.getDataLayout();

  // Collect innermost loops
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

  std::string FuncName = F.getName().str();
  int64_t QualifyingIndex = 0;

  for (Loop *L : InnermostLoops) {
    if (!isQualifyingLoop(L, SE, LAIs, DL))
      continue;

    const LoopAccessInfo &LAI = LAIs.getInfo(*L);
    LoopVectorFeatures Features = extractLoopVectorFeatures(L, SE, LAI, DL);

    json::Object LoopObj = featuresToJson(Features);
    LoopObj["location"] = L->getLocStr();
    LoopObj["qualifying_index"] = QualifyingIndex;
    ++QualifyingIndex;

    if (json::Value *Existing = GlobalFeatures->get(FuncName)) {
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
      (*GlobalFeatures)[FuncName] = std::move(FuncData);
    }
  }

  // Write qualifying_loop_count to the function-level object.
  if (QualifyingIndex > 0) {
    if (json::Value *Existing = GlobalFeatures->get(FuncName)) {
      if (json::Object *ExObj = Existing->getAsObject()) {
        (*ExObj)["qualifying_loop_count"] = QualifyingIndex;
      }
    }
  }

  return PreservedAnalyses::all();
}
