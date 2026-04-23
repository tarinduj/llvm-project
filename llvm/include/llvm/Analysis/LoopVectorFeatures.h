//===- LoopVectorFeatures.h - Loop features for vectorization decisions -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Target-neutral loop feature extraction for vectorization decision making.
// Used by target-specific passes (LoopStreamingSwitcher for AArch64 SSVE/NEON,
// etc.) and for JSON feature dumping (training data collection).
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_LOOPVECTORFEATURES_H
#define LLVM_ANALYSIS_LOOPVECTORFEATURES_H

#include "llvm/IR/PassManager.h"
#include "llvm/Support/JSON.h"

namespace llvm {

class DataLayout;
class Loop;
class LoopAccessInfo;
class ScalarEvolution;

struct LoopVectorFeatures {
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

  // Inductions / reductions
  int64_t induction_count = 0;
  bool has_primary_induction = false;
  int64_t reduction_count = 0;

  // Compute intensity
  int64_t ci_num_int_arith_ops = 0;
  int64_t ci_num_fp_arith_ops = 0;
  int64_t ci_num_logic_ops = 0;
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
  int64_t stride_num_unique_base_ptrs = 0;
  bool stride_has_non_affine = false;
  int64_t stride_min_stride_bytes = 0;
  int64_t stride_max_stride_bytes = 0;

  // Nested loop info
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

  // Dtype summary
  int64_t dtype_max_element_size_bytes = 0;
  int64_t dtype_min_element_size_bytes = 0;
  int64_t dtype_num_float_accesses = 0;
  int64_t dtype_num_int_accesses = 0;
  int64_t dtype_num_8bit_accesses = 0;
  int64_t dtype_num_16bit_accesses = 0;
  int64_t dtype_num_32bit_accesses = 0;
  int64_t dtype_num_64bit_accesses = 0;
};

LoopVectorFeatures extractLoopVectorFeatures(Loop *L, ScalarEvolution &SE,
                                             const LoopAccessInfo &LAI,
                                             const DataLayout &DL);

json::Object featuresToJson(const LoopVectorFeatures &F);

class LoopAccessInfoManager;

/// Check whether a loop is eligible for switcher-pass outlining:
/// LoopSimplify form, no non-intrinsic calls, LAI can vectorize memory.
bool isQualifyingLoop(Loop *L, ScalarEvolution &SE,
                      LoopAccessInfoManager &LAIs, const DataLayout &DL);

struct LoopVectorFeatureDumpPass
    : public PassInfoMixin<LoopVectorFeatureDumpPass> {
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
};

} // namespace llvm

#endif // LLVM_ANALYSIS_LOOPVECTORFEATURES_H
