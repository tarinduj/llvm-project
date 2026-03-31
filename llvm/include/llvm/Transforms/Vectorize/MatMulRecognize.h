//===- MatMulRecognize.h - MatMul pattern recognition + SME lowering ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass recognizes matrix multiply patterns in loop nests and optionally
// lowers them to AArch64 SME FMOPA (outer product accumulate) sequences.
//
// It is fully opt-in: disabled by default, does not affect any existing passes.
// Enable with -enable-matmul-sme=true or run standalone via
// opt -passes=matmul-recognize-and-lower.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_VECTORIZE_MATMULRECOGNIZE_H
#define LLVM_TRANSFORMS_VECTORIZE_MATMULRECOGNIZE_H

#include "llvm/IR/PassManager.h"

namespace llvm {

class Loop;
class ScalarEvolution;
class Value;
class PHINode;
class Type;

/// Information about a recognized matrix multiply pattern.
///
/// Represents: C[i][j] += alpha * A[i][k] * B[k][j]  (with variants)
///
/// The loops may be in any nesting order. A and/or B may be transposed.
/// Alpha and Beta scaling factors are optional.
struct MatMulInfo {
  // The three loops forming the matmul nest (any nesting order).
  Loop *LoopI = nullptr; // iterates over C rows
  Loop *LoopJ = nullptr; // iterates over C columns
  Loop *LoopK = nullptr; // reduction dimension

  // Induction variables for each loop.
  PHINode *IndI = nullptr;
  PHINode *IndJ = nullptr;
  PHINode *IndK = nullptr;

  // Array base pointers (after stripping GEPs).
  Value *BaseC = nullptr;
  Value *BaseA = nullptr;
  Value *BaseB = nullptr;

  // Optional scaling factors (nullptr means 1.0).
  Value *Alpha = nullptr;
  Value *Beta = nullptr;

  // Whether the A or B operand is transposed relative to canonical form.
  // Canonical: A[i][k], B[k][j].  Transposed: A[k][i] or B[j][k].
  bool ATransposed = false;
  bool BTransposed = false;

  // Element type of the computation (e.g., f32, f64).
  Type *ElemTy = nullptr;

  // Trip counts (0 if unknown at compile time).
  unsigned TripCountI = 0;
  unsigned TripCountJ = 0;
  unsigned TripCountK = 0;

  // Leading dimensions for GEP stride computation.
  Value *LdA = nullptr; // number of columns in A (or rows if transposed)
  Value *LdB = nullptr; // number of columns in B (or rows if transposed)
  Value *LdC = nullptr; // number of columns in C
};

struct MatMulRecognizeAndLowerPass
    : public PassInfoMixin<MatMulRecognizeAndLowerPass> {
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
};

// --- Functions defined in MatMulSMELower.cpp ---

/// Create a new internal function with a tiled FMOPA body for f32 GEMM.
/// The function signature is:
///   void @Name(ptr %C, ptr %A, ptr %B,
///              i64 %NI, i64 %NJ, i64 %NK,
///              i64 %LdC, i64 %LdA, i64 %LdB)
Function *createAndBuildFMOPAFunction(Module &M, StringRef Name);

/// Extract the trip count upper bound from a canonical loop's exit condition.
Value *getLoopBoundValue(Loop *L);

} // namespace llvm

#endif // LLVM_TRANSFORMS_VECTORIZE_MATMULRECOGNIZE_H
