//===- LoopAVX512Switcher.h - Per-loop AVX-512/AVX-256 switching --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass automatically decides, for each loop, whether to use 512-bit
// (AVX-512) or 256-bit (AVX/AVX2) vectorization using a trained decision tree.
// Loops predicted to benefit from 512-bit vectors are outlined into separate
// functions with prefer-vector-width=512.
//
// On Skylake-AVX512 and later Intel CPUs, AVX-512 instructions can cause a
// frequency drop that negates performance gains from wider vectors. The default
// prefer-vector-width=256 prevents zmm register use. This pass selectively
// promotes loops where 512-bit vectorization outweighs the frequency penalty.
//
// This pass must run BEFORE the Loop Vectorizer in the pipeline.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_SCALAR_LOOPAVX512SWITCHER_H
#define LLVM_TRANSFORMS_SCALAR_LOOPAVX512SWITCHER_H

#include "llvm/IR/PassManager.h"

namespace llvm {

struct LoopAVX512SwitcherPass
    : public PassInfoMixin<LoopAVX512SwitcherPass> {
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
};

} // namespace llvm

#endif // LLVM_TRANSFORMS_SCALAR_LOOPAVX512SWITCHER_H
