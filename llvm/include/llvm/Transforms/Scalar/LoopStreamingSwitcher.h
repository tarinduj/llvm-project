//===- LoopStreamingSwitcher.h - Per-loop SSVE/NEON switching ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass automatically decides, for each loop, whether to use Streaming SVE
// (SSVE) or NEON vectorization using a trained decision tree. Loops predicted
// to benefit from SSVE are outlined into separate functions marked with
// aarch64_pstate_sm_body, so the Loop Vectorizer and backend will use
// streaming SVE instructions.
//
// This pass must run BEFORE the Loop Vectorizer in the pipeline.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_SCALAR_LOOPSTREAMINGSWITCHER_H
#define LLVM_TRANSFORMS_SCALAR_LOOPSTREAMINGSWITCHER_H

#include "llvm/IR/PassManager.h"

namespace llvm {

struct LoopStreamingSwitcherPass
    : public PassInfoMixin<LoopStreamingSwitcherPass> {
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
};

} // namespace llvm

#endif // LLVM_TRANSFORMS_SCALAR_LOOPSTREAMINGSWITCHER_H
