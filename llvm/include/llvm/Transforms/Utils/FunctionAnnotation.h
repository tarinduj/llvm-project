//= FunctionAnnotation.h - Function Annotation with Function Attributes- C++ =//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the FunctionAnnotation class used to annotate functions
// with function attributes.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_UTILS_FUNCTIONANNOTATION_H_
#define LLVM_TRANSFORMS_UTILS_FUNCTIONANNOTATION_H_

#include "llvm/IR/PassManager.h"

namespace llvm {

class FunctionAnnotationPass : public PassInfoMixin<FunctionAnnotationPass> {
public:
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &FAM);
};

} // namespace llvm

#endif // LLVM_TRANSFORMS_UTILS_FUNCTIONANNOTATION_H_