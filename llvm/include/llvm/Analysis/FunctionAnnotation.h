//= FunctionAnnotation.h - Function Annotation with Optimization Level - C++ =//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the FunctionAnnotation class used to annotate functions
// with optimization levels.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_FUNCTIONANNOTATION_H_
#define LLVM_ANALYSIS_FUNCTIONANNOTATION_H_

#include "llvm/IR/PassManager.h"

namespace llvm {

class FunctionAnnotation : public AnalysisInfoMixin<FunctionAnnotation> {
  friend AnalysisInfoMixin<FunctionAnnotation>;
  static AnalysisKey Key;

public:
  using Result = StringRef;
  Result run(Function &F, FunctionAnalysisManager &FAM);
};

} // namespace llvm

#endif // LLVM_ANALYSIS_FUNCTIONANNOTATION_H_
