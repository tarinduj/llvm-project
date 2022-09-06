//==- MLPassPipelinePredictor.h -  ML Guided Pass Pipeline Predictor - C++ -==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the MLPassPipelinePredictor class used to predict 
// optimization levels for functions using machine learning. 
//
//===----------------------------------------------------------------------===//


#ifndef LLVM_ANALYSIS_MLPASSPIPELINEPREDICTOR_H
#define LLVM_ANALYSIS_MLPASSPIPELINEPREDICTOR_H

#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"

namespace llvm {
class Function;
class Module;

/// ML guided Pass Pipeline Predictor
template <typename IRUnitT, typename AnalysisManagerT>
class MLPassPipelinePredictor {
public:
  static StringRef getFunctionOptimizationLevel(IRUnitT &IR, AnalysisManagerT &AM);

  static void dumpTrainingCodeFeatures(IRUnitT &IR, AnalysisManagerT &AM);

private:
  std::unique_ptr<TFModelEvaluator> Evaluator;
};
};

} // namespace llvm

#endif // LLVM_ANALYSIS_MLPASSPIPELINEPREDICTOR_H
