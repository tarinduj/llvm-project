//===-  MLPassPipelinePredictor.cpp - ML Guided Pass Pipeline Predictor  -=== //
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

#include "llvm/Analysis/MLPassPipelinePredictor.h"
#ifdef LLVM_HAVE_TF_API
#include "llvm/Analysis/Utils/TFUtils.h"
#endif
#include "llvm/Analysis/FunctionPropertiesAnalysis.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"
#include <fstream>

namespace llvm {

// static cl::opt<std::string>
//     OutputDirectory("mlpm-feature-dump-dir", cl::init("-"), cl::Hidden,
//               cl::desc("Specify the output directory to dump training code features"));

static cl::opt<std::string>
    OutPath("mlpm-feature-dump-path", cl::init("-"), cl::Hidden,
              cl::desc("Specify the output path to dump training code features"));

template <>
StringRef MLPassPipelinePredictor<Function, FunctionAnalysisManager>::
    getFunctionOptimizationLevel(Function &F, FunctionAnalysisManager &FAM) {

#ifdef LLVM_HAVE_TF_API
  std::vector<TensorSpec> InputSpecs{TensorSpec::createSpec<int32_t>(
      "serving_default_input_1",
      {1, static_cast<int64_t>(
              IRToNativeSizeLearning::FunctionFeatures::FeatureCount)})};
  std::vector<TensorSpec> OutputSpecs{
      TensorSpec::createSpec<float>("StatefulPartitionedCall", {1})};
  Evaluator = std::make_unique<TFModelEvaluator>(
      TFIR2NativeModelPath.getValue().c_str(), InputSpecs, OutputSpecs);
  if (!Evaluator || !Evaluator->isValid()) {
    Evaluator.reset();
    return;
  }
#endif

  StringRef OptLevel = "O0";
  return OptLevel;
}

template <>
StringRef MLPassPipelinePredictor<Module, ModuleAnalysisManager>::
    getFunctionOptimizationLevel(Module &M, ModuleAnalysisManager &MAM) {

  StringRef OptLevel = "O0";
  return OptLevel;
}

template <>
void MLPassPipelinePredictor<Function, FunctionAnalysisManager>::
    dumpTrainingCodeFeatures(Function &F, FunctionAnalysisManager &FAM) {

  // std::string MkDirCommand = "mkdir -p " + OutputDirectory;
  // system(&MkDirCommand[0]);

  // SmallString<128> DirPath(OutputDirectory);
  // StringRef FileName = F.getParent()->getName();

  // sys::path::append(DirPath, FileName);
  // std::string OutPath = DirPath.str().str();
  // dbgs() << OutPath << "\n";

  std::ofstream OutFile;
  OutFile.open(OutPath, std::ios_base::app);

  std::string FunctionName = "####" + F.getName().str() + "\n";
  raw_string_ostream* OutString = new raw_string_ostream(FunctionName);

  FunctionPropertiesAnalysis::Result FPI = FAM.getResult<FunctionPropertiesAnalysis>(F);
  FPI.print(*OutString);
  // FPI.print(dbgs());

  OutFile  << OutString->str();

  OutString->flush();
  OutFile.close();

  // StringRef OptLevel = "O0";
  // return OptLevel
}

template <>
void MLPassPipelinePredictor<Module, ModuleAnalysisManager>::
    dumpTrainingCodeFeatures(Module &M, ModuleAnalysisManager &MAM) {
  // TO DO
}

} // namespace llvm
