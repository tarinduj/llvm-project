//=== FunctionAnnotation.cpp - Function Annotation with Function Attributes ==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the FunctionAnnotation class used to annotate functions
// with function attributes.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/FunctionAnnotation.h"
#include "llvm/Analysis/MLPassPipelinePredictor.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include <fstream>
#include <sstream>

namespace llvm {

static cl::opt<std::string> FunctionAnnotationAttributeName(
    "func-annotate-attribute-name", cl::init(""), cl::Hidden,
    cl::desc("Specify the name of the attribute to annotate functions with."));

static cl::opt<bool> DumpMLPMData("dump-mlpm-data", cl::init(false), cl::Hidden,
                  cl::desc("Dump the training data (code features) for MLPM"));

static cl::opt<std::string> FunctionAnnotationCSVPath(
    "func-annotate-csv-path", cl::init("-"), cl::Hidden,
    cl::desc("Specify the path to the CSV file containing the functions and "
             "their attribute values.\nCSV file header should have the format: "
             "[Index, Function Name, Attribute Value]"));

PreservedAnalyses FunctionAnnotationPass::run(Function &F, FunctionAnalysisManager &AM) {
  // annotate the function with the function attributes
  if (!DumpMLPMData) {
    const StringRef FunctionName = F.getName();

    if (FunctionAnnotationAttributeName.empty()) {
      report_fatal_error(
          "Name of the function annotation attribute not specified.");
    }
    const Attribute &A = F.getFnAttribute(FunctionAnnotationAttributeName);
    StringRef FnOptLevel = A.getValueAsString();
    
    if (FnOptLevel.empty()) {
      // annotate functions using the ML model (default option)
      if (FunctionAnnotationCSVPath == "-"){
        FnOptLevel = MLPassPipelinePredictor<Function, FunctionAnalysisManager>::getFunctionOptimizationLevel(F, AM);
        F.addFnAttr("opt-level", FnOptLevel);

      // annotate functions using the CSV file if provided
      } else {
        std::ifstream InputFile;
        InputFile.open(FunctionAnnotationCSVPath);
        if (!InputFile.is_open()) {
          report_fatal_error("Function annotation CSV file not found.");
        }

        std::string Line;
        while (std::getline(InputFile, Line)) {
          std::istringstream ISS(Line);
          std::string ID, Name, OLevel;

          if (std::getline(ISS, ID, ',') && std::getline(ISS, Name, ',') &&
              std::getline(ISS, OLevel)) {
            char *EndPtr = nullptr;
            if (Name.c_str() != EndPtr && Name == FunctionName) {
              F.addFnAttr(FunctionAnnotationAttributeName, OLevel);
              break;
            }
          }
        }
        InputFile.close();
      }
    }
  // dump the code features for training the ML model
  } else { 
    MLPassPipelinePredictor<Function, FunctionAnalysisManager>::dumpTrainingCodeFeatures(F, AM);
  }
  return PreservedAnalyses::all();

//   // dumping training data for ml based pass pipeline prediction

// /// Flag to dump mlpm training data (code features)
// extern cl::opt<bool> DumpMLPMData;

// if (DumpMLPMData){
//         mlpm::dumpMLPMTrainingData<IRUnitT>(IR, AM);
//       }

// template <typename IRUnitT, typename AnalysisManagerT>
// void dumpMLPMTrainingData(IRUnitT &IR, AnalysisManagerT &AM) {
//   if (any_isa<Function *>(Any(&IR))) { 
//       Function *F = any_cast<Function *>(Any(&IR));
//       const Attribute &A = F->getFnAttribute("opt-level");
//       StringRef FnOptLevel = A.getValueAsString();
      
//       // check to ensure only the code features at the first encounter of the function is recorded
//       if (FnOptLevel.empty()) {
//         // dbgs() << F->getName() << "\n";
//         FnOptLevel = MLPassPipelinePredictor<IRUnitT, AnalysisManagerT>::dumpTrainingCodeFeatures(IR, AM);
//         F->addFnAttr("opt-level", FnOptLevel);
//       }
//   }
// }
}

} // namespace llvm