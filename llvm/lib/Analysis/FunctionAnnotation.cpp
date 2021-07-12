//=== FunctionAnnotation.cpp - Function Annotation with Optimization Level ===//
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

#include "llvm/Analysis/FunctionAnnotation.h"

using namespace llvm;

AnalysisKey FunctionAnnotation::Key;

FunctionAnnotation::Result
FunctionAnnotation::run(Function &F, FunctionAnalysisManager &AM) {
  const StringRef FunctionName = F.getName();
  const Attribute &A = F.getFnAttribute("opt-level");
  const StringRef FnOptLevel = A.getValueAsString();

  // dbgs() << FunctionName << ": " << FnOptLevel << "\n";

  if (FnOptLevel.empty()) {
    std::ifstream InputFile("/Users/tarindujayatilaka/Documents/LLVM/results/"
                            "ASM/Switch Pipeline/lookuptable.csv");
    if (!InputFile.is_open()) {
      dbgs() << "File not found!"
             << "\n";
    }
    std::string Line;
    while (std::getline(InputFile, Line)) {
      std::istringstream ISS(Line);
      std::string ID, Name, OLevel;

      if (std::getline(ISS, ID, ',') && std::getline(ISS, Name, ',') &&
          std::getline(ISS, OLevel)) {
        char *endp = nullptr;
        if (Name.c_str() != endp && Name == FunctionName) {
          // dbgs() << "FOUND A MATACH\n";
          // dbgs() << "Setting Attribute: " << OLevel << "\n";
          F.addFnAttr("opt-level", OLevel);
          break;
        }
      }
    }
  }

  return FnOptLevel;
}