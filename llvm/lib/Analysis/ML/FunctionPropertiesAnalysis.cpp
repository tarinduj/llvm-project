//===- FunctionPropertiesAnalysis.cpp - Function Properties Analysis ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the FunctionPropertiesInfo and FunctionPropertiesAnalysis
// classes used to extract function properties.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/ML/FunctionPropertiesAnalysis.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

void FunctionPropertiesInfo::analyze(const Function &F) {

  Uses = ((!F.hasLocalLinkage()) ? 1 : 0) + F.getNumUses();

  for (const auto &BB : F) {
    ++BasicBlockCount;

    if (const auto *BI = dyn_cast<BranchInst>(BB.getTerminator())) {
      if (BI->isConditional())
        BlocksReachedFromConditionalInstruction += BI->getNumSuccessors();
    } else if (const auto *SI = dyn_cast<SwitchInst>(BB.getTerminator())) {
      BlocksReachedFromConditionalInstruction +=
          (SI->getNumCases() + (nullptr != SI->getDefaultDest()));
    }

    for (const auto &I : BB) {
      if (auto *CS = dyn_cast<CallBase>(&I)) {
        const auto *Callee = CS->getCalledFunction();
        if (Callee && !Callee->isIntrinsic() && !Callee->isDeclaration())
          ++DirectCallsToDefinedFunctions;
      }
    }
  }
}

AnalysisKey FunctionPropertiesAnalysis::Key;

FunctionPropertiesInfo
FunctionPropertiesAnalysis::run(Function &F, FunctionAnalysisManager &FAM) {
  FunctionPropertiesInfo FPI;
  FPI.analyze(F);
  return FPI;
}
