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

#include "llvm/Analysis/FunctionPropertiesAnalysis.h"
#include "llvm/IR/Instructions.h"

using namespace llvm;

FunctionPropertiesInfo
FunctionPropertiesInfo::getFunctionPropertiesInfo(const Function &F,
                                                  const LoopInfo &LI) {

  FunctionPropertiesInfo FPI;

  FPI.Uses = ((!F.hasLocalLinkage()) ? 1 : 0) + F.getNumUses();

  FPI.InstructionCount = F.getInstructionCount();
  for (const auto &BB : F) {
    ++FPI.BasicBlockCount;

    if (const auto *BI = dyn_cast<BranchInst>(BB.getTerminator())) {
      if (BI->isConditional())
        FPI.BlocksReachedFromConditionalInstruction += BI->getNumSuccessors();
    } else if (const auto *SI = dyn_cast<SwitchInst>(BB.getTerminator())) {
      FPI.BlocksReachedFromConditionalInstruction +=
          (SI->getNumCases() + (nullptr != SI->getDefaultDest()));
    }

    unsigned SuccSize = succ_size(&BB);
    unsigned PredSize = pred_size(&BB);
    if (SuccSize == 1)
      ++FPI.BasicBlockWithSingleSuccessor;
    else if (SuccSize == 2)
      ++FPI.BasicBlockWithTwoSuccessors;
    else if (SuccSize > 2)
      ++FPI.BasicBlockWithMoreThanTwoSuccessors;

    if (PredSize == 1)
      ++FPI.BasicBlockWithSinglePredecessor;
    else if (PredSize == 2)
      ++FPI.BasicBlockWithTwoPredecessors;
    else if (PredSize > 2)
      ++FPI.BasicBlockWithMoreThanTwoPredecessors;

    if (BB.size() > 500)
      ++FPI.BigBasicBlock;
    else if (BB.size() >= 15)
      ++FPI.MediumBasicBlock;
    else
      ++FPI.SmallBasicBlock;

    for (const auto &I : BB) {
      if (auto *CS = dyn_cast<CallBase>(&I)) {
        const auto *Callee = CS->getCalledFunction();
        if (Callee && !Callee->isIntrinsic() && !Callee->isDeclaration())
          ++FPI.DirectCallsToDefinedFunctions;
      }

      if (I.isBinaryOp() && I.getType()->isFloatTy())
        ++FPI.FloatingPointInstCount;
      else if (I.isBinaryOp() && I.getType()->isIntegerTy())
        ++FPI.IntegerInstCount;

      for (unsigned int i = 0; i < I.getNumOperands(); i++)
        if (auto *C = dyn_cast<Constant>(I.getOperand(i))) {
          if (C->getType()->isIntegerTy())
            ++FPI.IntegerConstantOccurrences;
          else if (C->getType()->isFloatTy())
            ++FPI.FloatingConstantOccurrences;
        }

      if (I.isCast())
        ++FPI.CastInstCount;

      switch (I.getOpcode()) {
      case Instruction::Load:
        ++FPI.LoadInstCount; break;
      case Instruction::Store:
        ++FPI.StoreInstCount; break;
      case Instruction::Call:
        ++FPI.CallInstCount; break;
      case Instruction::PHI:
        ++FPI.PHIInstCount; break;
      case Instruction::Alloca:
        ++FPI.AllocaInstCount; break;
      case Instruction::GetElementPtr:
        ++FPI.GEPInstCount; break;
      default:
        break;
      }
    }
    // Loop Depth of the Basic Block
    int64_t LoopDepth;
    LoopDepth = LI.getLoopDepth(&BB);
    if (FPI.MaxLoopDepth < LoopDepth)
      FPI.MaxLoopDepth = LoopDepth;
  }
  FPI.TopLevelLoopCount += llvm::size(LI);
  return FPI;
}

void FunctionPropertiesInfo::print(raw_ostream &OS) const {
  
  OS << "BasicBlockCount: " << BasicBlockCount << "\n"
     << "BasicBlockWithSingleSuccessor: " << BasicBlockWithSingleSuccessor << "\n"
     << "BasicBlockWithTwoSuccessors: " << BasicBlockWithTwoSuccessors << "\n"
     << "BasicBlockWithMoreThanTwoSuccessors: " << BasicBlockWithMoreThanTwoSuccessors << "\n"
     << "BasicBlockWithSinglePredecessor: " << BasicBlockWithSinglePredecessor << "\n"
     << "BasicBlockWithTwoPredecessors: " << BasicBlockWithTwoPredecessors << "\n"
     << "BasicBlockWithMoreThanTwoPredecessors: " << BasicBlockWithMoreThanTwoPredecessors << "\n"
     << "BigBasicBlock: " << BigBasicBlock << "\n"
     << "MediumBasicBlock: " << MediumBasicBlock << "\n"
     << "SmallBasicBlock: " << SmallBasicBlock << "\n"
     << "BlocksReachedFromConditionalInstruction: " << BlocksReachedFromConditionalInstruction << "\n"
     << "Uses: " << Uses << "\n"
     << "DirectCallsToDefinedFunctions: " << DirectCallsToDefinedFunctions<< "\n"
     << "MaxLoopDepth: " << MaxLoopDepth << "\n"
     << "TopLevelLoopCount: " << TopLevelLoopCount << "\n"
     << "IntegerConstantOccurrences: " << IntegerConstantOccurrences << "\n"
     << "FloatingConstantOccurrences: " << FloatingConstantOccurrences << "\n"
     << "InstructionCount: " << InstructionCount << "\n"
     << "CastInstCount: " << CastInstCount << "\n"
     << "FloatingPointInstCount: " << FloatingPointInstCount << "\n"
     << "IntegerInstCount: " << IntegerInstCount << "\n"
     << "LoadInstCount: " << LoadInstCount << "\n"
     << "StoreInstCount: " << StoreInstCount << "\n"
     << "CallInstCount: " << CallInstCount << "\n"
     << "PHIInstCount: " << PHIInstCount << "\n"
     << "AllocaInstCount: " << AllocaInstCount << "\n"
     << "GEPInstCount: " << GEPInstCount << "\n\n";
}

AnalysisKey FunctionPropertiesAnalysis::Key;

FunctionPropertiesInfo
FunctionPropertiesAnalysis::run(Function &F, FunctionAnalysisManager &FAM) {
  return FunctionPropertiesInfo::getFunctionPropertiesInfo(
      F, FAM.getResult<LoopAnalysis>(F));
}

PreservedAnalyses
FunctionPropertiesPrinterPass::run(Function &F, FunctionAnalysisManager &AM) {
  OS << "Printing analysis results of CFA for function "
     << "'" << F.getName() << "':"
     << "\n";
  AM.getResult<FunctionPropertiesAnalysis>(F).print(OS);
  return PreservedAnalyses::all();
}