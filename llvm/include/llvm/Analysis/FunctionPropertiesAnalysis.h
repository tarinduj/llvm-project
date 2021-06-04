//=- FunctionPropertiesAnalysis.h - Function Properties Analysis --*- C++ -*-=//
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

#ifndef LLVM_FUNCTIONPROPERTIESANALYSIS_H_
#define LLVM_FUNCTIONPROPERTIESANALYSIS_H_

#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/PassManager.h"

namespace llvm {
class Function;

class FunctionPropertiesInfo {
public:
  static FunctionPropertiesInfo getFunctionPropertiesInfo(const Function &F,
                                                          const LoopInfo &LI);

  void print(raw_ostream &OS) const;

  void testFunc(raw_ostream &OS) const;

  /// Number of basic blocks
  int64_t BasicBlockCount = 0;

  /// Number of Basic Blocks with specific successors.
  int64_t BasicBlockWithSingleSuccessor = 0;
  int64_t BasicBlockWithTwoSuccessors = 0;
  int64_t BasicBlockWithMoreThanTwoSuccessors = 0;

  /// Number of Basic Blocks with specific predecessors.
  int64_t BasicBlockWithSinglePredecessor = 0;
  int64_t BasicBlockWithTwoPredecessors = 0;
  int64_t BasicBlockWithMoreThanTwoPredecessors = 0;

  // Number of basic blocks with more than 500 instructions.
  int64_t BigBasicBlock = 0;

  // Number of basic blocks with 15 ~ 500 instructions.
  int64_t MediumBasicBlock = 0;

  // Number of basic blocks with less than 15 instructions.
  int64_t SmallBasicBlock = 0;

  /// Number of blocks reached from a conditional instruction, or that are
  /// 'cases' of a SwitchInstr.
  // FIXME: We may want to replace this with a more meaningful metric, like
  // number of conditionally executed blocks:
  // 'if (a) s();' would be counted here as 2 blocks, just like
  // 'if (a) s(); else s2(); s3();' would.
  int64_t BlocksReachedFromConditionalInstruction = 0;

  /// Number of uses of this function, plus 1 if the function is callable
  /// outside the module.
  int64_t Uses = 0;

  /// Number of direct calls made from this function to other functions
  /// defined in this module.
  int64_t DirectCallsToDefinedFunctions = 0;

  // Maximum Loop Depth in the Function
  int64_t MaxLoopDepth = 0;

  // Number of Top Level Loops in the Function
  int64_t TopLevelLoopCount = 0;

  /// Number of Occurences of Constants
  int64_t IntegerConstantOccurrences = 0;
  int64_t FloatingConstantOccurrences = 0;

  /// Number of all instuctions.
  int64_t InstructionCount = 0;

  /// Number of Cast-like Instruction Count in the Function.
  int64_t CastInstCount = 0;

  /// Number of Floating Point Instruction Count.
  int64_t FloatingPointInstCount = 0;

  /// Number of Integer Instruction Count.
  int64_t IntegerInstCount = 0;

  // Different Instruction (Opcode) Type Counts
  int64_t LoadInstCount = 0;
  int64_t StoreInstCount = 0;
  int64_t CallInstCount = 0;
  int64_t PHIInstCount = 0;
  int64_t AllocaInstCount = 0;
  int64_t GEPInstCount = 0;
};

// Analysis pass
class FunctionPropertiesAnalysis
    : public AnalysisInfoMixin<FunctionPropertiesAnalysis> {

public:
  static AnalysisKey Key;

  using Result = FunctionPropertiesInfo;

  Result run(Function &F, FunctionAnalysisManager &FAM);
};

/// Printer pass for the FunctionPropertiesAnalysis results.
class FunctionPropertiesPrinterPass
    : public PassInfoMixin<FunctionPropertiesPrinterPass> {
  raw_ostream &OS;

public:
  explicit FunctionPropertiesPrinterPass(raw_ostream &OS) : OS(OS) {}

  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
};

} // namespace llvm
#endif // LLVM_FUNCTIONPROPERTIESANALYSIS_H_