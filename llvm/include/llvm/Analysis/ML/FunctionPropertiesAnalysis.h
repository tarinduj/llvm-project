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

#include "llvm/IR/PassManager.h"

namespace llvm {
class Function;
class LoopInfo;

/*
/// If the user specifies the -func-properties-analysis argument on an LLVM tool command line
/// then the value of this boolean will be true, otherwise false.
/// This is the storage for the -func-properties-analysis option.
extern bool FunctionPropertiesAnalysisIsEnabled;
*/
class FunctionPropertiesInfo {
public:
  static FunctionPropertiesInfo getFunctionPropertiesInfo(const Function &F,
                                                          const LoopInfo &LI);

  void print(raw_ostream &OS) const;

  /// Number of basic blocks
  int64_t BasicBlockCount = 0;

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

  // Load Instruction Count
  int64_t LoadInstCount = 0;

  // Store Instruction Count
  int64_t StoreInstCount = 0;

  // Maximum Loop Depth in the Function
  int64_t MaxLoopDepth = 0;

  // Number of Top Level Loops in the Function
  int64_t TopLevelLoopCount = 0;
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

/*
/// This class implements -func-properties-analysis functionality for new pass manager.
class FunctionPropertiesHandler {
  /// Value of this type is capable of uniquely identifying pass invocations.
  /// It is a pair of string Pass-Identifier (which for now is common
  /// to all the instance of a given pass) + sequential invocation counter.
  using PassInvocationID = std::pair<StringRef, unsigned>;

  /// Custom output stream to print timing information into.
  /// By default (== nullptr) we emit time report into the stream created by
  /// CreateInfoOutputFile().
  raw_ostream *OutStream = nullptr;

  bool Enabled;

public:
  FunctionPropertiesHandler(bool Enabled = FunctionPropertiesAnalysisIsEnabled);

  // We intend this to be unique per-compilation, thus no copies.
  FunctionPropertiesHandler(const FunctionPropertiesHandler &) = delete;
  void operator = (const FunctionPropertiesHandler &) = delete;

  void registerCallbacks(PassInstrumentationCallbacks &PIC);

  /// Set a custom output stream for subsequent reporting.
  void setOutStream(raw_ostream &OutStream);

private:
  /// Dumps information for running/triggered timers, useful for debugging
  LLVM_DUMP_METHOD void dump() const;

  // Implementation of pass instrumentation callbacks.
  void runAfterPass(StringRef PassID);
}; */

} // namespace llvm

#endif // LLVM_FUNCTIONPROPERTIESANALYSIS_H_
