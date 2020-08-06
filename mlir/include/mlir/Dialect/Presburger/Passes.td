//===-- Passes.td - Presburger pass definition file --------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_PRESBURGER_PASSES
#define MLIR_DIALECT_PRESBURGER_PASSES

include "mlir/Pass/PassBase.td"


def PresburgerEvaluate : FunctionPass<"presburger-evaluate"> {
  let summary = "Evaluates Presburger operations";
  let constructor = "mlir::createPresburgerEvaluatePass()";
}

#endif // MLIR_DIALECT_PRESBURGER_PASSES
