//===- SimplexTest.cpp - Tests for ParamLexSimplex ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/Presburger/Set.h"
#include "mlir/Dialect/Presburger/Parser.h"

#include <ostream>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

using namespace mlir;
using namespace mlir::presburger;

static PresburgerSet setFromString(StringRef string) {
  ErrorCallback callback = [](SMLoc loc, const Twine &message) {
    // This is a hack to make the Parser compile
    llvm::errs() << "Parsing error " << message << " at " << loc.getPointer() << '\n';
    llvm_unreachable("PARSING ERROR!!");
    MLIRContext context;
    return mlir::emitError(UnknownLoc::get(&context), message);
  };
  Parser parser(string, callback);
  PresburgerParser setParser(parser);
  PresburgerSet res;
  setParser.parsePresburgerSet(res);
  return res;
}

void expectEqual(StringRef sDesc, StringRef tDesc) {
  auto s = setFromString(sDesc);
  auto t = setFromString(tDesc);
  EXPECT_TRUE(PresburgerSet::equal(s, t));
}

TEST(PresburgerSetTest, Equality) {
  expectEqual("(x) : (exists y, z : x = y + 3z and x >= y and z >= 0 and y >= 0)",
              "(x) : (x >= 0)");
  expectEqual("(x) : (exists y, z : x = y + 3z and x >= y and z >= 0 and y >= 0)",
              "(x) : (exists y, z : x = y + 3z and x >= y and z >= 0 and y >= 0)");
  expectEqual("(x) : (exists q = [(x)/2] : x = 2q)",
              "(x) : (exists q = [(x)/2] : x = 2q)");
  expectEqual("(x) : (exists q = [(x)/2] : x = 2q) or (exists q = [(x)/3] : x = 3q)",
              "(x) : (exists q = [(x)/2] : x = 2q) or (exists q = [(x)/3] : x = 3q)");
}

//   { // [x] -> {[y, z] : x = y + 3z and x >= y and z >= 0 and y >= 0}
//     ParamLexSimplex simplex(3, 1);
//     simplex.addInequality({1, -1, 0, 0});  // x >= y
//     simplex.addInequality({0, 1, 0, -0});  // y >= 0
//     simplex.addEquality({1, -1, -3, 0});  // x == y + 3z
//     simplex.findParamLexmin().dump();
//   }