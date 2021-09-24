#include "mlir/Analysis/Presburger/Coalesce.h"
#include "mlir/Analysis/Presburger/Set.h"
#include "mlir/Analysis/Presburger/TransprecSet.h"
#include "mlir/Dialect/Presburger/Parser.h"
#include "mlir/Analysis/Presburger/Presburger-impl.h"
#include <iostream>
#include <string>
#include <x86intrin.h>
#include <fstream>
#include <valgrind/callgrind.h>

using namespace mlir;
using namespace mlir::presburger;

unsigned TransprecSet::waterline = 0;

template <typename Int>
Optional<PresburgerSet<Int>> setFromString(StringRef string) {
  ErrorCallback callback = [&](SMLoc loc, const Twine &message) {
    // This is a hack to make the Parser compile
    // These have to be commented out currently because "errors" are raised
    // When an integer that can't fit in 32 bits appears in the input.
    // This is detected and handled by the transprecision infrastructures.
    // Unfortunately we do not yet check if it is an integer overflow error or
    // some other error, and all errors are assumed to be integer overflow errors.
    // If modifying something that might cause a different error here, note that
    // you have to uncomment the following to make the error be printed.
    // llvm::errs() << "Parsing error " << message << " at " << loc.getPointer()
    //              << '\n';
    // llvm::errs() << "invalid input " << string << '\n';

    // llvm_unreachable("PARSING ERROR!!");
    MLIRContext context;
    return mlir::emitError(UnknownLoc::get(&context), "");
  };
  Parser<Int> parser(string, callback);
  PresburgerParser<Int> setParser(parser);
  PresburgerSet<Int> res;
  if (failed(setParser.parsePresburgerSet(res)))
    return {};
  return res;
}

void consumeLine(unsigned cnt = 1) {
  while (cnt--) {
    char str[1'000'000];
    std::cin.getline(str, 1'000'000);
    // std::cerr << "Consumed '" << str << "'\n";
  }
}

TransprecSet getTransprecSetFromString(StringRef str) {
  // std::cerr << "Read '" << str << "'\n";
  if (auto set = setFromString<SafeInteger<int16_t>>(str))
    return TransprecSet(*set);
  else if (auto set = setFromString<SafeInteger<int64_t>>(str))
    return TransprecSet(*set);
  else if (auto set = setFromString<SafeInteger<__int128_t>>(str))
    return TransprecSet(*set);
  else if (auto set = setFromString<mpz_class>(str))
    return TransprecSet(*set);
  else
    llvm_unreachable("Input did not fit in 128-bits!");
}

TransprecSet getSetFromInput() {
  char str[1'000'000];
  std::cin.getline(str, 1'000'000);
  return getTransprecSetFromString(str);
}

void consumeNewline() {
  char c;
  std::cin.get(c);
  if (c != '\n') {
    std::cerr << "Expected newline!\n";
    exit(1);
  }
}

bool isBinary(const std::string &op) {
  return op == "equal" || op == "union" || op == "subtract" || op == "intersect";
}

int main(int argc, char **argv) {
  std::ios::sync_with_stdio(false);
  if (argc != 3) {
    std::cerr << "usage: ./run-presburger <op> <run index>\nPass input to stdin.\n";
    return 1;
  }

  const unsigned numRuns = 1;
  std::string op = argv[1];

  unsigned numCases;
  std::cin >> numCases;
  consumeNewline();

  unsigned runIdx = atoi(argv[2]);
  const unsigned casesPerRun = 1000;
  unsigned caseStart = runIdx * casesPerRun;
  unsigned caseEnd = std::min((runIdx + 1) * casesPerRun, numCases);
  if (caseStart >= numCases) {
    std::cerr << "Run index too high, nothing to do!\n";
    return 2;
  }

  unsigned casesToSkip = runIdx * casesPerRun;
  consumeLine((isBinary(op) ? 3 : 2) * casesToSkip);

  for (unsigned j = caseStart; j < caseEnd; ++j) {
    if (j % 100 == 0)
      std::cerr << op << ' ' << j << '/' << numCases << '\n';

    TransprecSet::waterline = 0;
    if (op == "empty") {
      TransprecSet setA = getSetFromInput();
      CALLGRIND_START_INSTRUMENTATION;
      for (unsigned i = 0; i < numRuns; ++i) {

        auto a = setA;
        if (i == numRuns - 1)
          CALLGRIND_TOGGLE_COLLECT;
        volatile auto res = a.isIntegerEmpty();
        if (i == numRuns - 1)
          CALLGRIND_TOGGLE_COLLECT;
        res = res;

      }
      CALLGRIND_STOP_INSTRUMENTATION;
    } else if (op == "equal") {
      TransprecSet setA = getSetFromInput();
      TransprecSet setB = getSetFromInput();
      CALLGRIND_START_INSTRUMENTATION;
      for (unsigned i = 0; i < numRuns; ++i) {

        auto a = setA;
        auto b = setB;
        if (i == numRuns - 1)
          CALLGRIND_TOGGLE_COLLECT;
        volatile auto res = a.equal(b);
        if (i == numRuns - 1)
          CALLGRIND_TOGGLE_COLLECT;
        res = res;

      }
      CALLGRIND_STOP_INSTRUMENTATION;
    } else if (op == "union") {
      TransprecSet setA = getSetFromInput();
      TransprecSet setB = getSetFromInput();
      CALLGRIND_START_INSTRUMENTATION;
      for (unsigned i = 0; i < numRuns; ++i) {

        auto a = setA;
        auto b = setB;
        if (i == numRuns - 1)
          CALLGRIND_TOGGLE_COLLECT;
        __sync_synchronize();
        a.unionSet(b);
        __sync_synchronize();

        if (i == numRuns - 1)
          CALLGRIND_TOGGLE_COLLECT;
      }
      CALLGRIND_STOP_INSTRUMENTATION;
    } else if (op == "intersect") {
      TransprecSet setA = getSetFromInput();
      TransprecSet setB = getSetFromInput();
      CALLGRIND_START_INSTRUMENTATION;
      for (unsigned i = 0; i < numRuns; ++i) {

        auto a = setA;
        auto b = setB;
        if (i == numRuns - 1)
          CALLGRIND_TOGGLE_COLLECT;
        __sync_synchronize();
        a.intersectSet(b);
        __sync_synchronize();

        if (i == numRuns - 1)
          CALLGRIND_TOGGLE_COLLECT;
      }
      CALLGRIND_STOP_INSTRUMENTATION;
    } else if (op == "subtract") {
      TransprecSet setA = getSetFromInput();
      TransprecSet setB = getSetFromInput();
      CALLGRIND_START_INSTRUMENTATION;
      for (unsigned i = 0; i < numRuns; ++i) {

        auto a = setA;
        auto b = setB;
        if (i == numRuns - 1)
          CALLGRIND_TOGGLE_COLLECT;
        __sync_synchronize();
        a.subtract(b);
        __sync_synchronize();

        if (i == numRuns - 1)
          CALLGRIND_TOGGLE_COLLECT;
      }
      CALLGRIND_STOP_INSTRUMENTATION;
    } else if (op == "coalesce") {
      TransprecSet setA = getSetFromInput();
      CALLGRIND_START_INSTRUMENTATION;
      for (unsigned i = 0; i < numRuns; ++i) {

        auto a = setA;
        if (i == numRuns - 1)
          CALLGRIND_TOGGLE_COLLECT;
        __sync_synchronize();
        auto res = a.coalesce();
        __sync_synchronize();

        if (i == numRuns - 1)
          CALLGRIND_TOGGLE_COLLECT;
      }
      CALLGRIND_STOP_INSTRUMENTATION;
    } else if (op == "complement") {
      TransprecSet setA = getSetFromInput();
      CALLGRIND_START_INSTRUMENTATION;
      for (unsigned i = 0; i < numRuns; ++i) {

        auto a = setA;
        if (i == numRuns - 1)
          CALLGRIND_TOGGLE_COLLECT;
        __sync_synchronize();
        auto res = a.complement();
        __sync_synchronize();

        if (i == numRuns - 1)
          CALLGRIND_TOGGLE_COLLECT;
      }
      CALLGRIND_STOP_INSTRUMENTATION;
    } else if (op == "eliminate") {
      TransprecSet setA = getSetFromInput();
      CALLGRIND_START_INSTRUMENTATION;
      for (unsigned i = 0; i < numRuns; ++i) {
        auto a = setA;
        if (i == numRuns - 1)
          CALLGRIND_TOGGLE_COLLECT;
        __sync_synchronize();
        auto res = a.eliminateExistentials();
        __sync_synchronize();

        if (i == numRuns - 1)
          CALLGRIND_TOGGLE_COLLECT;
      }
      CALLGRIND_STOP_INSTRUMENTATION;
    } else {
      std::cerr << "Unsupported operation " << op << "!\n";
      return 1;
    }
    consumeLine();
  }
}
