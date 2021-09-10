#include "mlir/Analysis/Presburger/Coalesce.h"
#include "mlir/Analysis/Presburger/Set.h"
#include "mlir/Analysis/Presburger/TransprecSet.h"
#include "mlir/Dialect/Presburger/Parser.h"
#include "mlir/Analysis/Presburger/Presburger-impl.h"
#include "mlir/Analysis/Presburger/perf_event_open.h"
#include <iostream>
#include <string>
#include <x86intrin.h>
#include <fstream>

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

std::ofstream fhw;
void print_to_fhw(uint64_t x) {
  fhw << x << '\n';
}

int main(int argc, char **argv) {
  if (argc != 3) {
    std::cerr << "usage: ./run-presburger <op>\nPass input to stdin.\n";
    return 1;
  }

  const unsigned numRuns = 5;
  std::string op = argv[1], modeStr = argv[2];
  int mode = atoi(argv[2]);

  unsigned numCases = 1;
  // std::cin >> numCases;

  init_perf_fds(mode);
  // consumeNewline();
  fhw = std::ofstream("data/hw_fpl_" + op + "_" + modeStr + ".txt", std::ios_base::app);
  for (unsigned j = 0; j < numCases; ++j) {
    // if (j % 50000 == 0)
    //   std::cerr << op << ' ' << j << '/' << numCases << '\n';

    TransprecSet::waterline = 0;
    if (op == "empty") {
      TransprecSet setA = getSetFromInput();
      for (unsigned i = 0; i < numRuns; ++i) {

        auto a = setA;
        if (i == numRuns - 1)
          reset_and_enable_all();
        volatile auto res = a.isIntegerEmpty();
        if (i == numRuns - 1)
          disable_all_and_print_counts(&print_to_fhw);
        res = res;

      }
    } else if (op == "equal") {
      TransprecSet setA = getSetFromInput();
      TransprecSet setB = getSetFromInput();
      for (unsigned i = 0; i < numRuns; ++i) {

        auto a = setA;
        auto b = setB;
        if (i == numRuns - 1)
          reset_and_enable_all();
        volatile auto res = a.equal(b);
        if (i == numRuns - 1)
          disable_all_and_print_counts(&print_to_fhw);
        res = res;

      }
    } else if (op == "union") {
      TransprecSet setA = getSetFromInput();
      TransprecSet setB = getSetFromInput();
      for (unsigned i = 0; i < numRuns; ++i) {

        auto a = setA;
        auto b = setB;
        if (i == numRuns - 1)
          reset_and_enable_all();
        a.unionSet(b);

        if (i == numRuns - 1)
          disable_all_and_print_counts(&print_to_fhw);
      }
    } else if (op == "intersect") {
      TransprecSet setA = getSetFromInput();
      TransprecSet setB = getSetFromInput();
      for (unsigned i = 0; i < numRuns; ++i) {

        auto a = setA;
        auto b = setB;
        if (i == numRuns - 1)
          reset_and_enable_all();
        a.intersectSet(b);

        if (i == numRuns - 1)
          disable_all_and_print_counts(&print_to_fhw);
      }
    } else if (op == "subtract") {
      TransprecSet setA = getSetFromInput();
      TransprecSet setB = getSetFromInput();
      for (unsigned i = 0; i < numRuns; ++i) {

        auto a = setA;
        auto b = setB;
        if (i == numRuns - 1)
          reset_and_enable_all();
        a.subtract(b);

        if (i == numRuns - 1)
          disable_all_and_print_counts(&print_to_fhw);
      }
    } else if (op == "coalesce") {
      TransprecSet setA = getSetFromInput();
      for (unsigned i = 0; i < numRuns; ++i) {

        auto a = setA;
        if (i == numRuns - 1)
          reset_and_enable_all();
        auto res = a.coalesce();

        if (i == numRuns - 1)
          disable_all_and_print_counts(&print_to_fhw);
      }
    } else if (op == "complement") {
      TransprecSet setA = getSetFromInput();
      for (unsigned i = 0; i < numRuns; ++i) {

        auto a = setA;
        if (i == numRuns - 1)
          reset_and_enable_all();
        auto res = a.complement();

        if (i == numRuns - 1)
          disable_all_and_print_counts(&print_to_fhw);
      }
    } else if (op == "eliminate") {
      TransprecSet setA = getSetFromInput();
      for (unsigned i = 0; i < numRuns; ++i) {
        auto a = setA;
        if (i == numRuns - 1)
          reset_and_enable_all();
        auto res = a.eliminateExistentials();

        if (i == numRuns - 1)
          disable_all_and_print_counts(&print_to_fhw);
      }
    } else {
      std::cerr << "Unsupported operation " << op << "!\n";
      return 1;
    }
    consumeLine();
  }
}
