#include "mlir/Analysis/Presburger/Coalesce.h"
#include "mlir/Analysis/Presburger/Set.h"
#include "mlir/Dialect/Presburger/Parser.h"
#include "mlir/Analysis/Presburger/Presburger-impl.h"
#include <iostream>
#include <string>
#include <x86intrin.h>

using namespace mlir;
using namespace mlir::presburger;

static TransprecSet setFromString(StringRef string) {
  ErrorCallback callback = [&](SMLoc loc, const Twine &message) {
    // This is a hack to make the Parser compile
    llvm::errs() << "Parsing error " << message << " at " << loc.getPointer()
                 << '\n';
    llvm::errs() << "invalid input " << string << '\n';

    llvm_unreachable("PARSING ERROR!!");
    MLIRContext context;
    return mlir::emitError(UnknownLoc::get(&context), message);
  };
  TransprecParser parser(string, callback);
  TransprecPresburgerParser setParser(parser);
  TransprecSet res;
  setParser.parsePresburgerSet(res);
  return res;
}

TransprecSet getSetFromInput() {
  char str[1'000'000];
  std::cin.getline(str, 1'000'000);
  return setFromString(str);
}

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cerr << "usage: ./run-presburger <op>\nPass input to stdin.\n";
    return 1;
  }

  const unsigned numRuns = 5;
  std::string op = argv[1];
  if (op == "empty") {
    TransprecSet setA = getSetFromInput();
    for (unsigned i = 0; i < numRuns; ++i) {
      auto a = setA;
      unsigned int dummy;
      unsigned long long start = __rdtscp(&dummy);
      auto res = a.isIntegerEmpty();
      unsigned long long end = __rdtscp(&dummy);
      if (SafeInteger<DefaultInt>::overflow) {
        std::cerr << "Overflow!\n";
        exit(1);
      }
      std::cerr << end - start << '\n';
      if (i == numRuns - 1)
        std::cerr << res << '\n';
    }
  } else if (op == "equal") {
    TransprecSet setA = getSetFromInput();
    TransprecSet setB = getSetFromInput();
    for (unsigned i = 0; i < numRuns; ++i) {
      auto a = setA;
      auto b = setB;
      unsigned int dummy;
      unsigned long long start = __rdtscp(&dummy);
      auto res = TransprecSet::equal(a, b);
      unsigned long long end = __rdtscp(&dummy);
      if (SafeInteger<DefaultInt>::overflow) {
        std::cerr << "Overflow!\n";
        exit(1);
      }
      std::cerr << end - start << '\n';
      if (i == numRuns - 1)
        llvm::errs() << res << '\n';
    }
  } else if (op == "union") {
    TransprecSet setA = getSetFromInput();
    TransprecSet setB = getSetFromInput();
    for (unsigned i = 0; i < numRuns; ++i) {
      auto a = setA;
      auto b = setB;
      unsigned int dummy;
      unsigned long long start = __rdtscp(&dummy);
      a.unionSet(std::move(b));
      unsigned long long end = __rdtscp(&dummy);
      if (SafeInteger<DefaultInt>::overflow) {
        std::cerr << "Overflow!\n";
        exit(1);
      }
      std::cerr << end - start << '\n';
      if (i == numRuns - 1)
        a.dumpISL();
    }
  } else if (op == "intersect") {
    TransprecSet setA = getSetFromInput();
    TransprecSet setB = getSetFromInput();
    for (unsigned i = 0; i < numRuns; ++i) {
      auto a = setA;
      auto b = setB;
      unsigned int dummy;
      unsigned long long start = __rdtscp(&dummy);
      a.intersectSet(std::move(b));
      unsigned long long end = __rdtscp(&dummy);
      if (SafeInteger<DefaultInt>::overflow) {
        std::cerr << "Overflow!\n";
        exit(1);
      }
      std::cerr << end - start << '\n';
      if (i == numRuns - 1)
        a.dumpISL();
    }
  } else if (op == "subtract") {
    TransprecSet setA = getSetFromInput();
    TransprecSet setB = getSetFromInput();
    for (unsigned i = 0; i < numRuns; ++i) {
      auto a = setA;
      auto b = setB;
      unsigned int dummy;
      unsigned long long start = __rdtscp(&dummy);
      a.subtract(std::move(b));
      unsigned long long end = __rdtscp(&dummy);
      if (SafeInteger<DefaultInt>::overflow) {
        std::cerr << "Overflow!\n";
        exit(1);
      }
      std::cerr << end - start << '\n';
      if (i == numRuns - 1)
        a.dumpISL();
    }
  } else if (op == "coalesce") {
    TransprecSet setA = getSetFromInput();
    for (unsigned i = 0; i < numRuns; ++i) {
      auto a = setA;
      unsigned int dummy;
      unsigned long long start = __rdtscp(&dummy);
      auto res = coalesce(a);
      unsigned long long end = __rdtscp(&dummy);
      if (SafeInteger<DefaultInt>::overflow) {
        std::cerr << "Overflow!\n";
        exit(1);
      }
      std::cerr << end - start << '\n';
      if (i == numRuns - 1)
        res.dumpISL();
    }
  } else if (op == "complement") {
    TransprecSet setA = getSetFromInput();
    for (unsigned i = 0; i < numRuns; ++i) {
      auto a = setA;
      unsigned int dummy;
      unsigned long long start = __rdtscp(&dummy);
      auto res = TransprecSet::complement(a);
      unsigned long long end = __rdtscp(&dummy);
      if (SafeInteger<DefaultInt>::overflow) {
        std::cerr << "Overflow!\n";
        exit(1);
      }
      std::cerr << end - start << '\n';
      if (i == numRuns - 1)
        res.dumpISL();
    }
  } else if (op == "eliminate") {
    TransprecSet setA = getSetFromInput();
    for (unsigned i = 0; i < numRuns; ++i) {
      auto a = setA;
      unsigned int dummy;
      unsigned long long start = __rdtscp(&dummy);
      auto res = TransprecSet::eliminateExistentials(a);
      unsigned long long end = __rdtscp(&dummy);
      if (SafeInteger<DefaultInt>::overflow) {
        std::cerr << "Overflow!\n";
        exit(1);
      }
      std::cerr << end - start << '\n';
      if (i == numRuns - 1)
        a.dumpISL();
    }
  } else {
    std::cout << "Unsupported operation " << op << "!\n";
    return 1;
  }
}
