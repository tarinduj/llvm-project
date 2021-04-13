#include "mlir/Analysis/Presburger/Coalesce.h"
#include "mlir/Analysis/Presburger/Set.h"
#include "mlir/Analysis/Presburger/TransprecSet.h"
#include "mlir/Dialect/Presburger/Parser.h"
#include "mlir/Analysis/Presburger/Presburger-impl.h"
#include <iostream>
#include <string>
#include <x86intrin.h>

using namespace mlir;
using namespace mlir::presburger;

template <typename Int>
Optional<PresburgerSet<Int>> setFromString(StringRef string) {
  ErrorCallback callback = [&](SMLoc loc, const Twine &message) {
    // This is a hack to make the Parser compile
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

void dumpStats(TransprecSet &a) {
  std::visit([&](auto &&set) {
    unsigned ids = set.getNumDims() + set.getNumSyms(), nDivs = 0, nEqs = 0, nIneqs = 0, nBS = 0;
    for (auto &bs : set.getBasicSets()) {
      ids = std::max(ids, bs.getNumTotalDims());
      nDivs += bs.getDivisions().size();
      nEqs += bs.getNumEqualities();
      nIneqs += bs.getNumInequalities();
      nBS += 1;
    }
    std::cerr << ids << ' ' << nBS << ' ' << nDivs << ' ' << nIneqs << ' ' << nEqs << '\n';
  }, a.setvar);
}


TransprecSet getSetFromInput() {
  char str[1'000'000];
  std::cin.getline(str, 1'000'000);
  if (auto set = setFromString<int16_t>(str))
    return TransprecSet(*set);
  else if (auto set = setFromString<int64_t>(str))
    return TransprecSet(*set);
  else if (auto set = setFromString<mpz_class>(str))
    return TransprecSet(*set);
  else
    llvm_unreachable("Input did not fit in 128-bits!");
  // return setFromString(str);
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
      auto res = a.equal(b);
      unsigned long long end = __rdtscp(&dummy);
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
      a.unionSet(b);
      unsigned long long end = __rdtscp(&dummy);
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
      a.intersectSet(b);
      unsigned long long end = __rdtscp(&dummy);
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
      a.subtract(b);
      unsigned long long end = __rdtscp(&dummy);
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
      auto res = a.coalesce();
      unsigned long long end = __rdtscp(&dummy);
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
      auto res = a.complement();
      unsigned long long end = __rdtscp(&dummy);
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
      auto res = a.eliminateExistentials();
      unsigned long long end = __rdtscp(&dummy);
      std::cerr << end - start << '\n';
      if (i == numRuns - 1)
        a.dumpISL();
    }
  } else {
    std::cout << "Unsupported operation " << op << "!\n";
    return 1;
  }
}
