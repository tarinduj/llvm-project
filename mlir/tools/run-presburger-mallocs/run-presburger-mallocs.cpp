#include "mlir/Analysis/Presburger/Coalesce.h"
#include "mlir/Analysis/Presburger/Set.h"
#include "mlir/Analysis/Presburger/TransprecSet.h"
#include "mlir/Dialect/Presburger/Parser.h"
#include "mlir/Analysis/Presburger/Presburger-impl.h"
#include "mlir/Analysis/Presburger/perf_event_open.h"
#include "malloc_count.h"
#include <iostream>
#include <string>
#include <x86intrin.h>
#include <fstream>

using namespace mlir;
using namespace mlir::presburger;

unsigned TransprecSet::waterline = 0;

void print_malloc_counts(std::ostream &out) {
  malloc_counts counts = malloc_count_get_count();
  out << counts.allocs << '\n' << counts.frees << '\n';
}

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

void dumpStats(std::ofstream &f, TransprecSet &a) {
  // a.dumpISL();
  // return;
  std::visit([&](auto &&set) {
    unsigned ids = set.getNumDims() + set.getNumSyms(), nDivs = 0, nEqs = 0, nIneqs = 0, nBS = 0;
    for (auto &bs : set.getBasicSets()) {
      ids = std::max(ids, bs.getNumTotalDims());
      nDivs += bs.getDivisions().size();
      nEqs += bs.getNumEqualities();
      nIneqs += bs.getNumInequalities();
      nBS += 1;
    }
    f << ids << ' ' << nBS << ' ' << nDivs << ' ' << nIneqs << ' ' << nEqs << '\n';
  }, a.setvar);
}

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cerr << "usage: ./run-presburger <op>\nPass input to stdin.\n";
    return 1;
  }

  const unsigned numRuns = 1;
  std::string op = argv[1];

  unsigned numCases;
  std::cin >> numCases;
  consumeNewline();

  malloc_count_init();

  std::ofstream fmallocs("data/mallocs_fpl_" + op + ".txt");

  std::error_code EC;
  llvm::raw_fd_ostream fout("data/outputs_fpl_" + op + ".txt", EC);
  if (EC) {
    std::cerr << "Could not open outputs_fpl_" + op + ".txt!\n";
    std::abort();
  }
  fout << numCases << '\n';

  std::ofstream fwaterline = std::ofstream("data/waterline_fpl_" + op + ".txt");
  std::ofstream fstat = std::ofstream("data/stats_fpl_" + op + ".txt");

  for (unsigned j = 0; j < numCases; ++j) {
    if (j % 50000 == 0)
      std::cerr << op << ' ' << j << '/' << numCases << '\n';

    TransprecSet::waterline = 0;
    if (op == "empty") {
      TransprecSet setA = getSetFromInput();
      for (unsigned i = 0; i < numRuns; ++i) {

        auto a = setA;
        if (i == numRuns - 1)
          malloc_count_reset_count();
        __sync_synchronize();
        volatile auto res = a.isIntegerEmpty();
        __sync_synchronize();
        if (i == numRuns - 1) {
          fwaterline << TransprecSet::waterline << '\n';
          fout << res << '\n';
          print_malloc_counts(fmallocs);
        }
        res = res;

      }
    } else if (op == "equal") {
      TransprecSet setA = getSetFromInput();
      TransprecSet setB = getSetFromInput();
      for (unsigned i = 0; i < numRuns; ++i) {

        auto a = setA;
        auto b = setB;
        if (i == numRuns - 1)
          malloc_count_reset_count();
        __sync_synchronize();
        volatile auto res = a.equal(b);
        __sync_synchronize();
        if (i == numRuns - 1) {
          fwaterline << TransprecSet::waterline << '\n';
          fout << res << '\n';
          print_malloc_counts(fmallocs);
        }
        res = res;

      }
    } else if (op == "union") {
      TransprecSet setA = getSetFromInput();
      TransprecSet setB = getSetFromInput();
      for (unsigned i = 0; i < numRuns; ++i) {

        auto a = setA;
        auto b = setB;
        if (i == numRuns - 1)
          malloc_count_reset_count();
        __sync_synchronize();
        a.unionSet(b);
        __sync_synchronize();

        if (i == numRuns - 1) {
          fwaterline << TransprecSet::waterline << '\n';
          dumpStats(fstat, a);
          a.printISL(fout);
          print_malloc_counts(fmallocs);
        }
      }
    } else if (op == "intersect") {
      TransprecSet setA = getSetFromInput();
      TransprecSet setB = getSetFromInput();
      for (unsigned i = 0; i < numRuns; ++i) {

        auto a = setA;
        auto b = setB;
        if (i == numRuns - 1)
          malloc_count_reset_count();
        __sync_synchronize();
        a.intersectSet(b);
        __sync_synchronize();

        if (i == numRuns - 1) {
          fwaterline << TransprecSet::waterline << '\n';
          dumpStats(fstat, a);
          a.printISL(fout);
          print_malloc_counts(fmallocs);
        }
      }
    } else if (op == "subtract") {
      TransprecSet setA = getSetFromInput();
      TransprecSet setB = getSetFromInput();
      for (unsigned i = 0; i < numRuns; ++i) {

        auto a = setA;
        auto b = setB;
        if (i == numRuns - 1)
          malloc_count_reset_count();
        __sync_synchronize();
        a.subtract(b);
        __sync_synchronize();

        if (i == numRuns - 1) {
          fwaterline << TransprecSet::waterline << '\n';
          dumpStats(fstat, a);
          a.printISL(fout);
          print_malloc_counts(fmallocs);
        }
      }
    } else if (op == "coalesce") {
      TransprecSet setA = getSetFromInput();
      for (unsigned i = 0; i < numRuns; ++i) {

        auto a = setA;
        if (i == numRuns - 1)
          malloc_count_reset_count();
        __sync_synchronize();
        auto res = a.coalesce();
        __sync_synchronize();

        if (i == numRuns - 1) {
          fwaterline << TransprecSet::waterline << '\n';
          dumpStats(fstat, res);
          res.printISL(fout);
          print_malloc_counts(fmallocs);
        }
      }
    } else if (op == "complement") {
      TransprecSet setA = getSetFromInput();
      for (unsigned i = 0; i < numRuns; ++i) {

        auto a = setA;
        if (i == numRuns - 1)
          malloc_count_reset_count();
        __sync_synchronize();
        auto res = a.complement();
        __sync_synchronize();

        if (i == numRuns - 1) {
          fwaterline << TransprecSet::waterline << '\n';
          dumpStats(fstat, res);
          res.printISL(fout);
          print_malloc_counts(fmallocs);
        }
      }
    } else if (op == "eliminate") {
      TransprecSet setA = getSetFromInput();
      for (unsigned i = 0; i < numRuns; ++i) {
        auto a = setA;
        if (i == numRuns - 1)
          malloc_count_reset_count();
        __sync_synchronize();
        auto res = a.eliminateExistentials();
        __sync_synchronize();

        if (i == numRuns - 1) {
          fwaterline << TransprecSet::waterline << '\n';
          dumpStats(fstat, res);
          res.printISL(fout);
          print_malloc_counts(fmallocs);
        }
      }
    } else {
      std::cerr << "Unsupported operation " << op << "!\n";
      return 1;
    }
    consumeLine();
  }
}
