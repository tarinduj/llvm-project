#include "llvm/Support/FileSystem.h"
#include "mlir/Analysis/Presburger/Coalesce.h"
#include "mlir/Analysis/Presburger/Set.h"
// #include "mlir/Analysis/Presburger/TransprecSet.h"
#include "mlir/Dialect/Presburger/Parser.h"
#include "mlir/Analysis/Presburger/Presburger-impl.h"
#include "mlir/Analysis/Presburger/SafeInteger.h"
#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include "llvm/ADT/Optional.h"

event_collector rpcollector;

using namespace mlir;
using namespace mlir::presburger;

// unsigned TransprecSet::waterline = 0;

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

// void dumpStats(std::ofstream &f, TransprecSet &a) {
//   // a.dumpISL();
//   // return;
//   std::visit([&](auto &&set) {
//     unsigned ids = set.getNumDims() + set.getNumSyms(), nDivs = 0, nEqs = 0, nIneqs = 0, nBS = 0;
//     for (auto &bs : set.getBasicSets()) {
//       ids = std::max(ids, bs.getNumTotalDims());
//       nDivs += bs.getDivisions().size();
//       nEqs += bs.getNumEqualities();
//       nIneqs += bs.getNumInequalities();
//       nBS += 1;
//     }
//     f << ids << ' ' << nBS << ' ' << nDivs << ' ' << nIneqs << ' ' << nEqs << '\n';
//   }, a.setvar);
// }

void consumeLine(unsigned cnt = 1) {
  while (cnt--) {
    char str[1'000'000];
    std::cin.getline(str, 1'000'000);
    // std::cerr << "Consumed '" << str << "'\n";
  }
}

// TransprecSet getTransprecSetFromString(StringRef str) {
//   // std::cerr << "Read '" << str << "'\n";
//   if (auto set = setFromString<SafeInteger<int16_t>>(str))
//     return TransprecSet(*set);
//   else if (auto set = setFromString<SafeInteger<int64_t>>(str))
//     return TransprecSet(*set);
//   else if (auto set = setFromString<mpz_class>(str))
//     return TransprecSet(*set);
//   else
//     llvm_unreachable("Input did not fit in 128-bits!");
//   // return setFromString(str);
// }

template <typename Set>
Set getSetFromInput() {
  char str[1'000'000];
  std::cin.getline(str, 1'000'000);
  // if constexpr (std::is_same_v<Set, TransprecSet>) {
  //   return getTransprecSetFromString(str);
  // } else {
    if (auto set = setFromString<typename Set::UnderlyingInt>(str)) {
      return *set;
    } else
      llvm_unreachable("Input did not fit in specified precision!");
  // }
}

void consumeNewline() {
  char c;
  std::cin.get(c);
  if (c != '\n') {
    std::cerr << "Expected newline!\n";
    exit(1);
  }
}

template <typename Set, bool printAuxInfo>
void run(std::string op, std::string suffix, llvm::Optional<unsigned> maxWaterline) {
  std::cout << "Running " << op << '\n';
  if (!suffix.empty())
    assert(!printAuxInfo && "NYI");
  if (printAuxInfo)
    assert(!maxWaterline && "NYI");

  const unsigned numRuns = 5;
  unsigned numCases;
  std::cin >> numCases;
  consumeNewline();

  if (!suffix.empty()) {
    suffix = "_" + suffix;
    #ifdef ENABLE_SME
      suffix = suffix + "-sme";
    #endif  
  }
    
  std::ifstream fwaterlineIn("data/waterline_fpl_" + op + ".txt");
  std::ofstream fruntime("data/runtime" + suffix + "_" + op + ".txt");
  std::ofstream fcycles("data/cycles" + suffix + "_" + op + ".txt");
  std::ofstream finstructions("data/instructions" + suffix + "_" + op + ".txt");

  // td::ofstream fwaterline, fstat;
  std::error_code EC;
  llvm::raw_fd_ostream fout(printAuxInfo ? "data/outputs" + suffix + "_" + op + ".txt" : "data/empty_file_used_for_a_hack", EC, llvm::sys::fs::OpenFlags::OF_Append);
  if (printAuxInfo) {
    // fwaterline = std::ofstream("data/waterline_fpl_" + op + ".txt", std::ios_base::app);
    // fstat = std::ofstream("data/stats_fpl_" + op + ".txt", std::ios_base::app);
    if (EC) {
      std::cerr << "Could not open outputs_fpl_" + op + ".txt!\n";
      std::abort();
    }
    fout << numCases << '\n';
  }

  int fpexcepts = 0;
  for (unsigned j = 0; j < numCases; ++j) {
    std::feclearexcept(FE_ALL_EXCEPT); // Clear all exceptions

    std::vector<int> times(numRuns);
    std::vector<int> cycles(numRuns);
    std::vector<int> instructions(numRuns);

    // printing progress
    // if (j % 1 == 0)
      std::cerr << op << ' ' << j << '/' << numCases << '\n';

    if (maxWaterline) {
      // std::cout << "maxWaterline\n";
      unsigned waterline;
      fwaterlineIn >> waterline;
      if (waterline > *maxWaterline) {
        consumeLine();
        consumeLine();
        if (op == "subtract" || op == "union" || op == "intersect" || op == "equal")
          consumeLine();
        fruntime << "0\n";
        fcycles << "0\n";
        finstructions << "0\n";
        continue;
      }
    }

    // if constexpr (printAuxInfo)
    //   Set::waterline = 0;
    if (op == "empty") {
      Set setA = getSetFromInput<Set>();
      for (unsigned i = 0; i < numRuns; ++i) {
        auto a = setA;
        unsigned int dummy;
        rpcollector.start();
        volatile auto res = a.isIntegerEmpty();
        res = res;
        event_count allocate_count = rpcollector.end();
        times[i] = static_cast<int>(allocate_count.elapsed_ns());
        cycles[i] = static_cast<int>(allocate_count.cycles());
        instructions[i] = static_cast<int>(allocate_count.instructions());
        if (i == numRuns - 1) {
          std::sort(times.begin(), times.end());
          std::sort(cycles.begin(), cycles.end());
          std::sort(instructions.begin(), instructions.end());
          fruntime << times[numRuns/2] << '\n';
          fcycles << cycles[numRuns/2] << '\n';
          finstructions << instructions[numRuns/2] << '\n';
          if constexpr (printAuxInfo) {
  //           fwaterline << Set::waterline << '\n';
            fout << res << '\n';
          }
        }
      }
    } else if (op == "equal") {
      Set setA = getSetFromInput<Set>();
      Set setB = getSetFromInput<Set>();
      for (unsigned i = 0; i < numRuns; ++i) {
        auto a = setA;
        auto b = setB;
        unsigned int dummy;
        rpcollector.start();
        volatile auto res = Set::equal(a, b);
        res = res;
        event_count allocate_count = rpcollector.end();
        times[i] = static_cast<int>(allocate_count.elapsed_ns());
        cycles[i] = static_cast<int>(allocate_count.cycles());
        instructions[i] = static_cast<int>(allocate_count.instructions());
        if (i == numRuns - 1) {
          std::sort(times.begin(), times.end());
          std::sort(cycles.begin(), cycles.end());
          std::sort(instructions.begin(), instructions.end());
          fruntime << times[numRuns/2] << '\n';
          fcycles << cycles[numRuns/2] << '\n';
          finstructions << instructions[numRuns/2] << '\n';
          if constexpr (printAuxInfo) {
  //           fwaterline << Set::waterline << '\n';
            fout << res << '\n';
          }
        }
      }
    } else if (op == "union") {
      Set setA = getSetFromInput<Set>();
      Set setB = getSetFromInput<Set>();
      for (unsigned i = 0; i < numRuns; ++i) {
        auto a = setA;
        auto b = setB;
        unsigned int dummy;
        rpcollector.start();
        a.unionSet(b);
        event_count allocate_count = rpcollector.end();
        times[i] = static_cast<int>(allocate_count.elapsed_ns());
        cycles[i] = static_cast<int>(allocate_count.cycles());
        instructions[i] = static_cast<int>(allocate_count.instructions());
        if (i == numRuns - 1) {
          std::sort(times.begin(), times.end());
          std::sort(cycles.begin(), cycles.end());
          std::sort(instructions.begin(), instructions.end());
          fruntime << times[numRuns/2] << '\n';
          fcycles << cycles[numRuns/2] << '\n';
          finstructions << instructions[numRuns/2] << '\n';
          if constexpr (printAuxInfo) {
  //           fwaterline << Set::waterline << '\n';
  //           dumpStats(fstat, a);
            a.printISL(fout);
            fout << '\n';
          }
        }
      }
    } else if (op == "intersect") {
      Set setA = getSetFromInput<Set>();
      Set setB = getSetFromInput<Set>();
      for (unsigned i = 0; i < numRuns; ++i) {
        auto a = setA;
        auto b = setB;
        unsigned int dummy;
        auto start = std::chrono::high_resolution_clock::now();
        a.intersectSet(b);
        event_count allocate_count = rpcollector.end();
        times[i] = static_cast<int>(allocate_count.elapsed_ns());
        cycles[i] = static_cast<int>(allocate_count.cycles());
        instructions[i] = static_cast<int>(allocate_count.instructions());
        if (i == numRuns - 1) {
          std::sort(times.begin(), times.end());
          std::sort(cycles.begin(), cycles.end());
          std::sort(instructions.begin(), instructions.end());
          fruntime << times[numRuns/2] << '\n';
          fcycles << cycles[numRuns/2] << '\n';
          finstructions << instructions[numRuns/2] << '\n';
          if constexpr (printAuxInfo) {
  //           fwaterline << Set::waterline << '\n';
  //           dumpStats(fstat, a);
            a.printISL(fout);
            fout << '\n';
          }
        }
      }
    } else if (op == "subtract") {
      Set setA = getSetFromInput<Set>();
      Set setB = getSetFromInput<Set>();
      for (unsigned i = 0; i < numRuns; ++i) {
        
        #ifdef ENABLE_SME
          asm volatile("smstart za");

          // __asm__ __volatile__(
          //   "smstart sm                                               \n"
          //   "ptrue p0.s                                               \n"
          //   "cntw x9                                                   \n"
          //   "mov x15, #0                                               \n"
          //   "mov w11, #1                                               \n"
          //   "1:                                                       \n"
          //   "index z0.s, w11, #1                                       \n"
          //   "mov za0h.s[w15, 0], p0/m, z0.s                           \n"
          //   "add w11, w11, w9                                          \n"
          //   "add x15, x15, #1                                          \n"
          //   "cmp x15, x9                                               \n"
          //   "b.lo 1b                                                   \n"
          //   "smstop sm                                                \n"
          // );

          // SMEMatrix<int32_t> matrix(16,16);

          // int32_t x = matrix(2, 2);
          // int32_t x2 = matrix.at(2,2);

          // std::cout << "x: " << x << " " << x2 << std::endl;

          // matrix(2, 2) = 10;

          // int32_t ix = matrix(2, 2);

          // std::cout << "ix: " << ix << std::endl;

          // matrix.dump();
          // exit(0);


        #endif
        
        auto a = setA;
        auto b = setB;
        unsigned int dummy;
        rpcollector.start();
        a.subtract(b);
        event_count allocate_count = rpcollector.end();
        times[i] = static_cast<int>(allocate_count.elapsed_ns());
        cycles[i] = static_cast<int>(allocate_count.cycles());
        instructions[i] = static_cast<int>(allocate_count.instructions());
        if (i == numRuns - 1) {
          std::sort(times.begin(), times.end());
          std::sort(cycles.begin(), cycles.end());
          std::sort(instructions.begin(), instructions.end());
          fruntime << times[numRuns/2] << '\n';
          fcycles << cycles[numRuns/2] << '\n';
          finstructions << instructions[numRuns/2] << '\n';
          if constexpr (printAuxInfo) {
            // fwaterline << Set::waterline << '\n';
            // dumpStats(fstat, a);
            a.printISL(fout);
            fout << '\n';
          }
        }
      }
      
      #ifdef ENABLE_SME
        asm volatile("smstop za");
      #endif

    } else if (op == "coalesce") {
      Set setA = getSetFromInput<Set>();
      for (unsigned i = 0; i < numRuns; ++i) {
        auto a = setA;
        unsigned int dummy;
        rpcollector.start();
        Set res = coalesce(a);
        event_count allocate_count = rpcollector.end();
        times[i] = static_cast<int>(allocate_count.elapsed_ns());
        cycles[i] = static_cast<int>(allocate_count.cycles());
        instructions[i] = static_cast<int>(allocate_count.instructions());
        if (i == numRuns - 1) {
          std::sort(times.begin(), times.end());
          std::sort(cycles.begin(), cycles.end());
          std::sort(instructions.begin(), instructions.end());
          fruntime << times[numRuns/2] << '\n';
          fcycles << cycles[numRuns/2] << '\n';
          finstructions << instructions[numRuns/2] << '\n';
          if constexpr (printAuxInfo) {
  //           fwaterline << Set::waterline << '\n';
  //           dumpStats(fstat, res);
            res.printISL(fout);
            fout << '\n';
          }
        }
      }
    } else if (op == "complement") {
      Set setA = getSetFromInput<Set>();
      for (unsigned i = 0; i < numRuns; ++i) {
        auto a = setA;
        unsigned int dummy;
        rpcollector.start();
        auto res = Set::complement(a);
        event_count allocate_count = rpcollector.end();
        times[i] = static_cast<int>(allocate_count.elapsed_ns());
        cycles[i] = static_cast<int>(allocate_count.cycles());
        instructions[i] = static_cast<int>(allocate_count.instructions());
        if (i == numRuns - 1) {
          std::sort(times.begin(), times.end());
          std::sort(cycles.begin(), cycles.end());
          std::sort(instructions.begin(), instructions.end());
          fruntime << times[numRuns/2] << '\n';
          fcycles << cycles[numRuns/2] << '\n';
          finstructions << instructions[numRuns/2] << '\n';
          if constexpr (printAuxInfo) {
  //           fwaterline << Set::waterline << '\n';
  //           dumpStats(fstat, a);
            res.printISL(fout);
            fout << '\n';
          }
        }
      }
    } else if (op == "eliminate") {
      Set setA = getSetFromInput<Set>();
      for (unsigned i = 0; i < numRuns; ++i) {
        auto a = setA;
        unsigned int dummy;
        rpcollector.start();
        auto res = Set::eliminateExistentials(a);
        event_count allocate_count = rpcollector.end();
        times[i] = static_cast<int>(allocate_count.elapsed_ns());
        cycles[i] = static_cast<int>(allocate_count.cycles());
        instructions[i] = static_cast<int>(allocate_count.instructions());
        if (i == numRuns - 1) {
          std::sort(times.begin(), times.end());
          std::sort(cycles.begin(), cycles.end());
          std::sort(instructions.begin(), instructions.end());
          fruntime << times[numRuns/2] << '\n';
          fcycles << cycles[numRuns/2] << '\n';
          finstructions << instructions[numRuns/2] << '\n';
          if constexpr (printAuxInfo) {
  //           fwaterline << Set::waterline << '\n';
  //           dumpStats(fstat, a);
            a.printISL(fout);
            fout << '\n';
          }
        }
      }
    } else {
      std::cerr << "Unsupported operation " << op << "!\n";
      std::abort();
    }
    consumeLine();

    if (std::fetestexcept(FE_ALL_EXCEPT)) {
      // std::cerr << op << ' ' << j << '/' << numCases << '\n';
      // std::cerr << "Floating point exception!\n";
      // std::abort();
      fpexcepts++;
    }
  }
  if (fpexcepts)
    std::cerr << "Floating point exceptions: " << fpexcepts << '\n';
}

int main(int argc, char **argv) {
  // if (argc != 2 && argc != 3) {
  //   std::cerr << "usage: ./run-presburger <op> [precision:16/64/128/gmp/T]\nPass input to stdin.\n";
  //   return 1;
  // }

  const char* filename = "/Users/tarindujayatilaka/Documents/arm-sme/fpl-sme/benchmark/fpl/subtract.txt";
  std::ifstream infile(filename);

  if (!infile) {
      std::cerr << "Error opening file: " << filename << std::endl;
      return 1;
  }

  // Redirect std::cin to read from the file
  std::cin.rdbuf(infile.rdbuf());

  std::string op = argv[1];
  std::string prec = argc == 2 ? "T" : argv[2];

  if (prec == "16")
    run<PresburgerSet<int16_t>, true>(op, "16", {});
  // else if (prec == "64")
  //   run<PresburgerSet<int64_t>, true>(op, "64", {});
  else if (prec == "32")
    run<PresburgerSet<int32_t>, true>(op, "32", {});
}