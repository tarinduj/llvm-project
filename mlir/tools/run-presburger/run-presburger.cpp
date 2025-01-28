#include "llvm/Support/FileSystem.h"
#include "mlir/Analysis/Presburger/Coalesce.h"
#include "mlir/Analysis/Presburger/Set.h"
// #include "mlir/Analysis/Presburger/TransprecSet.h"
#include "mlir/Dialect/Presburger/Parser.h"
#include "mlir/Analysis/Presburger/Presburger-impl.h"
#include <iostream>
#include <string>
#include <fstream>
#include "llvm/ADT/Optional.h"
#include <cfenv>

using namespace mlir;
using namespace mlir::presburger;

// unsigned TransprecSet::waterline = 0;

#include <csignal>
#include <csetjmp>

extern bool VALIDINPUT;
extern bool PIVOTCALLED;
extern int MAXMATSIZE;

sigjmp_buf jumpBuffer;

void segfaultHandler(int signal) {
    std::cerr << "Segmentation fault caught in function call!" << std::endl;
    siglongjmp(jumpBuffer, 1); // Jump back to the safe point
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

void consumeLine(unsigned cnt = 1, std::string *savedInput = nullptr) {
  while (cnt--) {
    char str[1'000'000];
    std::cin.getline(str, 1'000'000);
    if (savedInput) {
      *savedInput = std::string(str); // Save the input to the provided string pointer
    }
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
Set getSetFromInput(std::string *savedInput = nullptr) {
  char str[1'000'000];
  std::cin.getline(str, 1'000'000);
  // if constexpr (std::is_same_v<Set, TransprecSet>) {
  //   return getTransprecSetFromString(str);
  // } else {
  if (savedInput) {
    *savedInput = std::string(str); // Save the input to the provided string pointer
  }

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

  const unsigned numRuns = 1;
  unsigned numCases;
  std::cin >> numCases;
  consumeNewline();



  if (!suffix.empty())
    suffix = "_" + suffix;
  std::ifstream fwaterlineIn("data/waterline_fpl_" + op + ".txt");
  std::ofstream fruntime("data/runtime_fpl" + suffix + "_" + op + ".txt");

  std::ofstream filtered_bench("benchmark/fpl-sme/" + op + ".txt");

  std::ifstream isl_bench("benchmark/isl/" + op + ".txt");
  std::ofstream filtered_isl_bench("benchmark/isl-sme/" + op + ".txt");
  
  filtered_bench << numCases << '\n';
  filtered_isl_bench << numCases << '\n';

  std::ofstream matrix_size("data/matrix-size-" + op + ".txt");

  unsigned filteredNumCases = 0;

  // consume numcases
  std::string line;
  std::getline(isl_bench, line);

  std::error_code EC;
  llvm::raw_fd_ostream fout(printAuxInfo ? "data/outputs_fpl" + suffix + "_" + op + ".txt" : "data/empty_file_used_for_a_hack", EC, llvm::sys::fs::OpenFlags::OF_Append);
  if (printAuxInfo) {
    // fwaterline = std::ofstream("data/waterline_fpl_" + op + ".txt", std::ios_base::app);
    // fstat = std::ofstream("data/stats_fpl_" + op + ".txt", std::ios_base::app);
    if (EC) {
      std::cerr << "Could not open outputs_fpl_" + op + ".txt!\n";
      std::abort();
    }
    fout << numCases << '\n';
  }


  for (unsigned j = 0; j < numCases; ++j) {
    std::feclearexcept(FE_ALL_EXCEPT); // Clear all exceptions
    
    VALIDINPUT = true;
    PIVOTCALLED = false;
    MAXMATSIZE = 0;

    std::string inputStringA;
    std::string inputStringB;

    std::string currentISLCase;
    while (true) {
      if (!std::getline(isl_bench, line)) {
        break;
      }
      if (line == "=====") {
        currentISLCase += line + '\n';
        break;
      }
      currentISLCase += line + '\n';
    }

    int times[numRuns];
    // printing progress
    if (j % 1 == 0)
      std::cerr << op << ' ' << j << '/' << numCases << '\n';

    if (maxWaterline) {
      unsigned waterline;
      fwaterlineIn >> waterline;
      if (waterline > *maxWaterline) {
        consumeLine();
        consumeLine();
        if (op == "subtract" || op == "union" || op == "intersect" || op == "equal")
          consumeLine();
        fruntime << "0\n";
        continue;
      }
    }

    // if constexpr (printAuxInfo)
    //   Set::waterline = 0;
    if (op == "empty") {
      Set setA = getSetFromInput<Set>(&inputStringA);
      for (unsigned i = 0; i < numRuns; ++i) {
        auto a = setA;
        unsigned int dummy;
        auto start = std::chrono::high_resolution_clock::now();
        volatile auto res = a.isIntegerEmpty();
        res = res;
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
times[i] = static_cast<int>(duration);
        if (i == numRuns - 1) {
          std::sort(times, times + numRuns);
          fruntime << times[numRuns/2] << '\n';
          if constexpr (printAuxInfo) {
  //           fwaterline << Set::waterline << '\n';
            fout << res << '\n';
          }
        }
      }
    } else if (op == "equal") {
      Set setA = getSetFromInput<Set>(&inputStringA);
      Set setB = getSetFromInput<Set>(&inputStringB);
      for (unsigned i = 0; i < numRuns; ++i) {
        auto a = setA;
        auto b = setB;
        unsigned int dummy;
        auto start = std::chrono::high_resolution_clock::now();
        volatile auto res = Set::equal(a, b);
        res = res;
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
times[i] = static_cast<int>(duration);
        if (i == numRuns - 1) {
          std::sort(times, times + numRuns);
          fruntime << times[numRuns/2] << '\n';
          if constexpr (printAuxInfo) {
  //           fwaterline << Set::waterline << '\n';
            fout << res << '\n';
          }
        }
      }
    } else if (op == "union") {
      Set setA = getSetFromInput<Set>(&inputStringA);
      Set setB = getSetFromInput<Set>(&inputStringB);
      for (unsigned i = 0; i < numRuns; ++i) {
        auto a = setA;
        auto b = setB;
        unsigned int dummy;
        auto start = std::chrono::high_resolution_clock::now();
        a.unionSet(b);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
times[i] = static_cast<int>(duration);
        if (i == numRuns - 1) {
          std::sort(times, times + numRuns);
          fruntime << times[numRuns/2] << '\n';
          if constexpr (printAuxInfo) {
  //           fwaterline << Set::waterline << '\n';
  //           dumpStats(fstat, a);
            a.printISL(fout);
            fout << '\n';
          }
        }
      }
    } else if (op == "intersect") {
      Set setA = getSetFromInput<Set>(&inputStringA);
      Set setB = getSetFromInput<Set>(&inputStringB);
      for (unsigned i = 0; i < numRuns; ++i) {
        auto a = setA;
        auto b = setB;
        unsigned int dummy;
        auto start = std::chrono::high_resolution_clock::now();
        a.intersectSet(b);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
times[i] = static_cast<int>(duration);
        if (i == numRuns - 1) {
          std::sort(times, times + numRuns);
          fruntime << times[numRuns/2] << '\n';
          if constexpr (printAuxInfo) {
  //           fwaterline << Set::waterline << '\n';
  //           dumpStats(fstat, a);
            a.printISL(fout);
            fout << '\n';
          }
        }
      }
    } else if (op == "subtract") {
      Set setA = getSetFromInput<Set>(&inputStringA);
      Set setB = getSetFromInput<Set>(&inputStringB);
      for (unsigned i = 0; i < numRuns; ++i) {
        auto a = setA;
        auto b = setB;
        unsigned int dummy;
        auto start = std::chrono::high_resolution_clock::now();
        a.subtract(b);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
times[i] = static_cast<int>(duration);
        if (i == numRuns - 1) {
          std::sort(times, times + numRuns);
          fruntime << times[numRuns/2] << '\n';
          if constexpr (printAuxInfo) {
  //           fwaterline << Set::waterline << '\n';
  //           dumpStats(fstat, a);
            a.printISL(fout);
            fout << '\n';
          }
        }
      }
    } else if (op == "coalesce") {
      Set setA = getSetFromInput<Set>(&inputStringA);
      for (unsigned i = 0; i < numRuns; ++i) {
        auto a = setA;
        unsigned int dummy;
        auto start = std::chrono::high_resolution_clock::now();
        Set res = coalesce(a);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        times[i] = static_cast<int>(duration);
        if (i == numRuns - 1) {
          std::sort(times, times + numRuns);
          fruntime << times[numRuns/2] << '\n';
          if constexpr (printAuxInfo) {
  //           fwaterline << Set::waterline << '\n';
  //           dumpStats(fstat, res);
            res.printISL(fout);
            fout << '\n';
          }
        }
      }
    } else if (op == "complement") {
      Set setA = getSetFromInput<Set>(&inputStringA);
      for (unsigned i = 0; i < numRuns; ++i) {
        auto a = setA;
        unsigned int dummy;
        auto start = std::chrono::high_resolution_clock::now();
        auto res = Set::complement(a);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
times[i] = static_cast<int>(duration);
        if (i == numRuns - 1) {
          std::sort(times, times + numRuns);
          fruntime << times[numRuns/2] << '\n';
          if constexpr (printAuxInfo) {
  //           fwaterline << Set::waterline << '\n';
  //           dumpStats(fstat, a);
            res.printISL(fout);
            fout << '\n';
          }
        }
      }
    } else if (op == "eliminate") {
      Set setA = getSetFromInput<Set>(&inputStringA);
      for (unsigned i = 0; i < numRuns; ++i) {
        auto a = setA;
        unsigned int dummy;
        auto start = std::chrono::high_resolution_clock::now();
        std::signal(SIGSEGV, segfaultHandler);
        if (sigsetjmp(jumpBuffer, 1) == 0) { 

          try {
            auto res = Set::eliminateExistentials(a);
          } catch (std::runtime_error &e) {
            std::cerr << "Error: " << e.what() << std::endl;
            VALIDINPUT = false;
          }

        } else {
        // Segmentation fault recovery
          std::cerr << "Recovered from segmentation fault in riskyFunction." << std::endl;
          VALIDINPUT = false;
        } 
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
times[i] = static_cast<int>(duration);
        if (i == numRuns - 1) {
          std::sort(times, times + numRuns);
          fruntime << times[numRuns/2] << '\n';
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

    std::string outputString;
    consumeLine(1, &outputString);

    if (std::fetestexcept(FE_ALL_EXCEPT)) {
      VALIDINPUT = false;
    } 

    if (!PIVOTCALLED) {
      VALIDINPUT = false;
    }

    if (!VALIDINPUT) {
      // std::cout << "INVALID INPUT\n";
    } else {
      filteredNumCases++;
      filtered_bench << inputStringA << '\n';
      if (!inputStringB.empty()) 
        filtered_bench << inputStringB << '\n';
      filtered_bench << outputString << '\n';

      filtered_isl_bench << currentISLCase;

      // std::cout << "MAXMAT: " << MAXMATSIZE << "\n";
      matrix_size << MAXMATSIZE << "\n";
    }
  }

  filtered_bench.close();
  filtered_isl_bench.close();

  std::fstream file("benchmark/fpl-sme/" + op + ".txt", std::ios::in | std::ios::out);
  if (file.is_open()) {
      file.seekp(0); // Go to the beginning of the file
      file << filteredNumCases << '\n'; // Overwrite the number of cases
      file.close();
  }
  std::fstream isl_file("benchmark/isl-sme/" + op + ".txt", std::ios::in | std::ios::out);
  if (isl_file.is_open()) {
      isl_file.seekp(0); // Go to the beginning of the file
      isl_file << filteredNumCases << '\n'; // Overwrite the number of cases
      isl_file.close();
  }
  
}

int main(int argc, char **argv) {
  if (argc != 2 && argc != 3) {
    std::cerr << "usage: ./run-presburger <op> [precision:16/64/128/gmp/T]\nPass input to stdin.\n";
    return 1;
  }

  std::string op = argv[1];
  std::string prec = argc == 2 ? "T" : argv[2];
  if (prec == "16")
    run<PresburgerSet<int16_t>, false>(op, "16", 0);
  else if (prec == "64")
    run<PresburgerSet<int64_t>, false>(op, "64", 1);
  // else if (prec == "gmp")
  //   run<PresburgerSet<mpz_class>, false>(op, "gmp", 3);
  // else if (prec == "T")
  //   run<TransprecSet, true>(op, "", {});
  else if (prec == "f32")
    run<PresburgerSet<float_t>, true>(op, "f32", {});
}

