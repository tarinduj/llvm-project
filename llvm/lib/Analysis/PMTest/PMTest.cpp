#include "llvm/IR/Function.h"
#include "llvm/Analysis/FunctionPropertiesAnalysis.h"
#include "llvm/IR/Module.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/FormatVariadic.h"
#include <fstream>
#include <functional>
#include <map>
#include <string> 
#include <set>
#include <vector>
#include <utility>

using namespace llvm;

namespace {
class PMTest {
  std::unique_ptr<DominatorTree> DT;
  std::unique_ptr<LoopInfo> LI;

  std::map<std::string, unsigned> passOptimizationLevel;

  bool beforePass(StringRef PassID, Any IR) {
    passOptimizationLevel["LoopDistributePass"] = 1;
    passOptimizationLevel["InjectTLIMappings"] = 1;
    passOptimizationLevel["LoopVectorizePass"] = 0;
    dbgs() << PassID << "\n";

    // set test
    std::set<int> testsset = {};

    for (int const& i : testsset)
    {
        dbgs() << i << ' ';
    }
    dbgs() << "\n";

    // vector test
    std::vector<std::pair < int, std::set<int> > > testvect;
    testvect.push_back(std::make_pair(0, std::set<int> {2,3}) );

    for (auto &element : testvect)
    {
        dbgs()  << element.first << "\t";
        for (int const& i : element.second )
        {
            dbgs() << i << ' ';
        }
        dbgs() << "\n";
    }
  


    if (any_isa<const Module *>(IR)) { //dbgs() << "Module\n"; 
    }

    else if (any_isa<const Function *>(IR)) { 
      //dbgs() << "*********** Function ***********\n"; 
      const Function *F = any_cast<const Function *>(IR);

      // dbgs() << "Pass Optimization Level: " << passOptimizationLevel[std::string(PassID)] << "\n";
      // dbgs() << "Function Optimization Level: " << F->getOptimizationLevel() << "\n";
      // if (passOptimizationLevel[std::string(PassID)] > F->getOptimizationLevel()){
      //   dbgs() << "PASS SKIPPED \n";
      //   return false;
      // }
      // return true;
    }

    else if (any_isa<const LazyCallGraph::SCC *>(IR)) { //dbgs() << "LazyCallGraph\n"; 
    }

    else if (any_isa<const Loop *>(IR)) { //dbgs() << "Loop\n"; 
    } 
    return true;
  }

  void afterPass(StringRef PassID, Any IR) {
    // dbgs() << PassID << "\n";
  }

  public:
    void registerCallbacks(PassInstrumentationCallbacks& PIC) {
      using namespace std::placeholders;
      PIC.registerBeforePassCallback(
        std::bind(&PMTest::beforePass, this, _1, _2));
      PIC.registerAfterPassCallback(
        std::bind(&PMTest::afterPass, this, _1, _2));
    }
};
} // end anonymous namespace

static PMTest PM;

extern "C" ::llvm::PassPluginLibraryInfo LLVM_ATTRIBUTE_WEAK
llvmGetPassPluginInfo() {
  return {
    LLVM_PLUGIN_API_VERSION, "NewPMPrintAfterInstrumentPlugin", "v0.1",
    [](PassBuilder& PB) {
      auto& PIC = *PB.getPassInstrumentationCallbacks();
      PM.registerCallbacks(PIC);
    }
  };
}

