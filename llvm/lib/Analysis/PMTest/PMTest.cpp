#include "llvm/IR/Function.h"
#include "llvm/Analysis/FunctionPropertiesAnalysis.h"
#include "llvm/IR/Module.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/ADT/SmallSet.h"
#include <fstream>
#include <functional>
#include <map>
#include <string> 
#include <set>
#include <vector>
#include <utility>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>

using namespace llvm;

namespace {
enum OptimizationLevel { O0, O1, O2, O3 };
class PMTest {
  std::unique_ptr<DominatorTree> DT;
  std::unique_ptr<LoopInfo> LI;

  std::map<std::string, unsigned> passOptimizationLevel;

  bool beforePass(StringRef PassID, Any IR) {
    passOptimizationLevel["LoopDistributePass"] = 1;
    passOptimizationLevel["InjectTLIMappings"] = 1;
    passOptimizationLevel["LoopVectorizePass"] = 0;
    dbgs() << PassID << "\n";

    //testPass({O1, O2});

    // set test
    SmallSet<OptimizationLevel, 4> testsset;
    testsset.insert(O1);

    // for (unsigned const& i : testsset)
    // {
    //     dbgs() << i << ' ';
    // }
    // dbgs() << "\n";

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

    // else if (any_isa<const Function *>(IR)) { 
    //   //dbgs() << "*********** Function ***********\n"; 
    //   const Function *F = any_cast<const Function *>(IR);
    //   const std::string FunctionName = F->getName().str();
    //   dbgs() << "REAL FUNCTION: " << FunctionName << "\n";

    //   const unsigned FunctionOptLevel = F->getOptimizationLevel();
      
    //   if (FunctionOptLevel == 0) {
    //     dbgs() << "OOPS\n";

    //     std::ifstream ip("lookuptable.csv");
    //     if (!ip.is_open())
    //         dbgs() << "File not found" << "\n";
    //     std::string line;
    //     while (std::getline(ip, line))
    //     {
    //       //dbgs() << "LINE: " << line << "\n";
    //       std::istringstream iss(line);
    //       std::string id, name, olevel;

    //       if (std::getline(iss, id, ',') &&
    //           std::getline(iss, name, ',') &&
    //           std::getline(iss, olevel))
    //       {
    //         //dbgs() << "NAME: " << name << "\n";
    //         //dbgs() << "OLEVEL: " << olevel << "\n";
    //         char *endp = nullptr;
    //         if (name.c_str() != endp && name == FunctionName) {
    //             dbgs() << "FOUND A MATACH\n";
    //             unsigned level = std::atoi(olevel.substr(1).c_str());
    //             dbgs() << "LEVEL: " << level << "\n";
    //             //F->setOptimizationLevel(level);
                
    //         }
    //       }
    //     }
    //   } else {
    //     dbgs() << "YAY\n";
    //   };




    //   // dbgs() << "Pass Optimization Level: " << passOptimizationLevel[std::string(PassID)] << "\n";
    //   // dbgs() << "Function Optimization Level: " << F->getOptimizationLevel() << "\n";
    //   // if (passOptimizationLevel[std::string(PassID)] > F->getOptimizationLevel()){
    //   //   dbgs() << "PASS SKIPPED \n";
    //   //   return false;
    //   // }
    //   // return true;
    // }

    else if (any_isa<const LazyCallGraph::SCC *>(IR)) { //dbgs() << "LazyCallGraph\n"; 
    }

    else if (any_isa<const Loop *>(IR)) { //dbgs() << "Loop\n"; 
    } 
    return true;
  }

  void testPass(SmallSet<OptimizationLevel, 4> OptimizationLevels = {}){
    dbgs() << "TEST SUCCESS!\n";
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

