#include "llvm/IR/Function.h"
#include "llvm/Analysis/ML/FunctionPropertiesAnalysis.h"
#include "llvm/IR/Module.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/FormatVariadic.h"
#include <fstream>
#include <functional>

using namespace llvm;

namespace {
class FunctionPropertiesAnalysisPassInstrument {
  std::unique_ptr<DominatorTree> DT;
  std::unique_ptr<LoopInfo> LI;

  FunctionPropertiesInfo buildFPI(Function *F) {
    //dbgs() << "test BITCH 1\n";
    DT.reset(new DominatorTree(*F));
    //dbgs() << "test BITCH 2\n";
    LI.reset(new LoopInfo(*DT));
    //dbgs() << "test BITCH 3\n";
    return FunctionPropertiesInfo::getFunctionPropertiesInfo(*F, *LI);
  }

  void afterPass(StringRef PassID, Any IR) {
    if (any_isa<const Module *>(IR)) {
      const Module *M = any_cast<const Module *>(IR);
      
      //dbgs() << formatv("*** Module Pass: {0} ***\n", PassID);

      for (const auto &F : M->functions()) {
        
        std::string moduleName = Twine(M->getName()).str();
        
        std::string delimiter = "CTMark/";
        moduleName.erase(0, moduleName.find(delimiter) + delimiter.length());
        replace(moduleName.begin(), moduleName.end(), '/', '_');
        std::string outFilePath = "/Users/tarindujayatilaka/Documents/LLVM/results/CTMark/" + moduleName + ".txt";
        
        std::ofstream outFile;
        outFile.open(outFilePath, std::ios_base::app);

        std::string banner = formatv("*** Module Pass # {0} # {1} #\n", PassID, F.getName().str());
        raw_string_ostream* outString = new raw_string_ostream(banner);

        FunctionPropertiesInfo FPI = buildFPI(const_cast<Function *>(&F));
        FPI.print(*outString);

        outFile  << outString->str();

        outString->flush();
        outFile.close();
      } 
    }

    else if (any_isa<const Function *>(IR)) {
      const Function *F = any_cast<const Function *>(IR);
      const Module *M = F->getParent();
      
      //dbgs() << formatv("*** Function Pass: {0} ***\n", PassID);

      std::string moduleName = Twine(M->getName()).str();
      
      std::string delimiter = "CTMark/";
      //dbgs() << "moduleNameBef F:" << moduleName << "\n";
      moduleName.erase(0, moduleName.find(delimiter) + delimiter.length());
      replace(moduleName.begin(), moduleName.end(), '/', '_');
      //dbgs() << "moduleNameAft F:" << moduleName << "\n";
      std::string outFilePath = "/Users/tarindujayatilaka/Documents/LLVM/results/CTMark/" + moduleName + ".txt";

      std::ofstream outFile;
      outFile.open(outFilePath, std::ios_base::app);

      //dbgs() << "outFilePath F:" << outFilePath << "\n";
      ////////////////
      
      std::string banner = formatv("*** Function Pass # {0} # {1} #\n", PassID, F->getName().str());
      raw_string_ostream* outString = new raw_string_ostream(banner);

      FunctionPropertiesInfo FPI = buildFPI(const_cast<Function *>(F));
      FPI.print(*outString);
      
      outFile  << outString->str();

      outString->flush();
      outFile.close();
    }

    else if (any_isa<const LazyCallGraph::SCC *>(IR)) {
      const LazyCallGraph::SCC *C = any_cast<const LazyCallGraph::SCC *>(IR);
      
      //dbgs() << formatv("*** LazyCallGraph Pass: {0} ***\n", PassID);

      for (const LazyCallGraph::Node &N : *C) {
        const Function &F = N.getFunction();
        const Module *M = N.getFunction().getParent();
        
        std::string moduleName = Twine(M->getName()).str();

        std::string delimiter = "CTMark/";
        moduleName.erase(0, moduleName.find(delimiter) + delimiter.length());
        replace(moduleName.begin(), moduleName.end(), '/', '_');
        std::string outFilePath = "/Users/tarindujayatilaka/Documents/LLVM/results/CTMark/" + moduleName + ".txt";

        std::ofstream outFile;
        outFile.open(outFilePath, std::ios_base::app);
        
        std::string banner = formatv("*** LazyCallGraph Pass # {0} # {1} #\n", PassID, F.getName().str());
        raw_string_ostream* outString = new raw_string_ostream(banner);

        FunctionPropertiesInfo FPI = buildFPI(const_cast<Function *>(&F));
        FPI.print(*outString);
        
        outFile  << outString->str();

        outString->flush();
        outFile.close();
      }
    }

    else if (any_isa<const Loop *>(IR)) {
      const Loop *L = any_cast<const Loop *>(IR);
      const Function *F = L->getHeader()->getParent();
      const Module *M = F->getParent();
      
      //dbgs() << formatv("*** Loop Pass: {0} ***\n", PassID);

      std::string moduleName = Twine(M->getName()).str();

      std::string delimiter = "CTMark/";
      moduleName.erase(0, moduleName.find(delimiter) + delimiter.length());
      replace(moduleName.begin(), moduleName.end(), '/', '_');
      std::string outFilePath = "/Users/tarindujayatilaka/Documents/LLVM/results/CTMark/" + moduleName + ".txt";

      std::ofstream outFile;
      outFile.open(outFilePath, std::ios_base::app);
      
      std::string banner = formatv("*** Loop Pass # {0} # {1} #\n", PassID, F->getName().str());
      raw_string_ostream* outString = new raw_string_ostream(banner);

      FunctionPropertiesInfo FPI = buildFPI(const_cast<Function *>(F));
      FPI.print(*outString);
      
      outFile  << outString->str();

      outString->flush();
      outFile.close();
    }
  }

    public:
      void registerCallbacks(PassInstrumentationCallbacks& PIC) {
        using namespace std::placeholders;
        PIC.registerAfterPassCallback(
          std::bind(&FunctionPropertiesAnalysisPassInstrument::afterPass, this, _1, _2));
      }
};
} // end anonymous namespace

static FunctionPropertiesAnalysisPassInstrument FPAPI;

extern "C" ::llvm::PassPluginLibraryInfo LLVM_ATTRIBUTE_WEAK
llvmGetPassPluginInfo() {
  return {
    LLVM_PLUGIN_API_VERSION, "NewPMStopAfterInstrumentPlugin", "v0.1",
    [](PassBuilder& PB) {
      auto& PIC = *PB.getPassInstrumentationCallbacks();
      FPAPI.registerCallbacks(PIC);
    }
  };
}
