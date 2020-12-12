#include "mlir/Dialect/Presburger/PresburgerOptions.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ManagedStatic.h"

using namespace mlir;
using namespace llvm;

namespace {
struct PresburgerOptions {
  cl::opt<bool> presburgerRuntimes{"print-presburger-runtime", cl::desc("TODO"),
                                   cl::value_desc("TODO"), cl::init(false)};
  cl::opt<bool> presburgerSets{"print-presburger-results", cl::desc("TODO"),
                               cl::value_desc("TODO"), cl::init(false)};
  cl::opt<bool> passManagerRuntime{"print-presburger-pm-runtime",
                                   cl::desc("TODO"), cl::value_desc("TODO"),
                                   cl::init(false)};
};
} // namespace

static llvm::ManagedStatic<PresburgerOptions> options;

void mlir::registerPresburgerCLOptions() {
  // Make sure that the options struct has been constructed.
  *options;
}

bool mlir::printPresburgerRuntimes() { return options->presburgerRuntimes; }
bool mlir::dumpResults() { return options->presburgerSets; }
bool mlir::printPassManagerRuntime() { return options->passManagerRuntime; }
