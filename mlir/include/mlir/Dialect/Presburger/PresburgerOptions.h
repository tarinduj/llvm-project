#ifndef MLIR_PRESBURGER_OPTIONS_H
#define MLIR_PRESBURGER_OPTIONS_H

namespace mlir {
void registerPresburgerCLOptions();
bool printPresburgerRuntimes();
bool dumpResults();
bool printPassManagerRuntime();
} // namespace mlir

#endif // MLIR_PRESBURGER_OPTIONS_H
