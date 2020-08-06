#ifndef MLIR_DIALECT_PRESBURGER_PASSES_H
#define MLIR_DIALECT_PRESBURGER_PASSES_H

namespace mlir {
class FuncOp;
template <typename T>
class OperationPass;

void populatePresburgerEvaluatePatterns(OwningRewritePatternList &patterns,
                                        MLIRContext *ctx);

/// Create a pass to evaluate Presburger operations and therefore eliminate all
/// operations on sets and expressions.
std::unique_ptr<OperationPass<FuncOp>> createPresburgerEvaluatePass();

} // namespace mlir

#endif //  MLIR_DIALECT_PRESBURGER_PASSES_H
