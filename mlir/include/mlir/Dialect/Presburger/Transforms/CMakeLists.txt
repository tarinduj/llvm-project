set(LLVM_TARGET_DEFINITIONS EvaluationPatterns.td )
mlir_tablegen(EvaluationPatterns.cpp.inc -gen-rewriters
    "-I${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Presburger")
add_public_tablegen_target(MLIRPresburgerEvaluationPatterns)

set(LLVM_TARGET_DEFINITIONS CanonicalizationPatterns.td )
mlir_tablegen(CanonicalizationPatterns.cpp.inc -gen-rewriters
    "-I${MLIR_MAIN_INCLUDE_DIR}")
add_public_tablegen_target(MLIRPresburgerCanonicalizationPatterns)
