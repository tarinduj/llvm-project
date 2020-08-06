add_mlir_dialect_library(MLIRPresburgerDialect
  PresburgerDialect.cpp
  PresburgerOps.cpp
  Attributes.cpp
  Types.cpp
  Parser.cpp
  Transforms/Evaluation.cpp
  
  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/include

  DEPENDS
  MLIRPresburgerOpsIncGen
  MLIRPresburgerPassIncGen
  MLIRPresburgerEvaluationPatterns
)

target_link_libraries(MLIRPresburgerDialect PUBLIC MLIRIR)

target_include_directories(MLIRPresburgerDialect PUBLIC "${MLIR_MAIN_INCLUDE_DIR}")
target_include_directories(MLIRPresburgerDialect PUBLIC "${MLIR_MAIN_INCLUDE_DIR}")


