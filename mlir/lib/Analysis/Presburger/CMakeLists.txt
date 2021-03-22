add_mlir_library(MLIRPresburger
  Expr.cpp
  )

target_compile_options(obj.MLIRPresburger PRIVATE -march=native)
