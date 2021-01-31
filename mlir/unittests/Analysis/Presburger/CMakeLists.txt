add_mlir_unittest(MLIRPresburgerTests
  MatrixTest.cpp
  SimplexTest.cpp
  ParamLexSimplexTest.cpp
  CoalesceTest.cpp
  SetTest.cpp
)

target_link_libraries(MLIRPresburgerTests
  PRIVATE MLIRPresburger
  MLIRPresburgerDialect
  MLIRAnalysis
  MLIRLoopAnalysis
  /usr/lib/llvm-11/lib/clang/11.0.0/lib/linux/libclang_rt.builtins-x86_64.a
)

