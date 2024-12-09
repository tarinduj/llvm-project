set (LIBS
MLIRPresburgerDialect
)

add_llvm_tool(run-presburger
  run-presburger.cpp

  DEPENDS
  ${LIBS}
  )

target_link_libraries(run-presburger PRIVATE ${LIBS} gmp gmpxx)
target_compile_options(run-presburger PRIVATE -march=native)

llvm_update_compile_flags(run-presburger)
mlir_check_all_link_libraries(run-presburger)
