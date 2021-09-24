set (LIBS
MLIRPresburgerDialect
)

add_llvm_tool(run-presburger-mallocs
  run-presburger-mallocs.cpp
  malloc_count.c

  DEPENDS
  ${LIBS}
  )

target_link_libraries(run-presburger-mallocs PRIVATE ${LIBS} /usr/lib/llvm-11/lib/clang/11.0.0/lib/linux/libclang_rt.builtins-x86_64.a gmp gmpxx)
target_compile_options(run-presburger-mallocs PRIVATE -march=native -DPRINT_PIVOTS)

llvm_update_compile_flags(run-presburger-mallocs)
mlir_check_all_link_libraries(run-presburger-mallocs)
