add_mlir_dialect(PresburgerOps presburger PresburgerOps)

add_subdirectory(Transforms)

set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls)
add_public_tablegen_target(MLIRPresburgerPassIncGen)

# TODO upgrade to newer mlir version first
#add_mlir_doc(PresburgerDialect -gen-dialect-doc PresburgerDialect Presburger/)
#add_mlir_doc(PresburgerOps -gen-op-doc PresburgerOps Presburger/)
#add_mlir_doc(Passes -gen-pass-doc PresburgerPasses ./)

