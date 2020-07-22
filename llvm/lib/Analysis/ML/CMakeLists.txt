add_subdirectory(FunctionPropertiesAnalysisInstrumentation)

add_llvm_component_library(LLVMMLPolicies
  FunctionPropertiesAnalysis.cpp

  DEPENDS
  intrinsics_gen
  )
