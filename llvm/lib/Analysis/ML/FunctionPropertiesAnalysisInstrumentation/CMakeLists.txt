set(CMAKE_CXX_STANDARD 14)

list(APPEND CMAKE_MODULE_PATH ${LLVM_DIR})
include(AddLLVM)

add_definitions(-fno-rtti -fno-exceptions)
add_definitions(${LLVM_DEFINITIONS})
include_directories(${LLVM_INCLUDE_DIRS})
link_directories(${LLVM_LIBRARY_DIRS})

set(_SRC_FILES
    FunctionPropertiesAnalysisPassInstrument.cpp)
add_llvm_pass_plugin(FunctionPropertiesAnalysisPassInstrument
  ${_SRC_FILES})
