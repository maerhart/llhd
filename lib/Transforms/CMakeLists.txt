set(LIBS

  MLIREDSC
  MLIRIR
  MLIRStandardOps
  MLIRLLHD
  MLIRTransformUtils
  LLVMSupport)

add_llvm_library(MLIRLLHDTransforms
  ProcessLoweringPass.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/Transforms
)
target_link_libraries(MLIRLLHDTransforms ${LIBS})

# specify header directories
target_include_directories(MLIRLLHDTransforms PUBLIC "${PROJECT_SOURCE_DIR}/include")
target_include_directories(MLIRLLHDTransforms PUBLIC "${PROJECT_BINARY_DIR}/include")
