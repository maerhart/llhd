set(LIBS
  
  MLIREDSC
  MLIRIR
  MLIRStandardOps
  MLIRSupport
  MLIRTransformUtils
  LLVMSupport)

add_llvm_library(MLIRLLHD
  LLHDDialect.cpp
  LLHDOps.cpp
  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/Dialect/LLHD
)

# Make sure that the TableGen generated files are up-to-date
add_dependencies(MLIRLLHD
  MLIRLLHDOpsIncGen
  ${LIBS})
target_link_libraries(MLIRLLHD ${LIBS})

# specify header directories
target_include_directories(MLIRLLHD PUBLIC "${PROJECT_SOURCE_DIR}/include")
target_include_directories(MLIRLLHD PUBLIC "${PROJECT_BINARY_DIR}/include")

