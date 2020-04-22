get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(translation_libs GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)
set(LIBS
  ${dialect_libs}
  ${translation_libs}
  MLIRParser
  MLIRPass
  MLIRTranslation
  MLIRSupport

  MLIRLLHD
  MLIRLLHDTargetVerilog
)
add_llvm_executable(llhd-translate
  llhd-translate.cpp
)
llvm_update_compile_flags(llhd-translate)
target_link_libraries(llhd-translate PRIVATE MLIRIR MLIRTranslation ${LIBS} LLVMSupport)
