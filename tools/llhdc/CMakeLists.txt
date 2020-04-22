get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        MLIROptLib
        MLIRLLHD
        MLIRLLHDTransforms
        MLIRLLHDToLLVM
        )
add_llvm_executable(llhdc llhdc.cpp)

llvm_update_compile_flags(llhdc)
target_link_libraries(llhdc PRIVATE ${LIBS})
