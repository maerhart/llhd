add_llvm_executable(simplec simple.c)
target_link_libraries(simplec PRIVATE llhd-signals-runtime-wrappers)
