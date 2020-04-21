set(LLVM_OPTIONAL_SOURCES
    State.cpp
    Engine.cpp
    signals-runtime-wrappers.cpp
)

add_mlir_library(LLHDSimState State.cpp)
target_link_libraries(LLHDSimState LLVMSupport)

add_mlir_library(llhd-signals-runtime-wrappers  SHARED
    signals-runtime-wrappers.cpp 
)
target_link_libraries(llhd-signals-runtime-wrappers LLHDSimState)

add_mlir_library(LLHDSimEngine Engine.cpp)

set(LIBS
    MLIRExecutionEngine
    LLHDSimState
    llhd-signals-runtime-wrappers
)

target_link_libraries(LLHDSimEngine ${LIBS})
