set(LLVM_OPTIONAL_SOURCES
    State.cpp
    Engine.cpp
    signals-runtime-wrappers.cpp
)

add_mlir_library(LLHDSimState
    State.cpp

    LINK_LIBS PUBLIC
    LLVMSupport
    )

add_mlir_library(llhd-signals-runtime-wrappers SHARED
    signals-runtime-wrappers.cpp

    LINK_LIBS PUBLIC
    LLHDSimState
)

add_mlir_library(LLHDSimEngine
    Engine.cpp

    LINK_LIBS PUBLIC
    MLIRLLHD
    MLIRLLHDToLLVM
    LLHDSimState
    llhd-signals-runtime-wrappers
    MLIRExecutionEngine
    )
