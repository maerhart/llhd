#ifndef MLIR_LLHD_TRANSFORMS_PASSES
#define MLIR_LLHD_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

def ProcessLowering : Pass<"llhd-process-lowering", "ModuleOp"> {
    let summary = "Lowers LLHD Processes to Entities.";
    let description = [{
        TODO
    }];

    let constructor = "mlir::llhd::createProcessLoweringPass()";
}

#endif // MLIR_LLHD_TRANSFORMS_PASSES
