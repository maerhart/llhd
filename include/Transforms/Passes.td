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

def FunctionElimination : Pass<"llhd-function-elimination", "ModuleOp"> {
    let summary = "Deletes all functions.";
    let description = [{
        Deletes all functions in the module. In case there is still a function
        call in an entity or process, it fails.
        This pass is intended as a post-inlining pass to check if all functions
        could be successfully inlined and remove the inlined functions. This
        is necessary because Structural LLHD does not allow functions. Fails in
        the case that there is still a function call left in a `llhd.proc` or
        `llhd.entity`.
    }];

    let constructor = "mlir::llhd::createFunctionEliminationPass()";
}

#endif // MLIR_LLHD_TRANSFORMS_PASSES
