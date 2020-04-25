#ifndef LLHD_TRANSFORMS_PASSES_H
#define LLHD_TRANSFORMS_PASSES_H

#include "mlir/IR/Module.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace llhd {

#define GEN_PASS_CLASSES
#include "Transforms/Passes.h.inc"

std::unique_ptr<OperationPass<ModuleOp>> createProcessLoweringPass();

/// Register the LLHD Transformation passes.
inline void initLLHDTransformationPasses() {
#define GEN_PASS_REGISTRATION
#include "Transforms/Passes.h.inc"
}

} // namespace llhd
} // namespace mlir

#endif // LLHD_TRANSFORMS_PASSES_H
