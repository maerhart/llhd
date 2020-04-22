#ifndef LLHD_TRANSFORMS_PASSES_H
#define LLHD_TRANSFORMS_PASSES_H

#include "mlir/IR/Module.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace llhd {

std::unique_ptr<OperationPass<ModuleOp>> createProcessLoweringPass();

} // namespace llhd
} // namespace mlir

#endif // LLHD_TRANSFORMS_PASSES_H
