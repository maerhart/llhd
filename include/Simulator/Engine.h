#ifndef LLHD_SIMULATOR_ENGINE_H
#define LLHD_SIMULATOR_ENGINE_H

#include "Simulator/State.h"

#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/IR/Module.h"

namespace mlir {
namespace llhd {
namespace sim {

class Engine {
public:
  /// Initialize an LLHD simulation engine. This initializes the state, as well
  /// as the mlir::ExecutionEngine with the given module.
  Engine(llvm::raw_ostream &out, ModuleOp module);

  /// Run simulation up to n steps. Pass n=0 to run indefinitely.
  int simulate(int n);

private:
  llvm::raw_ostream &out;
  std::unique_ptr<State> state;
  std::unique_ptr<ExecutionEngine> engine;
  ModuleOp module;
};

} // namespace sim
} // namespace llhd
} // namespace mlir

#endif // LLHD_SIMULATOR_ENGINE_H
