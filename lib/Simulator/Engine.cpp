#include "Simulator/Engine.h"
#include "Simulator/State.h"

#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/IR/Module.h"

#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace llhd::sim;

Engine::Engine(ModuleOp module) : module(module) {
  state = std::make_unique<State>();
  // add 0-time event
  state->queue.push(Slot(Time()));

  // init jit
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  auto maybeEngine = mlir::ExecutionEngine::create(module);
  assert(maybeEngine && "failed to create JIT");
  engine = std::move(*maybeEngine);
}

int Engine::simulate(int n) {
  int i = 0;
  while (!state->queue.empty()) {
    if (n > 0 && i >= n) {
      break;
    }
    // pop queue
    auto pop = state->popQueue();

    // update time
    assert(state->time < pop.time || pop.time.time == 0);
    state->time = pop.time;

    // dump changes, only if actually changed
    for (auto change : pop.changes) {
      if (state->signals[change.first].value == change.second)
        continue;
      state->signals[change.first] = change.second;
      state->dumpSignal(change.first);
    }

    // run entity
    auto invocationResult = engine->invoke("Foo", state);
    if (invocationResult) {
      llvm::errs() << "Failed invocation of Foo: " << invocationResult;
      return -1;
    }
    i++;
  }
  llvm::errs() << "Finished after " << i << " steps.\n";
  return 0;
}
