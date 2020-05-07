#include "Simulator/Engine.h"
#include "Simulator/State.h"

#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/IR/Module.h"

#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace llhd::sim;

Engine::Engine(llvm::raw_ostream &out, ModuleOp module)
    : module(module), out(out) {
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
  assert(engine && "engine not found");
  assert(state && "state not found");
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
    unsigned actual = 0;
    for (auto change : pop.changes) {
      if (*state->signals[change.first].value == change.second)
        continue;
      state->updateSignal(change.first, change.second);
      state->dumpSignal(out, change.first);
      actual++;
    }
    // continue if no updates at non-zero time
    if (actual == 0 && !state->time.isZero()) {
      continue;
    }

    // run entity
    auto invocationResult = engine->invoke("Foo", state);
    if (invocationResult) {
      llvm::errs() << "Failed invocation of Foo: " << invocationResult;
      return -1;
    }

    // dump signals initial values
    if (state->time.isZero()) {
      for (int i = 0; i < state->signals.size(); i++) {
        state->dumpSignal(out, i);
      }
    }
    i++;
  }
  llvm::errs() << "Finished after " << i << " steps.\n";
  return 0;
}
