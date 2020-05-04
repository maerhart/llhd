#include "Simulator/signals-runtime-wrappers.h"
#include "Simulator/State.h"
#include "llvm/Support/raw_ostream.h"

#include <cassert>

using namespace mlir::llhd::sim;

//===----------------------------------------------------------------------===//
// Runtime interface
//===----------------------------------------------------------------------===//

int alloc_signal(State *state, char *name, char *owner, int init) {
  assert(state && "alloc_signal: state not found");
  std::string sName(name), sOwner(owner);
  // if a signal with the same name and owner has already been allocated, return
  // the index of the signal
  for (int i = 0; i < state->signals.size(); i++) {
    if (state->signals[i].name == sName && state->signals[i].owner == sOwner)
      return i;
  }
  return state->addSignal(sName, sOwner, init);
}

int *probe_signal(State *state, int index) {
  assert(state && "probe_signal: state not found");
  auto &sig = state->signals[index];
  return sig.value.get();
}

void drive_signal(State *state, int index, int value, int time) {
  assert(state && "drive_signal: state not found");
  // spawn new event
  state->pushQueue(time, index, value);
}

//===----------------------------------------------------------------------===//
// Testing functions
//===----------------------------------------------------------------------===//

State *init_state() {
  auto state = new State();
  auto zeroSlot = Slot(Time());
  state->queue.push(zeroSlot);
  return state;
}

int queue_empty(State *state) { return state->queue.empty(); }

void pop_queue(State *state) {
  auto pop = state->popQueue();
  state->time.time = pop.time.time;
  if (!pop.changes.empty()) {
    for (auto change : pop.changes) {
      if (*state->signals[change.first].value != change.second) {
        state->updateSignal(change.first, change.second);
        state->dumpSignal(llvm::outs(), change.first);
      }
    }
  }
}
