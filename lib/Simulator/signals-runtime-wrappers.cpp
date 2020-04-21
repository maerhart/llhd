#include "Simulator/signals-runtime-wrappers.h"
#include "Simulator/State.h"

#include <cassert>
#include <iostream>

using namespace mlir::llhd::sim;

//===----------------------------------------------------------------------===//
// Runtime interface
//===----------------------------------------------------------------------===//

int *probe_signal(State *state, int index) {
  Signal sig = state->signals[index];
  return &sig.value;
}

void drive_signal(State *state, int index, int value, int time) {
  // spawn new event
  state->pushQueue(time, index, value);
}

int alloc_signal(State *state, int index, int init) {
  if (index >= state->signals.size()) {
    Signal newSig = Signal(init);
    int newInd = state->addSignal(newSig);
    assert(index == newInd &&
           "the new signal index is expected to be the next index pushed");
    /// dump the initial value of the signal
    std::cout << state->time.dump() << "  " << newInd << "  " << init << "\n";
  }
  return index;
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
  if (!pop.changes.empty())
    state->signals[0] = pop.changes[0];
}

void dump_changes(State *state) {
  Slot slot = state->queue.top();
  if (!slot.changes.empty())
    for (auto change : slot.changes)
      std::cout << slot.time.dump() << "  " << change.first << "  "
                << change.second << "\n";
}
