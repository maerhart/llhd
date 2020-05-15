#include "Simulator/signals-runtime-wrappers.h"
#include "Simulator/State.h"
#include "llvm/Support/raw_ostream.h"

#include <cassert>

using namespace mlir::llhd::sim;

//===----------------------------------------------------------------------===//
// Runtime interface
//===----------------------------------------------------------------------===//

int alloc_signal(State *state, char *name, char *owner, uint8_t *value,
                 int64_t size) {
  assert(state && "alloc_signal: state not found");
  std::string sName(name), sOwner(owner);
  // if a signal with the same name and owner has already been allocated, return
  // the index of the signal
  for (int i = 0; i < state->signals.size(); i++) {
    if (state->signals[i].name == sName && state->signals[i].owner == sOwner)
      return i;
  }
  return state->addSignal(sName, sOwner, value, size);
}

int gather_signal(State *state, char *name, char *owner) {
  assert(state && "gather_signal: state not found");
  std::string sName(name), sOwner(owner);
  for (int i = 0; i < state->signals.size(); i++) {
    if (state->signals[i].name == sName && state->signals[i].owner == sOwner)
      return i;
  }

  llvm::errs() << "signal " << sOwner << "/" << sOwner << " not found"
               << "\n";
  exit(EXIT_FAILURE);
}

uint8_t *probe_signal(State *state, int index) {
  assert(state && "probe_signal: state not found");
  auto &sig = state->signals[index];
  return sig.value;
}

void drive_signal(State *state, int index, uint8_t *value, uint64_t size,
                  int time, int delta, int eps) {
  assert(state && "drive_signal: state not found");

  // move bytes to a vector (the alloca'd value pointer get's freed once the jit
  // function returns)
  std::vector<uint8_t> bytes;
  for (int i = 0; i < size; i++) {
    bytes.push_back(value[i]);
  }

  // create Time struct
  Time sTime(time, delta, eps);
  // spawn new event
  state->pushQueue(sTime, index, bytes);
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
      assert(state->signals[change.first].size == change.second.size() &&
             "size mismatch");
      bool equal = true;
      for (int i = 0; i < change.second.size(); i++) {
        equal &= (state->signals[change.first].value[i] == change.second[i]);
      }
      if (equal)
        continue;

      state->updateSignal(change.first, change.second);
      state->dumpSignal(llvm::outs(), change.first);
    }
  }
}
