#include "Simulator/signals-runtime-wrappers.h"
#include "Simulator/State.h"

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/raw_ostream.h"

#include <cassert>

using namespace llvm;
using namespace mlir::llhd::sim;

//===----------------------------------------------------------------------===//
// Runtime interface
//===----------------------------------------------------------------------===//

int alloc_signal(State *state, int index, char *owner, uint8_t *value,
                 int64_t size) {
  assert(state && "alloc_signal: state not found");
  std::string sOwner(owner);

  return state->addSignalData(index, sOwner, value, size);
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

SignalDetail *probe_signal(State *state, int index) {
  assert(state && "probe_signal: state not found");
  auto &sig = state->signals[index];
  return &sig.detail;
}

void drive_signal(State *state, int index, uint8_t *value, uint64_t width,
                  int time, int delta, int eps) {
  assert(state && "drive_signal: state not found");

  APInt drive(width, ArrayRef<uint64_t>(reinterpret_cast<uint64_t *>(value),
                                        state->signals[index].size));

  Time sTime(time, delta, eps);

  // track back origin signal
  int originIdx = index;
  while (state->signals[originIdx].origin >= 0) {
    originIdx = state->signals[originIdx].origin;
  }

  int bitOffset = (state->signals[index].detail.value -
                   state->signals[originIdx].detail.value) *
                      8 +
                  state->signals[index].detail.offset;

  // spawn new event
  state->pushQueue(sTime, originIdx, bitOffset, drive);
}

int add_subsignal(mlir::llhd::sim::State *state, int origin, uint8_t *ptr,
                  uint64_t len, uint64_t offset) {
  int size = std::ceil((len + offset) / 8.0);
  state->signals.push_back(Signal(origin, ptr, size, offset));
  return (state->signals.size() - 1);
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
      Signal *curr = &(state->signals[change.first]);
      APInt buff(
          curr->size * 8,
          ArrayRef<uint64_t>(reinterpret_cast<uint64_t *>(curr->detail.value),
                             curr->size));

      for (auto update : change.second) {
        if (update.second.getBitWidth() < buff.getBitWidth())
          buff.insertBits(update.second, update.first);
        else
          buff = update.second;
      }

      // continue if updated signal is equal to the initial
      if (std::memcmp(curr->detail.value, buff.getRawData(), curr->size) == 0)
        continue;

      // update signal value
      std::memcpy(curr->detail.value, buff.getRawData(),
                  state->signals[change.first].size);

      // dump the updated signal
      state->dumpSignal(llvm::outs(), change.first);
    }
  }
}
