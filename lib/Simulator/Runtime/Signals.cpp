#include "Simulator/Runtime/Signals.h"

#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace mlir::llhd::sim;
//===----------------------------------------------------------------------===//
// Data structures
//===----------------------------------------------------------------------===//

namespace mlir {
namespace llhd {
namespace sim {

struct Time {
public:
  /// Empty (zero) time constructor. All the time values are defaulted to 0.
  Time() = default;

  /// Construct with given time values.
  Time(unsigned time, unsigned delta, unsigned eps)
      : time(time), delta(delta), eps(eps) {}
  /// Get the stored time in a printable format.
  std::string dump() {
    std::stringstream dumpStr;
    dumpStr << time << "ns";
    return dumpStr.str();
  }

  unsigned time;
  unsigned delta;
  unsigned eps;
};

struct Signal {
public:
  /// Construct a signal with the given initial value.
  Signal(int value) : value(value) {}

  int value;
};

struct Slot {
  /// Construct a new slot.
  Slot(Time time) : time(time) {}

  Time time;
};

struct State {
public:
  /// Construct a new empty (at 0 time) state.
  State() = default;

  /// Pop the head of the queue and update the simulation time.
  void popQueue() {
    assert(!queue.empty() && "the event queue is empty");
    Slot pop = queue.front();
    time.time = pop.time.time;
    queue.erase(queue.begin());
  }

  /// Push a new event in the event queue and return the index of the new event
  /// in the queue.
  int pushQueue(int t) {
    Time newTime = Time(time.time + t, 0, 0);
    queue.push_back(newTime);
    return queue.size() - 1;
  }

  /// Get the signal at position i in the signal list.
  Signal getSignal(int i) { return signals.at(i); }

  /// Add a new signal to the state. Returns the index of the new signal.
  int addSignal(Signal sig) {
    signals.push_back(sig);
    return signals.size() - 1;
  }

  /// Update the signal at position i in the signals list to the given value.
  void updateSignal(int index, int value) {
    Signal sig = signals.at(index);
    sig.value = value;
    signals.at(index) = sig;
  }

  Time time;
  std::vector<Signal> signals;
  std::vector<Slot> queue;
};
} // namespace sim
} // namespace llhd
} // namespace mlir

//===----------------------------------------------------------------------===//
// Runtime interfaces
//===----------------------------------------------------------------------===//

void *probe_signal(void *state, int index) {
  State *casted = (State *)state;
  Signal sig = casted->getSignal(index);
  return (void *)&sig.value;
}

void drive_signal(void *state, int index, int value, int time) {
  State *casted = (State *)state;
  // update the signal value
  casted->updateSignal(index, value);
  // spawn new event
  int ind = casted->pushQueue(time);
  // dump trace
  std::cout << casted->queue.at(0).time.dump() << "  " << index << "  " << value
            << "\n";
}

int alloc_signal(void *state, int init) {
  State *casted = (State *)state;
  Signal newSig = Signal(init);
  int newInd = casted->addSignal(newSig);
  /// dump the initial value of the signal
  std::cout << casted->time.dump() << "  " << newInd << "  " << init << "\n";
  // spawn a new event at time 0 to avoid segfaults on first step
  casted->pushQueue(0);
  return newInd;
}

void *init_state() {
  State *state = new State();
  return (void *)state;
}

int queue_empty(void *state) {
  State *casted = (State *)state;
  return casted->queue.empty();
}

void pop_queue(void *state) {
  State *casted = (State *)state;
  casted->popQueue();
}
