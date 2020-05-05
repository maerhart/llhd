#ifndef LLHD_SIMULATOR_STATE_H
#define LLHD_SIMULATOR_STATE_H

#include <map>
#include <queue>
#include <string>
#include <vector>

#include "llvm/ADT/PriorityQueue.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {
namespace llhd {
namespace sim {

/// The simulator's internal representation of time.
struct Time {
  /// Empty (zero) time constructor. All the time values are defaulted to 0.
  Time() = default;

  /// Construct with given time values.
  Time(unsigned time, unsigned delta, unsigned eps)
      : time(time), delta(delta), eps(eps) {}

  /// Compare the time values in order of time, delta, eps.
  bool operator<(const Time &rhs) const;

  /// Return true if all the time values are equal.
  bool operator==(const Time &rhs) const;

  /// Add two time values.
  Time operator+(const Time &rhs) const;

  /// Return true if the time represents zero-time.
  bool isZero();

  /// Get the stored time in a printable format.
  std::string dump();

  unsigned time;
  unsigned delta;
  unsigned eps;

private:
};

/// The simulator's internal representation of a signal.
struct Signal {
  /// Construct a signal with the given name, owner and initial value.
  Signal(std::string name, std::string owner, int init);

  /// Return the signal value in dumpable format: "owner/name  value"
  std::string dump();

  std::string name;
  std::string owner;
  std::unique_ptr<int> value;
};

/// The simulator's internal representation of one queue slot.
struct Slot {
  /// Construct a new slot.
  Slot(Time time) : time(time) {}

  /// Returns true if the slot's time is smaller than the compared slot's time.
  bool operator<(const Slot &rhs) const;

  /// Returns true if the slot's time is greater than the compared slot's time.
  bool operator>(const Slot &rhs) const;

  /// Insert a change.
  void insertChange(int index, int value);

  // <signal-index, new-value>
  std::map<int, int> changes;
  Time time;
};

/// This is equivalent to and std::priorityQueue<Slot> ordered using the greater
/// operator, which adds an insertion method to add changes to a slot.
class UpdateQueue
    : public std::priority_queue<Slot, std::vector<Slot>, std::greater<Slot>> {
public:
  /// Check wheter a slot for the given time already exists. If that's the case,
  /// add the new change to it, else create a new slot and push it to the queue.
  void insertOrUpdate(Time time, int index, int value);
};

/// The simulator's state. It contains the current simulation time, signal
/// values and the event queue.
struct State {
  /// Construct a new empty (at 0 time) state.
  State() = default;

  /// Pop the head of the queue and update the simulation time.
  Slot popQueue();

  /// Push a new event in the event queue and return the index of the new event
  /// in the queue.
  void pushQueue(Time time, int index, int value);

  /// Get the signal at position i in the signal list.
  Signal getSignal(int index);

  /// Add a new signal to the state. Returns the index of the new signal.
  int addSignal(std::string name, std::string owner, int init);

  /// Update the signal at position i in the signals list to the given value.
  void updateSignal(int index, int value);

  /// Dump a signal to llvm::errs().
  void dumpSignal(llvm::raw_ostream &out, int index);

  Time time;
  std::vector<Signal> signals;
  UpdateQueue queue;
};

} // namespace sim
} // namespace llhd
} // namespace mlir

#endif // LLHD_SIMULATOR_STATE_H
