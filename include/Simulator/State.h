#ifndef LLHD_SIMULATOR_STATE_H
#define LLHD_SIMULATOR_STATE_H

#include <map>
#include <queue>
#include <string>
#include <vector>

#include "llvm/ADT/PriorityQueue.h"
#include "llvm/ADT/StringMap.h"
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
  /// Construct an "empty" signal.
  Signal(std::string name, std::string owner);

  /// Construct a signal with the given name, owner and initial value.
  Signal(std::string name, std::string owner, uint8_t *value, uint64_t size);

  /// Default signal destructor
  ~Signal() = default;

  /// Returns true if the signals match in name, owner, size and value
  bool operator==(const Signal &rhs) const;

  /// Returns true if the owner is lexically smaller than rhs, or the
  /// name is lexically smaller than rhs, in case they share the same owner
  bool operator<(const Signal &rhs) const;

  /// Return the signal value in dumpable format: "0x<value>"
  std::string dump();

  std::string name;
  std::string owner;
  // the list of instances this signal triggers
  std::vector<std::string> triggers;
  // the list of instances this signal is an output of
  std::vector<std::string> outOf;
  uint64_t size;
  uint8_t *value;
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
  void insertChange(int index, std::vector<uint8_t> &bytes);

  // <signal-index, new-value>
  std::map<int, std::vector<uint8_t>> changes;
  Time time;
};

/// This is equivalent to and std::priorityQueue<Slot> ordered using the greater
/// operator, which adds an insertion method to add changes to a slot.
class UpdateQueue
    : public std::priority_queue<Slot, std::vector<Slot>, std::greater<Slot>> {
public:
  /// Check wheter a slot for the given time already exists. If that's the case,
  /// add the new change to it, else create a new slot and push it to the queue.
  void insertOrUpdate(Time time, int index, std::vector<uint8_t> &bytes);
};

/// The simulator internal representation of an instance
struct Instance {
  Instance() = default;

  Instance(std::string name, std::string parent) : name(name), parent(parent) {}

  // the instance name
  std::string name;
  // the instance parent's name
  std::string parent;
  // the instance's base unit
  std::string unit;
  // the signals the unit defines
  std::vector<int> signalTable;
  // the input list
  std::vector<int> sensitivityList;
  // the output list
  std::vector<int> outputs;
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
  void pushQueue(Time time, int index, std::vector<uint8_t> &bytes);

  /// Get the signal at position i in the signal list.
  Signal getSignal(int index);

  /// Add a new signal to the state. Returns the index of the new signal.
  int addSignal(std::string name, std::string owner);

  int addSignalData(int index, std::string owner, uint8_t *value,
                    uint64_t size);

  /// Update the signal at position i in the signals list to the given value.
  void updateSignal(int index, std::vector<uint8_t> &bytes);

  /// Dump a signal to the out stream. One entry is added for every instance the
  /// signal appears in.
  void dumpSignal(llvm::raw_ostream &out, int index);

  /// Dump the instance layout. Used for testing purposes.
  void dumpLayout();

  /// Dump the instances each signal triggers. Used for testing purposes.
  void dumpSignalTriggers();

  Time time;
  std::string root;
  llvm::StringMap<Instance> instances;
  std::vector<Signal> signals;
  UpdateQueue queue;
};

} // namespace sim
} // namespace llhd
} // namespace mlir

#endif // LLHD_SIMULATOR_STATE_H
