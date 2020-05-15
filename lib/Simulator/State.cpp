#include "Simulator/State.h"

#include "llvm/Support/raw_ostream.h"

#include <iomanip>
#include <map>
#include <sstream>
#include <string>

using namespace mlir;
using namespace llhd::sim;

//===----------------------------------------------------------------------===//
// Time
//===----------------------------------------------------------------------===//

bool Time::operator<(const Time &rhs) const {
  if (time < rhs.time)
    return true;
  if (time == rhs.time && delta < rhs.delta)
    return true;
  if (time == rhs.time && delta == rhs.delta && eps < rhs.eps)
    return true;
  return false;
}

bool Time::operator==(const Time &rhs) const {
  return (time == rhs.time && delta == rhs.delta && eps == rhs.eps);
}

Time Time::operator+(const Time &rhs) const {
  return Time(time + rhs.time, delta + rhs.delta, eps + rhs.eps);
}

bool Time::isZero() { return (time == 0 && delta == 0 && eps == 0); }

std::string Time::dump() {
  std::stringstream dumpStr;
  dumpStr << time << "ns " << delta << "d " << eps << "e";
  return dumpStr.str();
}

//===----------------------------------------------------------------------===//
// Signal
//===----------------------------------------------------------------------===//

Signal::Signal(std::string name, std::string owner, uint8_t *value,
               uint64_t size)
    : name(name), owner(owner), size(size), value(value) {}

bool Signal::operator==(const Signal &rhs) const {
  if (owner != rhs.owner || name != rhs.name || size != rhs.size)
    return false;
  for (int i = 0; i < size; i++) {
    if (value[i] != rhs.value[i])
      return false;
  }
  return true;
}

bool Signal::operator<(const Signal &rhs) const {
  if (owner < rhs.owner)
    return true;
  if (owner == rhs.owner && name < rhs.name)
    return true;
  return false;
}

std::string Signal::dump() {
  std::stringstream ss;
  ss << owner << "/" << name << "  "
     << "0x";
  for (int i = size - 1; i >= 0; i--) {
    ss << std::setw(2) << std::setfill('0') << std::hex
       << static_cast<int>(value[i]);
  }
  return ss.str();
}

//===----------------------------------------------------------------------===//
// Slot
//===----------------------------------------------------------------------===//

bool Slot::operator<(const Slot &rhs) const { return time < rhs.time; }

bool Slot::operator>(const Slot &rhs) const { return rhs.time < time; }

void Slot::insertChange(int index, std::vector<uint8_t> &bytes) {
  changes[index] = bytes;
}

//===----------------------------------------------------------------------===//
// UpdateQueue
//===----------------------------------------------------------------------===//
void UpdateQueue::insertOrUpdate(Time time, int index,
                                 std::vector<uint8_t> &bytes) {
  for (int i = 0; i < c.size(); i++) {
    if (time == c[i].time) {
      c[i].insertChange(index, bytes);
      return;
    }
  }
  Slot newSlot(time);
  newSlot.insertChange(index, bytes);
  push(newSlot);
}

//===----------------------------------------------------------------------===//
// State
//===----------------------------------------------------------------------===//

Slot State::popQueue() {
  assert(!queue.empty() && "the event queue is empty");
  Slot pop = queue.top();
  queue.pop();
  return pop;
}

/// Push a new event in the event queue and return the index of the new event
/// in the queue.
void State::pushQueue(Time t, int index, std::vector<uint8_t> &bytes) {
  Time newTime = time + t;
  queue.insertOrUpdate(newTime, index, bytes);
}

/// Add a new signal to the state. Returns the index of the new signal.
int State::addSignal(std::string name, std::string owner, uint8_t *value,
                     uint64_t size) {
  signals.push_back(Signal(name, owner, value, size));
  return signals.size() - 1;
}

/// Update the signal at position i in the signals list to the given value.
void State::updateSignal(int index, std::vector<uint8_t> &bytes) {
  assert(signals[index].size == bytes.size() && "size mismatch");
  for (int i = 0; i < signals[index].size; i++) {
    signals[index].value[i] = bytes[i];
  }
}

void State::dumpSignal(llvm::raw_ostream &out, int index) {
  out << time.dump() << "  " << signals[index].dump() << "\n";
}
