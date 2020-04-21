#include "Simulator/State.h"

#include "llvm/Support/raw_ostream.h"

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
  return (this->time == rhs.time && this->delta == rhs.delta &&
          this->eps == rhs.eps);
}

Time Time::operator+(const Time &rhs) const {
  return Time(time + rhs.time, delta + rhs.delta, eps + rhs.eps);
}

std::string Time::dump() {
  std::stringstream dumpStr;
  dumpStr << time << "ns";
  return dumpStr.str();
}

//===----------------------------------------------------------------------===//
// Slot
//===----------------------------------------------------------------------===//

bool Slot::operator<(const Slot &rhs) const { return time < rhs.time; }

bool Slot::operator>(const Slot &rhs) const { return rhs.time < time; }

void Slot::insertChange(int index, int value) { changes[index] = value; }

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
void State::pushQueue(int t, int index, int value) {
  Time newTime = time + Time(t, 0, 0);
  queue.insertOrUpdate(newTime, index, value);
}

/// Add a new signal to the state. Returns the index of the new signal.
int State::addSignal(Signal sig) {
  signals.push_back(sig);
  return signals.size() - 1;
}

/// Update the signal at position i in the signals list to the given value.
void State::updateSignal(int index, int value) { signals[index].value = value; }

//===----------------------------------------------------------------------===//
// UpdateQueue
//===----------------------------------------------------------------------===//
void UpdateQueue::insertOrUpdate(Time time, int index, int value) {
  for (int i = 0; i < c.size(); i++) {
    if (time == c[i].time) {
      c[i].insertChange(index, value);
      return;
    }
  }
  Slot newSlot(time);
  newSlot.insertChange(index, value);
  push(newSlot);
}

void State::dumpSignal(int index) {
  llvm::errs() << time.dump() << "  " << index << "  " << signals[index].value
               << "\n";
}
