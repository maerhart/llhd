#ifndef LLHD_SIMULATOR_RUNTIME_SIGNALS_H
#define LLHD_SIMULATOR_RUNTIME_SIGNALS_H

//===----------------------------------------------------------------------===//
// Data structures
//===----------------------------------------------------------------------===//

#ifdef __cplusplus
namespace mlir {
namespace llhd {
namespace sim {

/// The simulator's internal representation of time.
struct Time;

/// The simulator's internal representation of a signal.
struct Signal;

/// The simulator's internal representation of one queue slot.
struct Slot;

/// The simulator's state. It contains the current simulation time, signal
/// values and the event queue.
struct State;

#else

// Expose the state type to C interface.
typedef struct State State;

#endif

//===----------------------------------------------------------------------===//
// Runtime interfaces
//===----------------------------------------------------------------------===//

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

/// Probe a signal and return a pointer to the carried value.
int *probe_signal(State *state, int index);

/// Drive a value onto a signal.
void drive_signal(State *state, int index, int value, int time);

/// Allocate a new signal. The index of the new signal in the state's list of
/// signals is returned.
int alloc_signal(State *state, int init);

/// Initialize an empty state and return a pointer to it.
State *init_state();

/// Return whether the queue is empty.
int queue_empty(State *state);

/// Pop the head of the queue.
void pop_queue(State *state);

#ifdef __cplusplus
}
} // namespace sim
} // namespace llhd
} // namespace mlir
#endif // __cplusplus

#endif // LLHD_SIMULATOR_RUNTIME_SIGNALS_H
