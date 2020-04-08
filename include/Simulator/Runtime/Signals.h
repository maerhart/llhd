#ifndef LLHD_SIM_SIGNALS_H
#define LLHD_SIM_SIGNALS_H

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

} // namespace sim
} // namespace llhd
} // namespace mlir

#endif

//===----------------------------------------------------------------------===//
// Runtime interfaces
//===----------------------------------------------------------------------===//

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

/// Probe a signal and return it's value as a generic pointer. Type information
/// has to be inferred prior to use to determine what to bitcast the returned
/// pointer to.
void *probe_signal(void *state, int index);

/// Drive a value onto a signal.
void drive_signal(void *state, int index, int value, int time);

/// Allocate a new signal. The index of the new signal in the state's list of
/// signals is returned.
int alloc_signal(void *state, int init);

/// Initialize an empty state and return it as a generic pointer.
void *init_state();

/// Return whether the queue is empty.
int queue_empty(void *state);

/// Pop the head of the queue.
void pop_queue(void *state);

#ifdef __cplusplus
}
#endif // __cplusplus

#endif // LLHD_SIM_SIGNALS_H
