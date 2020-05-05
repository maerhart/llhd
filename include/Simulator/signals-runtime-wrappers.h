#ifndef LLHD_SIMULATOR_SIGNALS_RUNTIME_WRAPPERS_H
#define LLHD_SIMULATOR_SIGNALS_RUNTIME_WRAPPERS_H

#include "Simulator/State.h"

extern "C" {

//===----------------------------------------------------------------------===//
// Runtime interfaces
//===----------------------------------------------------------------------===//

/// Allocate a new signal. The index of the new signal in the state's list of
/// signals is returned.
int alloc_signal(mlir::llhd::sim::State *state, char *name, char *owner,
                 int init);

/// Probe a signal and return a pointer to the carried value.
int *probe_signal(mlir::llhd::sim::State *state, int index);

/// Drive a value onto a signal.
void drive_signal(mlir::llhd::sim::State *state, int index, int value, int time,
                  int delta, int eps);

//===----------------------------------------------------------------------===//
// Testing methods
//===----------------------------------------------------------------------===//

/// Initialize an empty state and return a pointer to it.
mlir::llhd::sim::State *init_state();

/// Return whether the queue is empty.
int queue_empty(mlir::llhd::sim::State *state);

/// Pop the head of the queue.
void pop_queue(mlir::llhd::sim::State *state);

/// Dump the list of changes in the queue head.
void dump_changes(mlir::llhd::sim::State *state);
}

#endif // LLHD_SIMULATOR_SIGNALS_RUNTIME_WRAPPERS_H
