#include "Simulator/Runtime/Signals.h"
#include <stdio.h>

// @Foo
void foo(void *state) {
  // %2 = llhd.prb %1 : !llhd.sig<i1> -> i1
  int prb = *(int *)probe_signal(state, 0);
  // %3 = llhd.not %2 : i1
  prb = !prb;
  // llhd.drv %1, %3, 1ns : !llhd.sig<i1>, i1, !llhd.time
  drive_signal(state, 0, prb, 1);
}

void simulate() {
  void *state = init_state();
  // %0 = llhd.const 0 : i1
  // %1 = llhd.sig %0 : i1 -> !llhd.sig<i1>
  alloc_signal(state, 0);
  // run the simulation until no more events are queued, limited to 100 steps
  int i = 0;
  do {
    pop_queue(state);
    foo(state);
    ++i;
  } while (!queue_empty(state) && i < 100);
}

int main() {
  simulate();
  return 0;
}
