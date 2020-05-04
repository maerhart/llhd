#include <stdio.h>

typedef struct State State;

State *init_state();
int alloc_signal(State *, char *name, char *owner, int init);
int *probe_signal(State *state, int index);
void drive_signal(State *state, int index, int value, int time);
void dump_changes(State *state);
// int *probe_signal(State *state, int index, int init);

// @Foo
void foo(State *state) {
  // %0 = llhd.const 0 : i1
  // %1 = llhd.sig %0 : i1 -> !llhd.sig<i1>
  int index = alloc_signal(state, "toggle", "foo", 0);
  // %2 = llhd.prb %1 : !llhd.sig<i1> -> i1
  int *prb = probe_signal(state, index);
  int load = *prb;
  // %3 = llhd.not %2 : i1
  load = !load;
  // llhd.drv %1, %3, 1ns : !llhd.sig<i1>, i1, !llhd.time
  drive_signal(state, 0, load, 1);
}

void simulate() {
  State *state = init_state();
  // run the simulation until no more events are queued, limited to 100 steps
  int i = 0;
  while (!queue_empty(state) && i < 100) {
    pop_queue(state);
    foo(state);
    ++i;
  }
}

int main() {
  simulate();
  return 0;
}
