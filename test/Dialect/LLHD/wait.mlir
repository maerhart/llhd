// RUN: llhdc %s | llhdc | FileCheck %s

// Test Overview:
//   * 0 observed signals, no time, successor without arguments
//   * 0 observed signals, with time, sucessor with arguments
//   * 2 observed signals, no time, successor with arguments
//   * 2 observed signals, with time, successor with arguments

// CHECK-LABEL: @check_wait_0
llhd.proc @check_wait_0 () -> () {
    // CHECK: llhd.wait ^[[BB:.*]]
    "llhd.wait"() [^bb1] {operand_segment_sizes=dense<[0,0,0]> : vector<3xi32>} : () -> ()
    // CHECK-NEXT: ^[[BB:.*]]
^bb1:
    // CHECK-NEXT: llhd.wait ^[[BB:.*]]
    llhd.wait ^bb1
}

// CHECK-LABEL: @check_wait_1
llhd.proc @check_wait_1 () -> () {
    // CHECK-NEXT: llhd.const
    %time = llhd.const #llhd.time<0ns, 0d, 0e> : !llhd.time
    // CHECK-NEXT: llhd.wait for %[[TIME:.*]], ^[[BB:.*]](%[[TIME:.*]] : !llhd.time) : !llhd.time
    "llhd.wait"(%time, %time) [^bb1] {operand_segment_sizes=dense<[0,1,1]> : vector<3xi32>} : (!llhd.time, !llhd.time) -> ()
    // CHECK-NEXT: ^[[BB:.*]]
^bb1(%t : !llhd.time):
    // CHECK-NEXT: llhd.wait for %[[T:.*]], ^[[BB:.*]](%[[T:.*]] : !llhd.time) : !llhd.time
    llhd.wait for %t, ^bb1(%t : !llhd.time) : !llhd.time
}

// CHECK-LABEL: @check_wait_2
llhd.proc @check_wait_2 (%arg0 : !llhd.sig<i64>, %arg1 : !llhd.sig<i1>) -> () {
    // CHECK-NEXT: llhd.wait %[[ARG0:.*]], %[[ARG1:.*]], ^[[BB:.*]](%[[ARG1:.*]] : !llhd.sig<i1>) : !llhd.sig<i64>, !llhd.sig<i1>
    "llhd.wait"(%arg0, %arg1, %arg1) [^bb1] {operand_segment_sizes=dense<[2,0,1]> : vector<3xi32>} : (!llhd.sig<i64>, !llhd.sig<i1>, !llhd.sig<i1>) -> ()
    // CHECK: ^[[BB:.*]]
^bb1(%a : !llhd.sig<i1>):
    // CHECK-NEXT: llhd.wait %[[ARG0:.*]], %[[A:.*]], ^[[BB:.*]](%[[A:.*]] : !llhd.sig<i1>) : !llhd.sig<i64>, !llhd.sig<i1>
    llhd.wait %arg0, %a, ^bb1(%a : !llhd.sig<i1>) : !llhd.sig<i64>, !llhd.sig<i1>
}

// CHECK-LABEL: @check_wait_3
llhd.proc @check_wait_3 (%arg0 : !llhd.sig<i64>, %arg1 : !llhd.sig<i1>) -> () {
    // CHECK-NEXT: llhd.const
    %time = llhd.const #llhd.time<0ns, 0d, 0e> : !llhd.time
    // CHECK-NEXT: llhd.wait %[[ARG0:.*]], %[[ARG1:.*]] for %[[TIME:.*]], ^[[BB:.*]](%[[ARG1:.*]], %[[ARG0:.*]] : !llhd.sig<i1>, !llhd.sig<i64>) : !llhd.sig<i64>, !llhd.sig<i1>, !llhd.time
    "llhd.wait"(%arg0, %arg1, %time, %arg1, %arg0) [^bb1] {operand_segment_sizes=dense<[2,1,2]> : vector<3xi32>} : (!llhd.sig<i64>, !llhd.sig<i1>, !llhd.time, !llhd.sig<i1>, !llhd.sig<i64>) -> ()
    // CHECK: ^[[BB:.*]]
^bb1(%a : !llhd.sig<i1>, %b : !llhd.sig<i64>):
    // CHECK-NEXT: llhd.wait %[[ARG0:.*]], %[[A:.*]] for %[[TIME:.*]], ^[[BB:.*]](%[[A:.*]], %[[B:.*]] : !llhd.sig<i1>, !llhd.sig<i64>) : !llhd.sig<i64>, !llhd.sig<i1>, !llhd.time
    llhd.wait %arg0, %a for %time, ^bb1(%a, %b : !llhd.sig<i1>, !llhd.sig<i64>) : !llhd.sig<i64>, !llhd.sig<i1>, !llhd.time
}
