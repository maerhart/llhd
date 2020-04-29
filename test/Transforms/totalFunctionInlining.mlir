// RUN: llhdc %s -inline="disable-simplify" -llhd-function-elimination | FileCheck %s

// This test checks the presence of inlining into entities and processes
// and their general structure after inlining. It also checks that the functions
// are deleted by the elimination pass.
// Note: Only functions which can be reduced to one basic block can be inlined 
// into entities.

// CHECK-NOT: func
func @simple() -> i32 {
    %0 = llhd.const 5 : i32
    return %0 : i32
} 

// CHECK-NOT: func
func @complex(%flag : i1) -> i32 {
    cond_br %flag, ^bb1, ^bb2
^bb1:
    %0 = llhd.const 5 : i32
    return %0 : i32
^bb2:
    %1 = llhd.const 7 : i32
    return %1 : i32
}

// CHECK-LABEL: @check_entity_inline
llhd.entity @check_entity_inline() -> (%out : !llhd.sig<i32>) {
    // CHECK-NEXT: %{{.*}} = llhd.const
    // CHECK-NEXT: %{{.*}} = llhd.const
    // CHECK-NEXT: llhd.drv
    // CHECK-NEXT: }
    %1 = call @simple() : () -> i32
    %time = llhd.const #llhd.time<1ns, 0d, 0e> : !llhd.time
    llhd.drv %out, %1, %time : !llhd.sig<i32>, i32, !llhd.time
}

// CHECK-LABEL: @check_proc_inline
llhd.proc @check_proc_inline(%arg : !llhd.sig<i1>) -> (%out : !llhd.sig<i32>) {
    // CHECK-NEXT: %[[PRB:.*]] = llhd.prb
    // CHECK-NEXT: cond_br %[[PRB]], ^[[BB1:.*]], ^[[BB2:.*]]
    // CHECK-NEXT: ^[[BB1]]:
    // CHECK-NEXT: %[[C0:.*]] = llhd.const
    // CHECK-NEXT: br ^[[BB3:.*]](%[[C0]] : i32)
    // CHECK-NEXT: ^[[BB2]]:
    // CHECK-NEXT: %[[C1:.*]] = llhd.const
    // CHECK-NEXT: br ^[[BB3]](%[[C1]] : i32)
    // CHECK-NEXT: ^[[BB3]](%[[A:.*]]: i32):
    // CHECK-NEXT: %[[C2:.*]] = llhd.const
    // CHECK-NEXT: llhd.drv %{{.*}}, %[[A]], %[[C2]] : !llhd.sig<i32>, i32, !llhd.time
    // CHECK-NEXT: llhd.halt
    // CHECK-NEXT: }
    %0 = llhd.prb %arg : !llhd.sig<i1> -> i1
    %1 = call @complex(%0) : (i1) -> i32
    %time = llhd.const #llhd.time<1ns, 0d, 0e> : !llhd.time
    llhd.drv %out, %1, %time : !llhd.sig<i32>, i32, !llhd.time
    llhd.halt
}
