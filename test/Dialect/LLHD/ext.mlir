// RUN: llhdc %s -mlir-print-op-generic -split-input-file -verify-diagnostics | llhdc | FileCheck %s

// CHECK-LABEL: @exts_integers
// CHECK-SAME: %[[CI1:.*]]: i1
// CHECK-SAME: %[[CI32:.*]]: i32
func @exts_integers(%cI1 : i1, %cI32 : i32) {
    // CHECK-NEXT: %{{.*}} = llhd.exts %[[CI1]], 0, 1 : i1 to i1
    %0 = llhd.exts %cI1, 0, 1 : i1 to i1
    // CHECK-NEXT: %{{.*}} = llhd.exts %[[CI32]], 0, 5 : i32 to i5
    %1 = llhd.exts %cI32, 0, 5 : i32 to i5

    return
}

// CHECK-LABEL: @exts_signals
// CHECK-SAME: %[[SI1:.*]]: !llhd.sig<i1>
// CHECK-SAME: %[[SI32:.*]]: !llhd.sig<i32>
func @exts_signals (%sI1 : !llhd.sig<i1>, %sI32 : !llhd.sig<i32>) -> () {
    // CHECK-NEXT: %{{.*}} = llhd.exts %[[SI1]], 0, 1 : !llhd.sig<i1> to !llhd.sig<i1>
    %0 = llhd.exts %sI1, 0, 1 : !llhd.sig<i1> to !llhd.sig<i1>
    // CHECK-NEXT: %{{.*}} = llhd.exts %[[SI32]], 0, 5 : !llhd.sig<i32> to !llhd.sig<i5>
    %1 = llhd.exts %sI32, 0, 5 : !llhd.sig<i32> to !llhd.sig<i5>

    return
}

// -----

func @illegal_int_to_sig(%c : i32) {
    // expected-error @+1 {{the target and result kinds have to match}}
    %0 = llhd.exts %c, 0, 10 : i32 to !llhd.sig<i10>

    return
}

// -----

func @illegal_sig_to_int(%s : !llhd.sig<i32>) {
    // expected-error @+1 {{the target and result kinds have to match}}
    %0 = llhd.exts %s, 0, 10 : !llhd.sig<i32> to i10

    return
}

// -----

 func @illegal_out_int_width(%c : i32) {
     // expected-error @+1 {{the result bit width has to match the given length. Expected 10 but got 5}}
    %0 = llhd.exts %c, 0, 10 : i32 to i5

     return
 }

// -----

func @illegal_out_sig_width(%s : !llhd.sig<i32>) {
    // expected-error @+1 {{the result bit width has to match the given length. Expected 10 but got 5}}
    %0 = llhd.exts %s, 0, 10 : !llhd.sig<i32> to !llhd.sig<i5>

    return
}

// -----

func @illegal_out_too_big(%c : i32) {
    // expected-error @+1 {{the result bit width cannot be larger than the target width}}
    %0 = llhd.exts %c, 0, 40 : i32 to i40

    return
}