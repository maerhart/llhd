// RUN: llhdc %s -split-input-file -verify-diagnostics | llhdc | FileCheck %s

llhd.entity @check_sig_inst () -> () {
    // CHECK: %[[CI1:.*]] = llhd.const
    %cI1 = llhd.const 0 : i1
    // CHECK-NEXT: %{{.*}} = llhd.sig "sigI1" %[[CI1]] : i1 -> !llhd.sig<i1>
    %sigI1 = "llhd.sig"(%cI1) {name = "sigI1"} : (i1) -> !llhd.sig<i1>
    // CHECK-NEXT: %[[CI64:.*]] = llhd.const
    %cI64 = llhd.const 0 : i64
    // CHECK-NEXT: %{{.*}} = llhd.sig "sigI64" %[[CI64]] : i64 -> !llhd.sig<i64>
    %sigI64 = "llhd.sig"(%cI64) {name = "sigI64"} : (i64) -> !llhd.sig<i64>
}

// CHECK-LABEL: check_prb
// CHECK-SAME: %[[SI1:.*]]: !llhd.sig<i1>
// CHECK-SAME: %[[SI64:.*]]: !llhd.sig<i64>
func @check_prb(%sigI1 : !llhd.sig<i1>, %sigI64 : !llhd.sig<i64>) {
    // CHECK: %{{.*}} = llhd.prb %[[SI1]] : !llhd.sig<i1> -> i1
    %0 = "llhd.prb"(%sigI1) {} : (!llhd.sig<i1>) -> i1
    // CHECK-NEXT: %{{.*}} = llhd.prb %[[SI64]] : !llhd.sig<i64> -> i64
    %1 = "llhd.prb"(%sigI64) {} : (!llhd.sig<i64>) -> i64

    return
}

// CHECK-LABEL: check_drv
// CHECK-SAME: %[[SI1:.*]]: !llhd.sig<i1>
// CHECK-SAME: %[[SI64:.*]]: !llhd.sig<i64>
// CHECK-SAME: %[[CI1:.*]]: i1
// CHECK-SAME: %[[CI64:.*]]: i64
// CHECK-SAME: %[[TIME:.*]]: !llhd.time
func @check_drv(%sigI1 : !llhd.sig<i1>, %sigI64 : !llhd.sig<i64>, %cI1 : i1, %cI64 : i64, %t : !llhd.time) {
    // CHECK-NEXT: llhd.drv %[[SI1]], %[[CI1]], %[[TIME]] : !llhd.sig<i1>, i1, !llhd.time
    "llhd.drv"(%sigI1, %cI1, %t) {} : (!llhd.sig<i1>, i1, !llhd.time) -> ()
    // CHECK-NEXT: llhd.drv %[[SI64]], %[[CI64]], %[[TIME]] : !llhd.sig<i64>, i64, !llhd.time
    "llhd.drv" (%sigI64, %cI64, %t) {} : (!llhd.sig<i64>, i64, !llhd.time) -> ()
    // CHECK-NEXT: llhd.drv %[[SI64]], %[[CI64]], %[[TIME]], %[[CI1]] : !llhd.sig<i64>, i64, !llhd.time, i1
    "llhd.drv" (%sigI64, %cI64, %t, %cI1) {} : (!llhd.sig<i64>, i64, !llhd.time, i1) -> ()

    return
}

// -----

// expected-error @+3 {{The operand type is not equal to the signal type. Expected 'i32' but got 'i1'}}
llhd.entity @check_illegal_sig () -> () {
    %cI1 = llhd.const 0 : i1
    %sig1 = llhd.sig "foo" %cI1 : i1 -> !llhd.sig<i32>
}

// -----

// expected-error @+2 {{The operand type is not equal to the signal type. Expected 'i1' but got 'i32'}}
llhd.entity @check_illegal_prb (%sig : !llhd.sig<i1>) -> () {
    %prb = llhd.prb %sig : !llhd.sig<i1> -> i32
}

// -----

// expected-error @+4 {{The new value's type is not equal to the signal type. Expected 'i1' but got 'i32'}}
llhd.entity @check_illegal_drv (%sig : !llhd.sig<i1>) -> () {
    %c = llhd.const 0 : i32
    %time = llhd.const #llhd.time<1ns, 0d, 0e> : !llhd.time
    llhd.drv %sig, %c, %time : !llhd.sig<i1>, i32, !llhd.time
}

// -----

// expected-error @+4 {{Redefinition of signal named 'sigI1'!}}
llhd.entity @check_unique_sig_names () -> () {
    %cI1 = llhd.const 0 : i1
    %sig1 = llhd.sig "sigI1" %cI1 : i1 -> !llhd.sig<i1>
    %sig2 = llhd.sig "sigI1" %cI1 : i1 -> !llhd.sig<i1>
}
