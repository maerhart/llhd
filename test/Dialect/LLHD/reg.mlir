// RUN: llhdc %s -split-input-file -verify-diagnostics | llhdc | FileCheck %s

// CHECK-LABEL: @check_reg
// CHECK-SAME: %[[IN1:.*]] : !llhd.sig<i1>
// CHECK-SAME: %[[IN64:.*]] : !llhd.sig<i64>
llhd.entity @check_reg (%in1 : !llhd.sig<i1>, %in64 : !llhd.sig<i64>) -> () {
    // CHECK: %[[C1:.*]] = llhd.const
    %c1 = llhd.const 0 : i1
    // CHECK-NEXT: %[[C64:.*]] = llhd.const
    %c64 = llhd.const 0 : i64
    // zero triggers
    // CHECK-NEXT: %{{.*}} = llhd.reg %[[C64]] : i64
    %0 = "llhd.reg"(%c64) {modes=[], gateMask=[], operand_segment_sizes=dense<[1,0,0,0]> : vector<4xi32>} : (i64) -> !llhd.sig<i64>
    // one trigger with optional gate
    // CHECK-NEXT: %{{.*}} = llhd.reg %[[C64]], (%[[C64]], "low" %[[C1]] if %[[C1]] : i64, i1, i1) : i64
    %1 = "llhd.reg"(%c64, %c64, %c1, %c1) {modes=[0], gateMask=[1], operand_segment_sizes=dense<[1,1,1,1]> : vector<4xi32>} : (i64, i64, i1, i1) -> !llhd.sig<i64>
    // two triggers with optional gates
    // CHECK-NEXT: %{{.*}} = llhd.reg %[[C64]], (%[[C64]], "low" %[[C1]] if %[[IN1]] : i64, i1, !llhd.sig<i1>), (%[[IN64]], "high" %[[IN1]] if %[[C1]] : !llhd.sig<i64>, !llhd.sig<i1>, i1) : i64
    %2 = "llhd.reg"(%c64, %c64, %in64, %c1, %in1, %in1, %c1) {modes=[0,1], gateMask=[1,2], operand_segment_sizes=dense<[1,2,2,2]> : vector<4xi32>} : (i64, i64, !llhd.sig<i64>, i1, !llhd.sig<i1>, !llhd.sig<i1>, i1) -> !llhd.sig<i64>
    // two triggers with only one optional gate
    // CHECK-NEXT: %{{.*}} = llhd.reg %[[C64]], (%[[C64]], "low" %[[C1]] : i64, i1), (%[[IN64]], "high" %[[IN1]] if %[[IN1]] : !llhd.sig<i64>, !llhd.sig<i1>, !llhd.sig<i1>) : i64
    %3 = "llhd.reg"(%c64, %c64, %in64, %c1, %in1, %in1) {modes=[0,1], gateMask=[0,1], operand_segment_sizes=dense<[1,2,2,1]> : vector<4xi32>} : (i64, i64, !llhd.sig<i64>, i1, !llhd.sig<i1>, !llhd.sig<i1>) -> !llhd.sig<i64>
}

// TODO: add verification tests (expected-error tests)
