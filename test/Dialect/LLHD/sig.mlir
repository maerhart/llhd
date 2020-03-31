// RUN: llhdc %s | llhdc | FileCheck %s

// check the construction of a new signal with the sig instruction
func @check_sig_inst() {
    // CHECK: %[[BOOL:.*]] = llhd.const
    %0 = llhd.const 1 : i1
    // CHECK-NEXT: %[[BSIG:.*]] = llhd.sig %[[BOOL:.*]] : i1 -> !llhd.sig<i1>
    %1 = "llhd.sig"(%0) {} : (i1) -> !llhd.sig<i1>
    // CHECK-NEXT: %[[INT:.*]] = llhd.const
    %2 = llhd.const 256 : i64
    // CHECK-NEXT: %[[ISIG:.*]] = llhd.sig %[[INT:.*]] : i64 -> !llhd.sig<i64>
    %3 = "llhd.sig"(%2) {} : (i64) -> !llhd.sig<i64>
    return
}

func @check_prb() {
    %0 = llhd.const 1 : i1
    // CHECK: %[[SIG:.*]] = llhd.sig
    %1 = llhd.sig %0 : i1 -> !llhd.sig<i1>
    // CHECK: %{{.*}} = llhd.prb %[[SIG:.*]] : !llhd.sig<i1> -> i1
    %2 = "llhd.prb"(%1) {} : (!llhd.sig<i1>) -> i1

    return
}

func @check_drv() {
    // CHECK: %[[TIME:.*]] = llhd.const
    %0 = llhd.const #llhd.time<1ns, 0d, 0e> : !llhd.time
    // CHECK-NEXT: %[[CONST2:.*]] = llhd.const
    %1 = llhd.const 1 : i1
    // CHECK-NEXT: %[[SIG2:.*]] = llhd.sig
    %2 = llhd.sig %1 : i1 -> !llhd.sig<i1>
    // CHECK-NEXT: llhd.drv %[[SIG2:.*]], %[[CONST2:.*]], %[[TIME:.*]] : !llhd.sig<i1>, i1, !llhd.time
    llhd.drv %2, %1, %0 : !llhd.sig<i1>, i1, !llhd.time

    return
}
