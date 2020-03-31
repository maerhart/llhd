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
