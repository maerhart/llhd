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

func @foo() {
    // CHECK: %[[CONST:.*]] = llhd.const 254 : i64
    %const = "llhd.const"() {value = 254} : () -> i64
    // CHECK-NEXT: %[[SIG:.*]] = llhd.sig %[[CONST:.*]] : i64 
    %sig = "llhd.sig"(%const) {} : (i64) -> !llhd.sig<i64> 
    // CHECK-NEXT: %[[PRB:.*]] = llhd.prb %[[SIG:.*]] : !llhd.sig<i64>
    %prb = "llhd.prb"(%sig) {} : (!llhd.sig<i64>) -> i64
    // CHECK-NEXT: llhd.drv %[[SIG:.*]], %[[CONST:.*]] : !llhd.sig<i64>
    "llhd.drv"(%sig, %const) {} : (!llhd.sig<i64>, i64) -> ()

    return
}