// RUN: llhdc %s | llhdc | FileCheck %s

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