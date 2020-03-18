// RUN: llhdc %s | llhdc | FileCheck %s

// CHECK-LABEL: func @foo
// CHECK-NEXT: %{{.*}} = llhd.const 1 : !llhd.sig<i64>
// CHECK-NEXT: %{{.*}} = llhd.const 254 : i64
// CHECK-NEXT: %{{.*}} = llhd.sig %{{.*}} : i64 
// CHECK-NEXT: %{{.*}} = llhd.prb %{{.*}} : !llhd.sig<i64>
// CHECK-NEXT: llhd.drv %{{.*}}, %{{.*}} : !llhd.sig<i64>
// CHECK-NEXT: return %{{.*}} : !llhd.sig<i64>
// CHECK-NEXT: }
func @foo() -> !llhd.sig<i64> {
    %sig = "llhd.const"() {value = 1} : () -> !llhd.sig<i64>
    %const = "llhd.const"() {value = 254} : () -> i64
    %sig_sig = "llhd.sig"(%const) {} : (i64) -> !llhd.sig<i64> 
    %prb = "llhd.prb"(%sig_sig) {} : (!llhd.sig<i64>) -> i64
    "llhd.drv"(%sig, %const) {} : (!llhd.sig<i64>, i64) -> ()
    return %sig_sig : !llhd.sig<i64>
}