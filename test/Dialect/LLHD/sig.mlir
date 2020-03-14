// RUN: llhdc %s | FileCheck %s

// CHECK-LABEL: func @foo
// CHECK-NEXT: %{{.*}} = llhd.const !llhd.sig<i64> 1
// CHECK-NEXT: return %{{.*}} : !llhd.sig<i64>
// CHECK-NEXT: }
func @foo() -> !llhd.sig<i64> {
    %0 = "llhd.const"() {value = 1} : () -> !llhd.sig<i64>
    return %0 : !llhd.sig<i64>
}