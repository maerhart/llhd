// RUN: llhdc %s | FileCheck %s

// CHECK-LABEL: func @foo
func @foo() -> !llhd.sig<i64> {
// CHECK-NEXT: %{{.*}} = llhd.const !llhd.sig<i64> 1
// CHECK-NEXT: %{{.*}} = llhd.const !llhd.sig<i64> 2
// CHECK-NEXT: %{{.*}} = llhd.const i64 254
// CHECK-NEXT: %{{.*}} = llhd.sig i64 %{{.*}}
// CHECK-NEXT: %{{.*}} = llhd.sig i64 %{{.*}}
// CHECK-NEXT: return %{{.*}} : !llhd.sig<i64>
// CHECK-NEXT: }
    %0 = "llhd.const"() {value = 1} : () -> !llhd.sig<i64>
    %1 = llhd.const !llhd.sig<i64> 2
    %2 = llhd.const i64 254
    %3 = "llhd.sig"(%2) {} : (i64) -> !llhd.sig<i64> 
    %4 = llhd.sig i64 %2
    return %1 : !llhd.sig<i64>
}