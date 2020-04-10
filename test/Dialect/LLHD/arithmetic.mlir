//RUN: llhdc %s | llhdc | FileCheck %s

// CHECK-LABEL: @check_arithmetic
func @check_arithmetic(%a : i64, %b : i64) -> () {
    // CHECK-NEXT: %{{.*}} = llhd.neg %{{.*}} : i64
    %0 = "llhd.neg"(%a) : (i64) -> i64
    // CHECK-NEXT: %{{.*}} = llhd.neg %{{.*}} : i64
    %1 = llhd.neg %a : i64

    // CHECK-NEXT: %{{.*}} = llhd.smod %{{.*}}, %{{.*}} : i64
    %2 = "llhd.smod"(%a, %b) : (i64, i64) -> i64
    // CHECK-NEXT: %{{.*}} = llhd.smod %{{.*}}, %{{.*}} : i64
    %3 = llhd.smod %a, %b : i64

    return
}
