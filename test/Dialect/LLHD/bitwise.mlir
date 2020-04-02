//RUN: llhdc %s | llhdc | FileCheck %s

// CHECK-LABEL: @check_bitwise
func @check_bitwise(%a : i64, %b : i64) -> () {
    // CHECK-NEXT: %{{.*}} = llhd.not %{{.*}} : i64
    %0 = "llhd.not"(%a) : (i64) -> i64
    // CHECK-NEXT: %{{.*}} = llhd.not %{{.*}} : i64
    %1 = llhd.not %a : i64

    // CHECK-NEXT: %{{.*}} = llhd.and %{{.*}}, %{{.*}} : i64
    %2 = "llhd.and"(%a, %b) : (i64, i64) -> i64
    // CHECK-NEXT: %{{.*}} = llhd.and %{{.*}}, %{{.*}} : i64
    %3 = llhd.and %a, %b : i64

    // CHECK-NEXT: %{{.*}} = llhd.or %{{.*}}, %{{.*}} : i64
    %4 = "llhd.or"(%a, %b) : (i64, i64) -> i64
    // CHECK-NEXT: %{{.*}} = llhd.or %{{.*}}, %{{.*}} : i64
    %5 = llhd.or %a, %b : i64

    // CHECK-NEXT: %{{.*}} = llhd.xor %{{.*}}, %{{.*}} : i64
    %6 = "llhd.xor"(%a, %b) : (i64, i64) -> i64
    // CHECK-NEXT: %{{.*}} = llhd.xor %{{.*}}, %{{.*}} : i64
    %7 = llhd.xor %a, %b : i64

    return
}