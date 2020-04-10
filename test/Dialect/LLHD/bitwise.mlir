//RUN: llhdc %s | llhdc | FileCheck %s

// CHECK-LABEL: @check_bitwise
func @check_bitwise(%a : i64, %b : i64, %c : i8, %sig1 : !llhd.sig<i32>, %sig2 : !llhd.sig<i4>) -> () {
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

    // CHECK-NEXT: %{{.*}} = llhd.shl %{{.*}}, %{{.*}}, %{{.*}} : (i64, i64, i8) -> i64
    %8 = "llhd.shl"(%a, %b, %c) : (i64, i64, i8) -> i64
    // CHECK-NEXT: %{{.*}} = llhd.shl %{{.*}}, %{{.*}}, %{{.*}} : (i64, i64, i8) -> i64
    %9 = llhd.shl %a, %b, %c : (i64, i64, i8) -> i64

    // CHECK-NEXT: %{{.*}} = llhd.shl %{{.*}}, %{{.*}}, %{{.*}} : (!llhd.sig<i32>, !llhd.sig<i4>, i8) -> !llhd.sig<i32>
    %10 = "llhd.shl"(%sig1, %sig2, %c) : (!llhd.sig<i32>, !llhd.sig<i4>, i8) -> !llhd.sig<i32>
    // CHECK-NEXT: %{{.*}} = llhd.shl %{{.*}}, %{{.*}}, %{{.*}} : (!llhd.sig<i32>, !llhd.sig<i4>, i8) -> !llhd.sig<i32>
    %11 = llhd.shl %sig1, %sig2, %c : (!llhd.sig<i32>, !llhd.sig<i4>, i8) -> !llhd.sig<i32>

    // CHECK-NEXT: %{{.*}} = llhd.shr %{{.*}}, %{{.*}}, %{{.*}} : (i64, i64, i8) -> i64
    %12 = "llhd.shr"(%a, %b, %c) : (i64, i64, i8) -> i64
    // CHECK-NEXT: %{{.*}} = llhd.shr %{{.*}}, %{{.*}}, %{{.*}} : (i64, i64, i8) -> i64
    %13 = llhd.shr %a, %b, %c : (i64, i64, i8) -> i64

    // CHECK-NEXT: %{{.*}} = llhd.shr %{{.*}}, %{{.*}}, %{{.*}} : (!llhd.sig<i32>, !llhd.sig<i4>, i8) -> !llhd.sig<i32>
    %14 = "llhd.shr"(%sig1, %sig2, %c) : (!llhd.sig<i32>, !llhd.sig<i4>, i8) -> !llhd.sig<i32>
    // CHECK-NEXT: %{{.*}} = llhd.shr %{{.*}}, %{{.*}}, %{{.*}} : (!llhd.sig<i32>, !llhd.sig<i4>, i8) -> !llhd.sig<i32>
    %15 = llhd.shr %sig1, %sig2, %c : (!llhd.sig<i32>, !llhd.sig<i4>, i8) -> !llhd.sig<i32>

    return
}
