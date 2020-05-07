//RUN: llhdc %s --convert-llhd-to-llvm | FileCheck %s

// CHECK-LABEL: convert_bitwise_i1
// CHECK-SAME: %[[LHS:.*]]: !llvm.i1,
// CHECK-SAME: %[[RHS:.*]]: !llvm.i1
func @convert_bitwise_i1(%lhs : i1, %rhs : i1) {
    // CHECK-NEXT: %[[MASK:.*]] = llvm.mlir.constant(1 : i1) : !llvm.i1
    // CHECK-NEXT: %{{.*}} = llvm.xor %[[LHS]], %[[MASK]] : !llvm.i1
    %0 = llhd.not %lhs : i1
    // CHECK-NEXT: %{{.*}} = llvm.and %[[LHS]], %[[RHS]] : !llvm.i1
    %1 = llhd.and %lhs, %rhs : i1
    // CHECK-NEXT: %{{.*}} = llvm.or %[[LHS]], %[[RHS]] : !llvm.i1
    %2 = llhd.or %lhs, %rhs : i1
    // CHECK-NEXT: %{{.*}} = llvm.xor %[[LHS]], %[[RHS]] : !llvm.i1
    %3 = llhd.xor %lhs, %rhs : i1

    return
}

// CHECK-LABEL: convert_bitwise_i32
// CHECK-SAME: %[[LHS:.*]]: !llvm.i32,
// CHECK-SAME: %[[RHS:.*]]: !llvm.i32
func @convert_bitwise_i32(%lhs : i32, %rhs : i32) {
    // CHECK-NEXT: %[[MASK:.*]] = llvm.mlir.constant(-1 : i32) : !llvm.i32
    // CHECK-NEXT: %{{.*}} = llvm.xor %[[LHS]], %[[MASK]] : !llvm.i32
    llhd.not %lhs : i32
    // CHECK-NEXT: %{{.*}} = llvm.and %[[LHS]], %[[RHS]] : !llvm.i32
    llhd.and %lhs, %rhs : i32
    // CHECK-NEXT: %{{.*}} = llvm.or %[[LHS]], %[[RHS]] : !llvm.i32
    llhd.or %lhs, %rhs : i32
    // CHECK-NEXT: %{{.*}} = llvm.xor %[[LHS]], %[[RHS]] : !llvm.i32
    llhd.xor %lhs, %rhs : i32

    return
}

// CHECK-LABEL: convert_shl_i5_i2_i2
// CHECK-SAME: %[[BASE:.*]]: !llvm.i5,
// CHECK-SAME: %[[HIDDEN:.*]]: !llvm.i2,
// CHECK-SAME: %[[AMOUNT:.*]]: !llvm.i2
func @convert_shl_i5_i2_i2(%base : i5, %hidden : i2, %amount : i2) {
    // CHECK-NEXT: %[[ZEXTB:.*]] = llvm.zext %[[BASE]] : !llvm.i5 to !llvm.i7
    // CHECK-NEXT: %[[ZEXTH:.*]] = llvm.zext %[[HIDDEN]] : !llvm.i2 to !llvm.i7
    // CHECK-NEXT: %[[ZEXTA:.*]] = llvm.zext %[[AMOUNT]] : !llvm.i2 to !llvm.i7
    // CHECK-NEXT: %[[HDNW:.*]] = llvm.mlir.constant(2 : i7) : !llvm.i7
    // CHECK-NEXT: %[[SHB:.*]] = llvm.shl %[[ZEXTB]], %[[HDNW]] : !llvm.i7
    // CHECK-NEXT: %[[COMB:.*]] = llvm.or %[[SHB]], %[[ZEXTH]] : !llvm.i7
    // CHECK-NEXT: %[[SA:.*]] = llvm.sub %[[HDNW]], %[[ZEXTA]] : !llvm.i
    // CHECK-NEXT: %[[SH:.*]] = llvm.lshr %[[COMB]], %[[SA]] : !llvm.i7
    // CHECK-NEXT: %{{.*}} = llvm.trunc %[[SH]] : !llvm.i7 to !llvm.i5
    %0 = llhd.shl %base, %hidden, %amount : (i5, i2, i2) -> i5

    return
}

// CHECK-LABEL: convert_shr_i5_i2_i2
// CHECK-SAME: %[[BASE:.*]]: !llvm.i5,
// CHECK-SAME: %[[HIDDEN:.*]]: !llvm.i2,
// CHECK-SAME: %[[AMOUNT:.*]]: !llvm.i2
func @convert_shr_i5_i2_i2(%base : i5, %hidden : i2, %amount : i2) {
    // CHECK-NEXT: %[[ZEXTB:.*]] = llvm.zext %[[BASE]] : !llvm.i5 to !llvm.i7
    // CHECK-NEXT: %[[ZEXTH:.*]] = llvm.zext %[[HIDDEN]] : !llvm.i2 to !llvm.i7
    // CHECK-NEXT: %[[ZEXTA:.*]] = llvm.zext %[[AMOUNT]] : !llvm.i2 to !llvm.i7
    // CHECK-NEXT: %[[BASEW:.*]] = llvm.mlir.constant(5 : i7) : !llvm.i7
    // CHECK-NEXT: %[[SHH:.*]] = llvm.shl %[[ZEXTH]], %[[BASEW]] : !llvm.i7
    // CHECK-NEXT: %[[COMB:.*]] = llvm.or %[[SHH]], %[[ZEXTB]] : !llvm.i7
    // CHECK-NEXT: %[[SH:.*]] = llvm.lshr %[[COMB]], %[[ZEXTA]] : !llvm.i7
    // CHECK-NEXT: %{{.*}} = llvm.trunc %[[SH]] : !llvm.i7 to !llvm.i5
    %0 = llhd.shr %base, %hidden, %amount : (i5, i2, i2) -> i5

    return
}
