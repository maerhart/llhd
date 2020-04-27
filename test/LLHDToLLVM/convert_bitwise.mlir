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
