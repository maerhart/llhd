//RUN: llhdc %s --convert-llhd-to-llvm --split-input-file | FileCheck %s

func @convert_bitwise_i1(%lhs : i1, %rhs : i1) {
    // CHECK: %[[RHS:.*]] = llvm.mlir.constant(1 : i1) : !llvm.i1
    // CHECK-NEXT: %{{.*}} = llvm.xor %{{.*}}, %[[RHS:.*]] : !llvm.i1
    llhd.not %lhs : i1
    // CHECK-NEXT: %{{.*}} = llvm.and %{{.*}}, %{{.*}} : !llvm.i1
    llhd.and %lhs, %rhs : i1
    // CHECK-NEXT: %{{.*}} = llvm.or %{{.*}}, %{{.*}} : !llvm.i1
    llhd.or %lhs, %rhs : i1
    // CHECK-NEXT: %{{.*}} = llvm.xor %{{.*}}, %{{.*}} : !llvm.i1
    llhd.xor %lhs, %rhs : i1

    return
}

func @convert_bitwise_i32(%lhs : i32, %rhs : i32) {
    // CHECK: %[[RHS:.*]] = llvm.mlir.constant(-1 : i32) : !llvm.i32
    // CHECK-NEXT: %{{.*}} = llvm.xor %{{.*}}, %[[RHS:.*]] : !llvm.i32
    llhd.not %lhs : i32
    // CHECK-NEXT: %{{.*}} = llvm.and %{{.*}}, %{{.*}} : !llvm.i32
    llhd.and %lhs, %rhs : i32
    // CHECK-NEXT: %{{.*}} = llvm.or %{{.*}}, %{{.*}} : !llvm.i32
    llhd.or %lhs, %rhs : i32
    // CHECK-NEXT: %{{.*}} = llvm.xor %{{.*}}, %{{.*}} : !llvm.i32
    llhd.xor %lhs, %rhs : i32
    
    return
}
