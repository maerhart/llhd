//RUN: llhdc %s --convert-llhd-to-llvm --split-input-file | FileCheck %s

func @convert_bitwise(%lhs : i1, %rhs : i1) {
    // CHECK: %[[RHS:.*]] = llvm.mlir.constant(true) : !llvm.i1
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
