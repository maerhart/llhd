//RUN: llhdc %s --convert-llhd-to-llvm --split-input-file | FileCheck %s

func @convert_arithmetic(%lhs : i1) {
    // CHECK: %[[RHS:.*]] = llvm.mlir.constant(true) : !llvm.i1
    // CHECK-NEXT: %{{.*}} = llvm.xor %{{.*}}, %[[RHS:.*]] : !llvm.i1
    llhd.not %lhs : i1

    return
}
