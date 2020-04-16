//RUN: llhdc %s --convert-llhd-to-llvm --split-input-file | FileCheck %s

llvm.func @convert_const() {
    // CHECK: %{{.*}} = llvm.mlir.constant(1 : i1) : !llvm.i1
    %0 = llhd.const 1 : i1

    // CHECK-NEXT %{{.*}} = llvm.mlir.constant(0 : i32) : !llvm.i32
    %1 = llhd.const 0 : i32

    // this gets erased
    %2 = llhd.const #llhd.time<0ns, 0d, 0e> : !llhd.time

    // CHECK-NEXT %{{.*}} = llvm.mlir.constant(123 : i64) : !llvm.i64
    %3 = llhd.const 123 : i64

    llvm.return
}
