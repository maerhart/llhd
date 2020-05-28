//RUN: llhdc %s --convert-llhd-to-llvm | FileCheck %s

// CHECK-LABEL: @convert_const
llvm.func @convert_const() {
    // CHECK-NEXT: %{{.*}} = llvm.mlir.constant(1 : i1) : !llvm.i1
    %0 = llhd.const 1 : i1

    // CHECK-NEXT %{{.*}} = llvm.mlir.constant(0 : i32) : !llvm.i32
    %1 = llhd.const 0 : i32

    // this gets erased
    %2 = llhd.const #llhd.time<0ns, 0d, 0e> : !llhd.time

    // CHECK-NEXT %{{.*}} = llvm.mlir.constant(123 : i64) : !llvm.i64
    %3 = llhd.const 123 : i64

    llvm.return
}

// CHECK-LABEL: @convert_exts
// CHECK-SAME: %[[CI32:.*]]: !llvm.i32
// CHECK-SAME: %[[CI100:.*]]: !llvm.i100
func @convert_exts(%cI32 : i32, %cI100 : i100) {
    // CHECK-NEXT: %[[CIND0:.*]] = llvm.mlir.constant(0 : index) : !llvm.i64
    // CHECK-NEXT: %[[ADJUST0:.*]] = llvm.trunc %[[CIND0]] : !llvm.i64 to !llvm.i32
    // CHECK-NEXT: %[[SHR0:.*]] = llvm.lshr %[[CI32]], %[[ADJUST0]] : !llvm.i32
    // CHECK-NEXT: %{{.*}} = llvm.trunc %[[SHR0]] : !llvm.i32 to !llvm.i10
    %0 = llhd.exts %cI32, 0, 10 : i32 to i10
    // CHECK-NEXT: %[[CIND1:.*]] = llvm.mlir.constant(0 : index) : !llvm.i64
    // CHECK-NEXT: %[[ADJUST1:.*]] = llvm.zext %[[CIND1]] : !llvm.i64 to !llvm.i100
    // CHECK-NEXT: %[[SHR1:.*]] = llvm.lshr %[[CI100]], %[[ADJUST1]] : !llvm.i100
    // CHECK-NEXT: %{{.*}} = llvm.trunc %[[SHR1]] : !llvm.i100 to !llvm.i10
    %2 = llhd.exts %cI100, 0, 10 : i100 to i10

    return
}
