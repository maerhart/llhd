//RUN: llhdc %s --convert-llhd-to-llvm --split-input-file | FileCheck %s

// CHECK: llvm.func @drive_signal(!llvm<"i8*">, !llvm.i32, !llvm.i1, !llvm.i32)
// CHECK-NEXT: llvm.func @probe_signal(!llvm<"i8*">, !llvm.i32) -> !llvm<"i8*">
// CHECK-NEXT: llvm.func @alloc_signal(!llvm<"i8*">, !llvm.i32, !llvm.i1) -> !llvm.i32
// CHECK-NEXT: llvm.func @Foo(%[[STATE:.*]]: !llvm<"i8*">) {
// CHECK-NEXT: %[[C1:.*]] = llvm.mlir.constant(0 : i1) : !llvm.i1
// CHECK-NEXT: %[[C2:.*]] = llvm.mlir.constant(0 : i32) : !llvm.i32
// CHECK-NEXT: %[[CALL1:.*]] = llvm.call @alloc_signal(%[[STATE:.*]], %[[C2.*]], %[[C1:.*]]) : (!llvm<"i8*">, !llvm.i32, !llvm.i1) -> !llvm.i32
// CHECK-NEXT: %[[CALL2:.*]] = llvm.call @probe_signal(%[[STATE:.*]], %[[CALL1:.*]]) : (!llvm<"i8*">, !llvm.i32) -> !llvm<"i8*">
// CHECK-NEXT: %[[B1:.*]] = llvm.bitcast %[[CALL2:.*]] : !llvm<"i8*"> to !llvm<"i1*">
// CHECK-NEXT: %[[L1:.*]] = llvm.load %[[B1:.*]] : !llvm<"i1*">
// CHECK-NEXT: %[[C3:.*]] = llvm.mlir.constant(1 : i1) : !llvm.i1
// CHECK-NEXT: %[[X1:.*]] = llvm.xor %[[CALL2:.*]], %[[C3:.*]] : !llvm.i1
// CHECK-NEXT: %[[C4:.*]] = llvm.mlir.constant(1 : i32) : !llvm.i32
// CHECK-NEXT: %[[CALL3:.*]] = llvm.call @drive_signal(%[[STATE:.*]], %[[CALL1:.*]], %[[X1:.*]], %[[C4:.*]]) : (!llvm<"i8*">, !llvm.i32, !llvm.i1, !llvm.i32) -> !llvm.void
// CHECK-NEXT: llvm.return
// CHECK-NEXT: }

llhd.entity @Foo () -> () {
    %0 = llhd.const 0 : i1
    %toggle = llhd.sig "toggle" %0 : i1 -> !llhd.sig<i1>
    %1 = llhd.prb %toggle : !llhd.sig<i1> -> i1
    %2 = llhd.not %1 : i1
    %dt = llhd.const #llhd.time<1ns, 0d, 0e> : !llhd.time
    llhd.drv %toggle, %2, %dt : !llhd.sig<i1>, i1, !llhd.time
}
