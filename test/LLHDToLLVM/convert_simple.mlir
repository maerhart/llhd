//RUN: llhdc %s --convert-llhd-to-llvm | FileCheck %s

// CHECK: llvm.func @Foo(%[[STATE:.*]]: !llvm<"i8*">, %[[SIGTAB:.*]]: !llvm<"i32*">, %[[ARGTAB:.*]]: !llvm<"i32*">) {
// CHECK-NEXT: %[[C1:.*]] = llvm.mlir.constant(0 : i1) : !llvm.i1
// CHECK-NEXT: %[[IDX0:.*]] = llvm.mlir.constant(0 : i32) : !llvm.i32
// CHECK-NEXT: %[[GEP0:.*]] = llvm.getelementptr %[[SIGTAB]][%[[IDX0]]] : (!llvm<"i32*">, !llvm.i32) -> !llvm<"i32*">
// CHECK-NEXT: %[[LOAD0:.*]] = llvm.load %[[GEP0]] : !llvm<"i32*">
// CHECK-NEXT: %[[CALL2:.*]] = llvm.call @probe_signal(%[[STATE]], %[[LOAD0]]) : (!llvm<"i8*">, !llvm.i32) -> !llvm<"i8*">
// CHECK-NEXT: %[[B1:.*]] = llvm.bitcast %[[CALL2]] : !llvm<"i8*"> to !llvm<"i1*">
// CHECK-NEXT: %[[L1:.*]] = llvm.load %[[B1]] : !llvm<"i1*">
// CHECK-NEXT: %[[C3:.*]] = llvm.mlir.constant(1 : i1) : !llvm.i1
// CHECK-NEXT: %[[X1:.*]] = llvm.xor %[[L1]], %[[C3]] : !llvm.i1
// CHECK-NEXT: %[[ARRSIZE0:.*]] = llvm.mlir.constant(1 : i32) : !llvm.i32
// CHECK-NEXT: %[[ALLOCA0:.*]] = llvm.alloca %[[ARRSIZE0]] x !llvm.i1 {alignment = 4 : i64} : (!llvm.i32) -> !llvm<"i1*">
// CHECK-NEXT: llvm.store %[[X1]], %[[ALLOCA0]] : !llvm<"i1*">
// CHECK-NEXT: %[[BC2:.*]] = llvm.bitcast %[[ALLOCA0]] : !llvm<"i1*"> to !llvm<"i8*">
// CHECK-NEXT: %[[SIZE0:.*]] = llvm.mlir.constant(1 : i64) : !llvm.i64
// CHECK-NEXT: %[[TIME:.*]] = llvm.mlir.constant(1 : i32) : !llvm.i32
// CHECK-NEXT: %[[DELTA:.*]] = llvm.mlir.constant(0 : i32) : !llvm.i32
// CHECK-NEXT: %[[EPS:.*]] = llvm.mlir.constant(0 : i32) : !llvm.i32
// CHECK-NEXT: %[[CALL3:.*]] = llvm.call @drive_signal(%[[STATE]], %[[LOAD0]], %[[BC2]], %[[SIZE0:.*]], %[[TIME]], %[[DELTA]], %[[EPS]]) : (!llvm<"i8*">, !llvm.i32, !llvm<"i8*">, !llvm.i64, !llvm.i32, !llvm.i32, !llvm.i32) -> !llvm.void
// CHECK-NEXT: llvm.return

llhd.entity @Foo () -> () {
    %0 = llhd.const 0 : i1
    %toggle = llhd.sig "toggle" %0 : i1
    %1 = llhd.prb %toggle : !llhd.sig<i1>
    %2 = llhd.not %1 : i1
    %dt = llhd.const #llhd.time<1ns, 0d, 0e> : !llhd.time
    llhd.drv %toggle, %2 after %dt : !llhd.sig<i1>
}
