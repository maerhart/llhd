//RUN: llhdc %s --convert-llhd-to-llvm | FileCheck %s

// CHECK-LABEL: @llhd_init
// CHECK-SAME: %[[STATE:.*]]: !llvm<"i8*">
// CHECK-NEXT: %[[ADDR0:.*]] = llvm.mlir.addressof @entity.sig_conversions : !llvm<"[16 x i8]*">
// CHECK-NEXT: %[[C0:.*]] = llvm.mlir.constant(0 : i32) : !llvm.i32
// CHECK-NEXT: %[[GEP0:.*]] = llvm.getelementptr %[[ADDR0]][%[[C0]], %[[C0]]] : (!llvm<"[16 x i8]*">, !llvm.i32, !llvm.i32) -> !llvm<"i8*">
// CHECK-NEXT: %[[ADDR1:.*]] = llvm.mlir.addressof @sig.sigI1 : !llvm<"[6 x i8]*">
// CHECK-NEXT: %[[C1:.*]] = llvm.mlir.constant(0 : i32) : !llvm.i32
// CHECK-NEXT: %[[GEP1:.*]] = llvm.getelementptr %[[ADDR1]][%[[C1]], %[[C1]]] : (!llvm<"[6 x i8]*">, !llvm.i32, !llvm.i32) -> !llvm<"i8*">
// CHECK-NEXT: %[[INIT0:.*]] = llvm.mlir.constant(0 : i1) : !llvm.i1
// CHECK-NEXT: %[[SIZE0:.*]] = llvm.mlir.constant(1 : i64) : !llvm.i64
// CHECK-NEXT: %[[MALL:.*]] = llvm.call @malloc(%[[SIZE0]]) : (!llvm.i64) -> !llvm<"i8*">
// CHECK-NEXT: %[[BC0:.*]] = llvm.bitcast %[[MALL]] : !llvm<"i8*"> to !llvm<"i1*">
// CHECK-NEXT: llvm.store %[[INIT0]], %[[BC0]] : !llvm<"i1*">
// CHECK-NEXT: %{{.*}} = llvm.call @alloc_signal(%[[STATE]], %[[GEP1]], %[[GEP0]], %[[MALL]], %[[SIZE0]]) : (!llvm<"i8*">, !llvm<"i8*">, !llvm<"i8*">, !llvm<"i8*">, !llvm.i64) -> !llvm.i32
// CHECK-NEXT: llvm.return

// CHECK-LABEL: @sig_conversions
// CHECK-SAME: %[[STATE:.*]]: !llvm<"i8*">
llhd.entity @sig_conversions () -> () {
    // CHECK-NEXT: %[[CI1:.*]] = llvm.mlir.constant(0 : i1) : !llvm.i1
    %init1 = llhd.const 0 : i1
    // CHECK-NEXT: %[[ADDR0:.*]] = llvm.mlir.addressof @entity.sig_conversions : !llvm<"[16 x i8]*">
    // CHECK-NEXT: %[[C0:.*]] = llvm.mlir.constant(0 : index) : !llvm.i64
    // CHECK-NEXT: %[[GEPENTITY:.*]] = llvm.getelementptr %[[ADDR0]][%[[C0]], %[[C0]]] : (!llvm<"[16 x i8]*">, !llvm.i64, !llvm.i64) -> !llvm<"i8*">
    // CHECK-NEXT: %[[ADDR1:.*]] = llvm.mlir.addressof @sig.sigI1 : !llvm<"[6 x i8]*">
    // CHECK-NEXT: %[[C1:.*]] = llvm.mlir.constant(0 : index) : !llvm.i64
    // CHECK-NEXT: %[[GEP_SIG:.*]] = llvm.getelementptr %[[ADDR1]][%[[C1]], %[[C1]]] : (!llvm<"[6 x i8]*">, !llvm.i64, !llvm.i64) -> !llvm<"i8*">
    // CHECK-NEXT: %[[SIG:.*]] = llvm.call @gather_signal(%[[STATE]], %[[GEP_SIG]], %[[GEPENTITY]]) : (!llvm<"i8*">, !llvm<"i8*">, !llvm<"i8*">) -> !llvm.i32
    %0 = llhd.sig "sigI1" %init1 : i1
    // CHECK-NEXT: %[[CALL:.*]] = llvm.call @probe_signal(%[[STATE]], %[[SIG]]) : (!llvm<"i8*">, !llvm.i32) -> !llvm<"i8*">
    // CHECK-NEXT: %[[BC:.*]] = llvm.bitcast %[[CALL]] : !llvm<"i8*"> to !llvm<"i1*">
    // CHECK-NEXT: %[[PRBD:.*]] = llvm.load %[[BC]] : !llvm<"i1*">
    %1 = llhd.prb %0 : !llhd.sig<i1>
    // CHECK-NEXT: %[[ARRSIZE:.*]] = llvm.mlir.constant(1 : i32) : !llvm.i32
    // CHECK-NEXT: %[[ALLOCA:.*]] = llvm.alloca %[[ARRSIZE]] x !llvm.i1 {alignment = 4 : i64}
    // CHECK-NEXT: llvm.store %[[PRBD]], %[[ALLOCA]] : !llvm<"i1*">
    // CHECK-NEXT: %[[BC1:.*]] = llvm.bitcast %[[ALLOCA]]
    // CHECK-NEXT: %[[SIZEC:.*]] = llvm.mlir.constant(1 : i64) : !llvm.i64
    // CHECK-NEXT: %[[TIME:.*]] = llvm.mlir.constant(1 : i32) : !llvm.i32
    // CHECK-NEXT: %[[DELTA:.*]] = llvm.mlir.constant(0 : i32) : !llvm.i32
    // CHECK-NEXT: %[[EPS:.*]] = llvm.mlir.constant(0 : i32) : !llvm.i32
    // CHECK-NEXT: %{{.*}} = llvm.call @drive_signal(%[[STATE]], %[[SIG]], %[[BC1]], %[[SIZEC]], %[[TIME]], %[[DELTA]], %[[EPS]])
    %t = llhd.const #llhd.time<1ns, 0d, 0e> : !llhd.time
    llhd.drv %0, %1 after %t : !llhd.sig<i1>

}
