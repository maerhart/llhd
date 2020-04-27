//RUN: llhdc %s --convert-llhd-to-llvm --split-input-file | FileCheck %s

llhd.entity @sig_conversions () -> () {
    %init1 = llhd.const 0 : i1
    // CHECK-NEXT: %[[INDEX:.*]] = llvm.mlir.constant(0 : i32) : !llvm.i32
    // CHECK-NEXT: %[[SIG:.*]] = llvm.call @alloc_signal(%[[STATE]], %[[INDEX]], %[[CI1]]) : (!llvm<"i8*">, !llvm.i32, !llvm.i1) -> !llvm.i32
    %0 = llhd.sig "sigI1" %init1 : i1 -> !llhd.sig<i1>
    // CHECK-NEXT: %[[CALL:.*]] = llvm.call @probe_signal(%[[STATE]], %[[SIG]]) : (!llvm<"i8*">, !llvm.i32) -> !llvm<"i8*">
    // CHECK-NEXT: %[[BC:.*]] = llvm.bitcast %[[CALL]] : !llvm<"i8*"> to !llvm<"i1*">
    // CHECK-NEXT: %[[PRBD:.*]] = llvm.load %[[BC]] : !llvm<"i1*">
    %1 = llhd.prb %0 : !llhd.sig<i1> -> i1
    // CHECK-NEXT: %[[T1:.*]] = llvm.mlir.constant(1 : i32) : !llvm.i32
    // CHECK-NEXT: %{{.*}} = llvm.call @drive_signal(%{{.*}}, %{{.*}}, %{{.*}}, %[[T1:.*]]) : (!llvm<"i8*">, !llvm.i32, !llvm.i1, !llvm.i32) -> !llvm.void
    %t = llhd.const #llhd.time<1ns, 0d, 0e> : !llhd.time
    llhd.drv %0, %init1, %t : !llhd.sig<i1>, i1, !llhd.time
}
