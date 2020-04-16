//RUN: llhdc %s --convert-llhd-to-llvm --split-input-file | FileCheck %s

func @convert_sig(%state : !llvm<"i8*">, %init1 : i1) {
    // CHECK: %[[I1:.*]] = llvm.mlir.constant(0 : i32) : !llvm.i32
    // CHECK-NEXT: %{{.*}} = llvm.call @alloc_signal(%{{.*}}, %{{.*}}, %[[I1:.*]]) : (!llvm<"i8*">, !llvm.i32, !llvm.i1) -> !llvm.i32
    %0 = llhd.sig %init1 : i1 -> !llhd.sig<i1>

    return
}

func @convert_prb(%state : !llvm<"i8*">, %init : i1) {
    %0 = llhd.sig %init : i1 -> !llhd.sig<i1>
    // CHECK: %[[C1:.*]] = llvm.call @probe_signal(%{{.*}}, %{{.*}}) : (!llvm<"i8*">, !llvm.i32) -> !llvm<"i8*">
    // CHECK-NEXT: %[[B1:.*]] = llvm.bitcast %[[C1:.*]] : !llvm<"i8*"> to !llvm<"i1*">
    // CHECK-NEXT: %{{.*}} = llvm.load %[[B1:.*]] : !llvm<"i1*">
    %1 = llhd.prb %0 : !llhd.sig<i1> -> i1

    return
}

func @convert_drv(%state : !llvm<"i8*">, %init : i1) {
    %0 = llhd.sig %init : i1 -> !llhd.sig<i1>
    %1 = llhd.const #llhd.time<1ns, 0d, 0e> : !llhd.time
    // CHECK: %[[T1:.*]] = llvm.mlir.constant(1 : i32) : !llvm.i32
    // CHECK-NEXT: %{{.*}} = llvm.call @drive_signal(%{{.*}}, %{{.*}}, %{{.*}}, %[[T1:.*]]) : (!llvm<"i8*">, !llvm.i32, !llvm.i1, !llvm.i32) -> !llvm.void
    llhd.drv %0, %init, %1 : !llhd.sig<i1>, i1, !llhd.time

    return
}
