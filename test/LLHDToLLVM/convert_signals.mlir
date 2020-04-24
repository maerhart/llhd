//RUN: llhdc %s --convert-llhd-to-llvm --split-input-file | FileCheck %s

llhd.entity @sig_conversions () -> () {
    %init1 = llhd.const 0 : i1 
    // CHECK: %[[I1:.*]] = llvm.mlir.constant(0 : i32) : !llvm.i32
    // CHECK-NEXT: %{{.*}} = llvm.call @alloc_signal(%{{.*}}, %{{.*}}, %[[I1:.*]]) : (!llvm<"i8*">, !llvm.i32, !llvm.i1) -> !llvm.i32
    %0 = llhd.sig %init1 : i1 -> !llhd.sig<i1>
    // CHECK-NEXT: %[[C1:.*]] = llvm.call @probe_signal(%{{.*}}, %{{.*}}) : (!llvm<"i8*">, !llvm.i32) -> !llvm<"i8*">
    // CHECK-NEXT: %[[B1:.*]] = llvm.bitcast %[[C1:.*]] : !llvm<"i8*"> to !llvm<"i1*">
    // CHECK-NEXT: %{{.*}} = llvm.load %[[B1:.*]] : !llvm<"i1*">
    %1 = llhd.prb %0 : !llhd.sig<i1> -> i1
    // CHECK-NEXT: %[[T1:.*]] = llvm.mlir.constant(1 : i32) : !llvm.i32
    // CHECK-NEXT: %{{.*}} = llvm.call @drive_signal(%{{.*}}, %{{.*}}, %{{.*}}, %[[T1:.*]]) : (!llvm<"i8*">, !llvm.i32, !llvm.i1, !llvm.i32) -> !llvm.void
    %t = llhd.const #llhd.time<1ns, 0d, 0e> : !llhd.time
    llhd.drv %0, %init1, %t : !llhd.sig<i1>, i1, !llhd.time
}
