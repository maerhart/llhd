// RUN: llhdc %s | llhdc | FileCheck %s

// check the construction of a new signal with the sig instruction
llhd.entity @check_sig_inst () -> () {
    // CHECK: %[[BOOL:.*]] = llhd.const
    %0 = llhd.const 1 : i1
    // CHECK-NEXT: %[[BSIG:.*]] = llhd.sig %[[BOOL:.*]] : i1 -> !llhd.sig<i1>
    %1 = "llhd.sig"(%0) {} : (i1) -> !llhd.sig<i1>
    // CHECK-NEXT: %[[INT:.*]] = llhd.const
    %2 = llhd.const 256 : i64
    // CHECK-NEXT: %[[ISIG:.*]] = llhd.sig %[[INT:.*]] : i64 -> !llhd.sig<i64>
    %3 = "llhd.sig"(%2) {} : (i64) -> !llhd.sig<i64>
}

func @check_prb(%sigI1 : !llhd.sig<i1>, %sigI64 : !llhd.sig<i64>) {
    // CHECK: %{{.*}} = llhd.prb %{{.*}} : !llhd.sig<i1> -> i1
    %0 = "llhd.prb"(%sigI1) {} : (!llhd.sig<i1>) -> i1
    // CHECK-NEXT: %{{.*}} = llhd.prb %{{.*}} : !llhd.sig<i64> -> i64
    %1 = "llhd.prb"(%sigI64) {} : (!llhd.sig<i64>) -> i64

    return
}

func @check_drv(%sigI1 : !llhd.sig<i1>, %sigI64 : !llhd.sig<i64>, %cI1 : i1, %cI64 : i64, %t : !llhd.time) {
    // CHECK: llhd.drv %{{.*}}, %{{.*}}, %{{.*}} : !llhd.sig<i1>, i1, !llhd.time
    llhd.drv %sigI1, %cI1, %t : !llhd.sig<i1>, i1, !llhd.time
    // CHECK-NEXT: llhd.drv %{{.*}}, %{{.*}}, %{{.*}} : !llhd.sig<i64>, i64, !llhd.time
    llhd.drv %sigI64, %cI64, %t : !llhd.sig<i64>, i64, !llhd.time

    return
}
