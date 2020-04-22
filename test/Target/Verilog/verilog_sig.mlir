//RUN: llhd-translate --llhd-to-verilog -split-input-file -verify-diagnostics %s | FileCheck %s

// CHECK-LABEL: _check_sig
llhd.entity @check_sig () -> () {
    // CHECK-NEXT: wire _{{.*}} = 1'd1;
    %0 = llhd.const 1 : i1
    // CHECK-NEXT: wire [63:0] _{{.*}} = 64'd256;
    %1 = llhd.const 256 : i64
    %2 = llhd.const #llhd.time<1ns, 0d, 0e> : !llhd.time
    // CHECK-NEXT: var _{{.*}} = _{{.*}};
    %3 = llhd.sig %0 : i1 -> !llhd.sig<i1>
    // CHECK-NEXT: var [63:0] _{{.*}} = _{{.*}};
    %4 = llhd.sig %1 : i64 -> !llhd.sig<i64>
    %5 = llhd.prb %3 : !llhd.sig<i1> -> i1
    // CHECK-NEXT: assign _{{.*}} = (#1ns) _{{.*}};
    llhd.drv %3, %0, %2 : !llhd.sig<i1>, i1, !llhd.time
    %6 = llhd.const #llhd.time<0ns, 1d, 0e> : !llhd.time
    // CHECK-NEXT: assign _{{.*}} = (#0ns) _{{.*}};
    llhd.drv %3, %0, %6 : !llhd.sig<i1>, i1, !llhd.time
}

// -----

llhd.entity @check_invalid_drv_time () -> () {
    %0 = llhd.const 1 : i1
    %1 = llhd.sig %0 : i1 -> !llhd.sig<i1>
    // expected-error @+2 {{Not possible to translate a time attribute with 0 real time and non-1 delta.}}
    // expected-error @+1 {{Operation not supported!}}
    %2 = llhd.const #llhd.time<0ns, 0d, 0e> : !llhd.time
    llhd.drv %1, %0, %2 : !llhd.sig<i1>, i1, !llhd.time
}
