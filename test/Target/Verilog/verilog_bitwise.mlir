//RUN: llhd-translate --llhd-to-verilog %s | FileCheck %s

// CHECK-LABEL: _check_bitwise
llhd.entity @check_bitwise() -> () {
    // CHECK-NEXT: wire [63:0] _{{.*}} = 64'd42;
    %a = llhd.const 42 : i64
    // CHECK-NEXT: wire [63:0] _{{.*}} = ~_{{.*}};
    %0 = llhd.not %a : i64
    // CHECK-NEXT: wire [63:0] _{{.*}} = _{{.*}} & _{{.*}};
    %1 = llhd.and %a, %a : i64
    // CHECK-NEXT: wire [63:0] _{{.*}} = _{{.*}} | _{{.*}};
    %2 = llhd.or %a, %a : i64
    // CHECK-NEXT: wire [63:0] _{{.*}} = _{{.*}} ^ _{{.*}};
    %3 = llhd.xor %a, %a : i64

    // CHECK-NEXT: wire [4:0] _{{.*}} = 5'd0;
    %hidden = llhd.const 0 : i5
    // CHECK-NEXT: wire [1:0] _{{.*}} = 2'd3;
    %amt = llhd.const 3 : i2

    // CHECK-NEXT: wire [68:0] _{{.*}} = {_{{.*}}, _{{.*}}};
    // CHECK-NEXT: wire [68:0] _{{.*}} = _{{.*}} << {{.*}};
    // CHECK-NEXT: wire [63:0] _{{.*}} = _{{.*}}[68:5];
    %4 = llhd.shl %a, %hidden, %amt : (i64, i5, i2) -> i64
    // CHECK-NEXT: wire [68:0] _{{.*}} = {_{{.*}}, _{{.*}}};
    // CHECK-NEXT: wire [68:0] _{{.*}} = _{{.*}} << {{.*}};
    // CHECK-NEXT: wire [63:0] _{{.*}} = _{{.*}}[63:0];
    %5 = llhd.shr %a, %hidden, %amt : (i64, i5, i2) -> i64
}
