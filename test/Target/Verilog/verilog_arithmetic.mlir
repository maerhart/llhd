//RUN: llhd-translate --llhd-to-verilog %s | FileCheck %s

// CHECK-LABEL: _check_arithmetic
llhd.entity @check_arithmetic() -> () {
    // CHECK-NEXT: wire [63:0] _{{.*}} = 64'd42;
    %a = llhd.const 42 : i64
    // CHECK-NEXT: wire [63:0] _{{.*}} = -_{{.*}};
    %0 = llhd.neg %a : i64
    // CHECK-NEXT: wire [63:0] _{{.*}} = _{{.*}} + _{{.*}};
    %1 = addi %a, %a : i64
    // CHECK-NEXT: wire [63:0] _{{.*}} = _{{.*}} - _{{.*}};
    %2 = subi %a, %a : i64
    // CHECK-NEXT: wire [63:0] _{{.*}} = _{{.*}} * _{{.*}};
    %3 = muli %a, %a : i64
    // CHECK-NEXT: wire [63:0] _{{.*}} = _{{.*}} / _{{.*}};
    %4 = divi_unsigned %a, %a : i64
    // CHECK-NEXT: wire [63:0] _{{.*}} = $signed(_{{.*}}) / $signed(_{{.*}});
    %5 = divi_signed %a, %a : i64
    // CHECK-NEXT: wire [63:0] _{{.*}} = _{{.*}} % _{{.*}};
    %6 = remi_unsigned %a, %a : i64
    // CHECK-NEXT: wire [63:0] _{{.*}} = $signed(_{{.*}}) % $signed(_{{.*}});
    %7 = remi_signed %a, %a : i64
}
