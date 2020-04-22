//RUN: llhd-translate --llhd-to-verilog %s | FileCheck %s

// CHECK: module _empty;
llhd.entity @empty () -> () {
// CHECK-NEXT: endmodule
}

// CHECK-NEXT: module _onlyinput(input [63:0] _{{.*}});
llhd.entity @onlyinput (%arg0 : !llhd.sig<i64>) -> () {
// CHECK-NEXT: endmodule
}

// CHECK-NEXT: module _onlyoutput(output [63:0] _{{.*}});
llhd.entity @onlyoutput () -> (%out0 : !llhd.sig<i64>) {
// CHECK-NEXT: endmodule
}

// CHECK-NEXT: module _inputandoutput(input [63:0] _{{.*}}, input [31:0] {{.*}}, output [7:0] _{{.*}}, output [15:0] {{.*}});
llhd.entity @inputandoutput (%arg0 : !llhd.sig<i64>, %arg1 : !llhd.sig<i32>) -> (%out0 : !llhd.sig<i8>, %out1 : !llhd.sig<i16>) {
// CHECK-NEXT: endmodule
}
