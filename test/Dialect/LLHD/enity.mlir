// RUN: llhdc %s | llhdc | FileCheck %s


// check inputs and outputs, usage
// CHECK: llhd.entity @foo (%[[ARG0:.*]] : !llhd.sig<i64>, %[[ARG1:.*]] : !llhd.sig<i64>) -> (%[[OUT0:.*]] : !llhd.sig<i64>) {
"llhd.entity"() ({
// CHECK-NEXT: %[[C0:.*]] = llhd.const 1
// CHECK-NEXT: %[[P0:.*]] = llhd.prb %[[ARG0:.*]]
// CHECK-NEXT: }
^body(%arg0 : !llhd.sig<i64>, %arg1 : !llhd.sig<i64>, %out0 : !llhd.sig<i64>):
    %0 = llhd.const 1 : i64
    %1 = llhd.prb %arg0 : !llhd.sig<i64> -> i64
    "llhd.terminator"() {} : () -> ()
}) {sym_name="foo", ins=2, type=(!llhd.sig<i64>, !llhd.sig<i64>, !llhd.sig<i64>)->()} : () -> ()

// check 0 inputs, empty body
// CHECK-NEXT: llhd.entity @bar () -> (%{{.*}} : !llhd.sig<i64>) {
"llhd.entity"() ({
// CHECK-NEXT: }
^body(%0 : !llhd.sig<i64>):
    "llhd.terminator"() {} : () -> ()
}) {sym_name="bar", ins=0, type=(!llhd.sig<i64>)->()} : () -> ()

// check 0 outputs, empty body
// CHECK-NEXT: llhd.entity @baz (%{{.*}} : !llhd.sig<i64>) -> () {
"llhd.entity"() ({
// CHECK-NEXT: }
^body(%arg0 : !llhd.sig<i64>):
    "llhd.terminator"() {} : () -> ()
}) {sym_name="baz", ins=1, type=(!llhd.sig<i64>)->()} : () -> ()

//check 0 arguments, empty body
// CHECK-NEXT: llhd.entity @out_of_names () -> () {
"llhd.entity"() ({
// CHECK-NEXT : }
^body:
    "llhd.terminator"() {} : () -> ()
}) {sym_name="out_of_names", ins=0, type=()->()} : () -> ()