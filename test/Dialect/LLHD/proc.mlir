// RUN: llhdc %s | llhdc | FileCheck %s

// no inputs and outputs
// CHECK: llhd.proc @generic_empty() -> () {
"llhd.proc"() ({
    // CHECK: llhd.halt
    // CHECK-NEXT: }
    "llhd.halt"() {} : () -> ()
}) {sym_name="generic_empty", ins=0, type=()->()} : () -> ()

// two inputs, one output
// CHECK-NEXT: llhd.proc @generic_inputandoutput(%{{.*}} : !llhd.sig<i64>, %{{.*}} : !llhd.sig<i64>) -> (%{{.*}} : !llhd.sig<i64>) {
"llhd.proc"() ({
    // CHECK-NEXT: llhd.halt
    // CHECK-NEXT: }
^body(%arg0 : !llhd.sig<i64>, %arg1 : !llhd.sig<i64>, %out0 : !llhd.sig<i64>):
    "llhd.halt"() {} : () -> ()
}) {sym_name="generic_inputandoutput", ins=2, type=(!llhd.sig<i64>, !llhd.sig<i64>, !llhd.sig<i64>)->()} : () -> ()

// zero inputs, one output
// CHECK-NEXT: llhd.proc @generic_output() -> (%{{.*}} : !llhd.sig<i64>) {
"llhd.proc"() ({
    // CHECK-NEXT: llhd.halt
    // CHECK-NEXT: }
^body(%0 : !llhd.sig<i64>):
    "llhd.halt"() {} : () -> ()
}) {sym_name="generic_output", ins=0, type=(!llhd.sig<i64>)->()} : () -> ()

// one input, zero outputs
// CHECK-NEXT: llhd.proc @generic_input(%{{.*}} : !llhd.sig<i64>) -> () {
"llhd.proc"() ({
    // CHECK-NEXT: llhd.halt
    // CHECK-NEXT: }
^body(%arg0 : !llhd.sig<i64>):
    "llhd.halt"() {} : () -> ()
}) {sym_name="generic_input", ins=1, type=(!llhd.sig<i64>)->()} : () -> ()


// no inputs and outputs
// CHECK: llhd.proc @empty() -> () {
llhd.proc @empty() -> () {
    // CHECK: llhd.halt
    // CHECK-NEXT: }
    llhd.halt
}

// two inputs, one output
// CHECK-NEXT: llhd.proc @inputandoutput(%arg0 : !llhd.sig<i64>, %arg1 : !llhd.sig<i1>) -> (%arg2 : !llhd.sig<i1>) {
llhd.proc @inputandoutput(!llhd.sig<i64>, !llhd.sig<i1>) -> (!llhd.sig<i1>) {
    // CHECK-NEXT:     llhd.halt
    // CHECK-NEXT: }
^bb0(%arg0 : !llhd.sig<i64>, %arg1 : !llhd.sig<i1>, %arg2 : !llhd.sig<i1>):
    llhd.halt
}

// zero inputs, one output
// CHECK-NEXT: llhd.proc @output() -> (%{{.*}} : !llhd.sig<i64>) {
llhd.proc @output() -> (%arg0 : !llhd.sig<i64>) {
    // CHECK-NEXT: llhd.halt
    // CHECK-NEXT: }
    llhd.halt
}

// one input, zero outputs
// CHECK-NEXT: llhd.proc @input(%{{.*}} : !llhd.sig<i64>) -> () {
llhd.proc @input(%arg0 : !llhd.sig<i64>) -> () {
    // CHECK-NEXT: llhd.halt
    // CHECK-NEXT: }
    llhd.halt
}
