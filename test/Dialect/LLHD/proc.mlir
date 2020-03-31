// RUN: llhdc %s | llhdc | FileCheck %s

// no inputs and outputs, empty basic block
// CHECK-LABEL: empty
llhd.proc @empty() -> () {
    // CHECK: llhd.halt
    // CHECK-NEXT: }
    //
    // GENERIC: "llhd.halt"()
    // GENERIC-NEXT: }
    llhd.halt
}

// two inputs, one output, one basic block
// CHECK-NEXT: llhd.proc @foo(%[[ARG0:.*]] : !llhd.sig<i64>, %[[ARG1:.*]] : !llhd.sig<i64>) -> (%[[OUT0:.*]] : !llhd.sig<i64>) {
"llhd.proc"() ({
// CHECK-NEXT: %[[C0:.*]] = llhd.const 1
// CHECK-NEXT: llhd.drv %[[OUT0:.*]], %[[C0:.*]] 
// CHECK-NEXT: %[[P0:.*]] = llhd.prb %[[ARG0:.*]]
// CHECK-NEXT: llhd.halt
// CHECK-NEXT: }
^body(%arg0 : !llhd.sig<i64>, %arg1 : !llhd.sig<i64>, %out0 : !llhd.sig<i64>):
    %0 = llhd.const 1 : i64
    llhd.drv %out0, %0 : !llhd.sig<i64>
    %1 = llhd.prb %arg0 : !llhd.sig<i64> -> i64
    "llhd.halt"() {} : () -> ()
}) {sym_name="foo", ins=2, type=(!llhd.sig<i64>, !llhd.sig<i64>, !llhd.sig<i64>)->()} : () -> ()

// zero inputs, one output, empty basic block
// CHECK-NEXT: llhd.proc @bar() -> (%{{.*}} : !llhd.sig<i64>) {
"llhd.proc"() ({
// CHECK-NEXT: llhd.halt
// CHECK-NEXT: }
^body(%0 : !llhd.sig<i64>):
    "llhd.halt"() {} : () -> ()
}) {sym_name="bar", ins=0, type=(!llhd.sig<i64>)->()} : () -> ()

// one input, zero outputs, empty basic block
// CHECK-NEXT: llhd.proc @baz(%{{.*}} : !llhd.sig<i64>) -> () {
"llhd.proc"() ({
// CHECK-NEXT: llhd.halt
// CHECK-NEXT: }
^body(%arg0 : !llhd.sig<i64>):
    "llhd.halt"() {} : () -> ()
}) {sym_name="baz", ins=1, type=(!llhd.sig<i64>)->()} : () -> ()

// two inputs, one output, empty basic block, custom syntax
// CHECK-NEXT: llhd.proc @test(%arg0 : !llhd.sig<i64>, %arg1 : !llhd.sig<i1>) -> (%arg2 : !llhd.sig<i1>) {
llhd.proc @test(!llhd.sig<i64>, !llhd.sig<i1>) -> (!llhd.sig<i1>) {
// CHECK-NEXT:     %0 = llhd.const 5 : i64
// CHECK-NEXT:     llhd.halt
// CHECK-NEXT: }
^bb0(%arg0 : !llhd.sig<i64>, %arg1 : !llhd.sig<i1>, %arg2 : !llhd.sig<i1>):
    %var1 = llhd.const 5 : i64
    llhd.halt
}

// two inputs, one output, two basic blocks (one with two arguments), custom syntax
// CHECK-NEXT: llhd.proc @test2(%arg0 : !llhd.sig<i64>, %arg1 : !llhd.sig<i1>) -> (%arg2 : !llhd.sig<i1>) {
llhd.proc @test2(%arg0 : !llhd.sig<i64>, %arg1 : !llhd.sig<i1>) -> (%arg2 : !llhd.sig<i1>) {
// CHECK-NEXT: %0 = llhd.const 5 : i64
// CHECK-NEXT: %1 = llhd.const 3 : i64
// CHECK-NEXT: llhd.wait %arg0 [^bb1(%0, %1 : i64, i64)] : !llhd.sig<i64>
// CHECK-NEXT: ^bb1(%2: i64, %3: i64):
    %var1 = llhd.const 5 : i64
    %tmp = llhd.const 3 : i64
    llhd.wait %arg0 [^bb1(%var1, %tmp : i64, i64)] : !llhd.sig<i64>
^bb1(%a : i64, %b : i64):
// CHECK-NEXT: %4 = llhd.add(%2, %3) : (i64, i64) -> i64
// CHECK-NEXT: llhd.wait %arg0, %arg1 [^bb1(%0, %4 : i64, i64)] : !llhd.sig<i64>, !llhd.sig<i1>
// CHECK-NEXT: }
    %var3 = llhd.add(%a, %b) : (i64, i64) -> i64
    llhd.wait %arg0, %arg1 [^bb1(%var1, %var3 : i64, i64)] : !llhd.sig<i64>, !llhd.sig<i1>
}

// two inputs, zero output, two basic blocks (one with zero arguments), custom syntax
// CHECK-NEXT: llhd.proc @test3(%arg0 : !llhd.sig<i64>, %arg1 : !llhd.sig<i1>) -> () {
llhd.proc @test3(%arg0 : !llhd.sig<i64>, %arg1 : !llhd.sig<i1>) -> () {
// CHECK-NEXT: llhd.wait %arg0 [^bb1] : !llhd.sig<i64>
// CHECK-NEXT: ^bb1:
    llhd.wait %arg0 [^bb1] : !llhd.sig<i64>
^bb1:
// CHECK-NEXT: llhd.wait %arg0, %arg1 [^bb1] : !llhd.sig<i64>, !llhd.sig<i1>
// CHECK-NEXT: }
    llhd.wait %arg0, %arg1 [^bb1] : !llhd.sig<i64>, !llhd.sig<i1>
}