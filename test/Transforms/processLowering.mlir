// RUN: llhdc %s -llhd-process-lowering -split-input-file -verify-diagnostics | FileCheck %s

// no inputs and outputs
// CHECK: llhd.entity @empty () -> () {
llhd.proc @empty() -> () {
    // CHECK-NEXT: }
    llhd.halt
}

// check that input and output signals are transferred correctly
// CHECK-NEXT: llhd.entity @inputAndOutput (%{{.*}} : !llhd.sig<i64>, %{{.*}} : !llhd.sig<i1>) -> (%{{.*}} : !llhd.sig<i1>) {
llhd.proc @inputAndOutput(%arg0 : !llhd.sig<i64>, %arg1 : !llhd.sig<i1>) -> (%arg2 : !llhd.sig<i1>) {
    // CHECK-NEXT: }
    llhd.halt
}

// check wait suspended process
// CHECK-NEXT: llhd.entity @simpleWait () -> () {
llhd.proc @simpleWait() -> () {
    // CHECK-NEXT: }
    br ^bb1
^bb1:
    llhd.wait ^bb1
}

// Check wait with observing probed signals
// CHECK-NEXT: llhd.entity @prbAndWait (%{{.*}} : !llhd.sig<i64>) -> () {
llhd.proc @prbAndWait(%arg0 : !llhd.sig<i64>) -> () {
    // CHECK-NEXT: %{{.*}} = llhd.prb
    // CHECK-NEXT: }
    br ^bb1
^bb1:
    %0 = llhd.prb %arg0 : !llhd.sig<i64>
    llhd.wait %arg0, ^bb1 : !llhd.sig<i64>
}

// Check wait with observing probed signals
// CHECK-NEXT: llhd.entity @prbAndWaitMoreObserved (%{{.*}} : !llhd.sig<i64>, %{{.*}} : !llhd.sig<i64>) -> () {
llhd.proc @prbAndWaitMoreObserved(%arg0 : !llhd.sig<i64>, %arg1 : !llhd.sig<i64>) -> () {
    // CHECK-NEXT: %{{.*}} = llhd.prb
    // CHECK-NEXT: }
    br ^bb1
^bb1:
    %0 = llhd.prb %arg0 : !llhd.sig<i64>
    llhd.wait %arg0, %arg1, ^bb1 : !llhd.sig<i64>, !llhd.sig<i64>
}

// -----

// Check wait with observing probed signals
llhd.proc @prbAndWaitNotObserved(%arg0 : !llhd.sig<i64>) -> () {
    br ^bb1
^bb1:
    %0 = llhd.prb %arg0 : !llhd.sig<i64>
    // expected-error @+1 {{Process-lowering: The wait terminator is required to have all probed signals as arguments!}}
    llhd.wait ^bb1
}

// -----

// Check that br is the only instruction in the entry block
// expected-error @+1 {{Process-lowering: The first block must contain the BranchOp terminator from the standard dialect only!}}
llhd.proc @multipleOpsInEntryBlock(%arg0 : !llhd.sig<i64>) -> () {
    %time = llhd.const #llhd.time<0ns, 0d, 0e> : !llhd.time
    %value = llhd.const 42 : i64
    llhd.drv %arg0, %value after %time : !llhd.sig<i64>
    br ^bb1
^bb1:
    llhd.wait ^bb1
}

// -----

// Check that block arguments for the second block are not allowed.
// expected-error @+1 {{Process-lowering: The second block (containing the llhd.wait) is not allowed to have arguments.}}
llhd.proc @blockArgumentsNotAllowed(%arg0 : !llhd.sig<i64>) -> () {
    br ^bb1(%arg0 : !llhd.sig<i64>)
^bb1(%a : !llhd.sig<i64>):
    llhd.wait ^bb1(%a : !llhd.sig<i64>)
}

// -----

// Check that the entry block is terminated by a br terminator.
// expected-error @+1 {{Process-lowering: The first block has to be terminated by a BranchOp from the standard dialect.}}
llhd.proc @entryBlockMustHaveBrTerminator() -> () {
    llhd.wait ^bb1
^bb1:
    llhd.wait ^bb1
}

// -----

// Check that there is no optional time operand in the wait terminator.
llhd.proc @noOptionalTime() -> () {
    br ^bb1
^bb1:
    %time = llhd.const #llhd.time<0ns, 0d, 0e> : !llhd.time
    // expected-error @+1 {{Process-lowering: llhd.wait terminators with optional time argument cannot be lowered to structural LLHD.}}
    llhd.wait for %time, ^bb1 : !llhd.time
}

// -----

// Check that if there are two blocks, the second one is terminated by a wait terminator.
// expected-error @+1 {{Process-lowering: The second block must be terminated by a WaitOp from the LLHD dialect.}}
llhd.proc @secondBlockTerminatedByWait() -> () {
    br ^bb1
^bb1:
    llhd.halt
}

// -----

// Check that there are not more than two blocks.
// expected-error @+1 {{Process-lowering only supports processes with either one basic block terminated by a llhd.halt operation or two basic blocks where the first one contains a std.br terminator only and the second one is terminated by a llhd.wait operation.}}
llhd.proc @moreThanTwoBlocksNotAllowed() -> () {
    br ^bb1
^bb1:
    br ^bb2
^bb2:
    llhd.wait ^bb1
}
