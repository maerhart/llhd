//RUN: llhdc %s -split-input-file -verify-diagnostics | llhdc | FileCheck %s

// Testing Objectives:
// * inst can only be used in entities
// * inst must always refer to a valid proc or entity (match symbol name, input and output operands)
// * syntax: no inputs and outputs, one input zero outputs, zero inputs one output, multiple inputs and outputs
// * check that number of inputs and number of outputs are verified separately


// CHECK-LABEL: @empty_entity
llhd.entity @empty_entity() -> () {}

// CHECK-LABEL: @one_input_entity
llhd.entity @one_input_entity(%arg : !llhd.sig<i32>) -> () {}

// CHECK-LABEL: @one_output_entity
llhd.entity @one_output_entity() -> (%arg : !llhd.sig<i32>) {}

// CHECK-LABEL: @entity
llhd.entity @entity(%arg0 : !llhd.sig<i32>, %arg1 : !llhd.sig<i16>) -> (%out0 : !llhd.sig<i8>, %out1 : !llhd.sig<i4>) {}

// CHECK-LABEL: @empty_proc
llhd.proc @empty_proc() -> () {
    llhd.halt
}

// CHECK-LABEL: @one_input_proc
llhd.proc @one_input_proc(%arg : !llhd.sig<i32>) -> () {
    llhd.halt
}

// CHECK-LABEL: @one_output_proc
llhd.proc @one_output_proc() -> (%arg : !llhd.sig<i32>) {
    llhd.halt
}

// CHECK-LABEL: @proc
llhd.proc @proc(%arg0 : !llhd.sig<i32>, %arg1 : !llhd.sig<i16>) -> (%out0 : !llhd.sig<i8>, %out1 : !llhd.sig<i4>) {
    llhd.halt
}

// CHECK: llhd.entity @caller (%[[ARG0:.*]] : !llhd.sig<i32>, %[[ARG1:.*]] : !llhd.sig<i16>) -> (%[[OUT0:.*]] : !llhd.sig<i8>, %[[OUT1:.*]] : !llhd.sig<i4>) {
llhd.entity @caller(%arg0 : !llhd.sig<i32>, %arg1 : !llhd.sig<i16>) -> (%out0 : !llhd.sig<i8>, %out1 : !llhd.sig<i4>) {
    // CHECK-NEXT: llhd.inst @empty_entity() -> () : () -> ()
    "llhd.inst"() {callee=@empty_entity, operand_segment_sizes=dense<[0,0]> : vector<2xi32>} : () -> ()
    // CHECK-NEXT: llhd.inst @empty_proc() -> () : () -> ()
    "llhd.inst"() {callee=@empty_proc, operand_segment_sizes=dense<[0,0]> : vector<2xi32>} : () -> ()
    // CHECK-NEXT: llhd.inst @one_input_entity(%[[ARG0]]) -> () : (!llhd.sig<i32>) -> ()
    "llhd.inst"(%arg0) {callee=@one_input_entity, operand_segment_sizes=dense<[1,0]> : vector<2xi32>} : (!llhd.sig<i32>) -> ()
    // CHECK-NEXT: llhd.inst @one_input_proc(%[[ARG0]]) -> () : (!llhd.sig<i32>) -> ()
    "llhd.inst"(%arg0) {callee=@one_input_proc, operand_segment_sizes=dense<[1,0]> : vector<2xi32>} : (!llhd.sig<i32>) -> ()
    // CHECK-NEXT: llhd.inst @one_output_entity() -> (%[[ARG0]]) : () -> !llhd.sig<i32>
    "llhd.inst"(%arg0) {callee=@one_output_entity, operand_segment_sizes=dense<[0,1]> : vector<2xi32>} : (!llhd.sig<i32>) -> ()
    // CHECK-NEXT: llhd.inst @one_output_proc() -> (%[[ARG0]]) : () -> !llhd.sig<i32>
    "llhd.inst"(%arg0) {callee=@one_output_proc, operand_segment_sizes=dense<[0,1]> : vector<2xi32>} : (!llhd.sig<i32>) -> ()
    // CHECK-NEXT: llhd.inst @entity(%[[ARG0]], %[[ARG1]]) -> (%[[OUT0]], %[[OUT1]]) : (!llhd.sig<i32>, !llhd.sig<i16>) -> (!llhd.sig<i8>, !llhd.sig<i4>)
    "llhd.inst"(%arg0, %arg1, %out0, %out1) {callee=@entity, operand_segment_sizes=dense<[2,2]> : vector<2xi32>} : (!llhd.sig<i32>, !llhd.sig<i16>, !llhd.sig<i8>, !llhd.sig<i4>) -> ()
    // CHECK-NEXT: llhd.inst @proc(%[[ARG0]], %[[ARG1]]) -> (%[[OUT0]], %[[OUT1]]) : (!llhd.sig<i32>, !llhd.sig<i16>) -> (!llhd.sig<i8>, !llhd.sig<i4>)
    "llhd.inst"(%arg0, %arg1, %out0, %out1) {callee=@proc, operand_segment_sizes=dense<[2,2]> : vector<2xi32>} : (!llhd.sig<i32>, !llhd.sig<i16>, !llhd.sig<i8>, !llhd.sig<i4>) -> ()
    // CHECK-NEXT: }
}

// -----

llhd.proc @empty_proc() -> () {
    llhd.halt
}

llhd.proc @fail() -> () {
    // expected-error @+1 {{expects parent op 'llhd.entity'}}
    llhd.inst @empty_proc() -> () : () -> ()
    llhd.halt
}

// -----

llhd.entity @operand_count_mismatch(%arg : !llhd.sig<i32>) -> () {}

llhd.entity @caller(%arg : !llhd.sig<i32>) -> () {
    // expected-error @+1 {{incorrect number of inputs for entity instantiation}}
    llhd.inst @operand_count_mismatch() -> (%arg) : () -> (!llhd.sig<i32>)
}

// -----

llhd.entity @caller() -> () {
    // expected-error @+1 {{does not reference a valid proc or entity}}
    llhd.inst @does_not_exist() -> () : () -> ()
}