// RUN: llhdc %s -mlir-print-op-generic -split-input-file -verify-diagnostics | llhdc | llhdc | FileCheck %s

// CHECK-LABEL: @inss_integers
// CHECK-SAME: %[[CI1:.*]]: i1
// CHECK-SAME: %[[CI32:.*]]: i32
func @inss_integers(%cI1 : i1, %cI32 : i32) {
    // CHECK-NEXT: %{{.*}} = llhd.inss %[[CI1]], %[[CI1]], 0 : i1 into i1
    %0 = llhd.inss %cI1, %cI1, 0 : i1 into i1
    // CHECK-NEXT: %{{.*}} = llhd.inss %[[CI1]], %[[CI32]], 31 : i1 into i32
    %1 = llhd.inss %cI1, %cI32, 31 : i1 into i32

    return
}

// CHECK-LABEL: @inss_vectors
// CHECK-SAME: %[[VEC2:.*]]: vector<2xi1>
// CHECK-SAME: %[[VEC5:.*]]: vector<5xi1>
func @inss_vectors(%vec2 : vector<2xi1>, %vec5 : vector<5xi1>) -> () {
    // CHECK-NEXT: %{{.*}} = llhd.inss %[[VEC2]], %[[VEC5]], 3 : vector<2xi1> into vector<5xi1>
    %0 = llhd.inss %vec2, %vec5, 3 : vector<2xi1> into vector<5xi1>
    // CHECK-NEXT: %{{.*}} = llhd.inss %[[VEC2]], %[[VEC2]], 0 : vector<2xi1> into vector<2xi1>
    %1 = llhd.inss %vec2, %vec2, 0 : vector<2xi1> into vector<2xi1>

    return
}

// CHECK-LABEL: @insf_tuples
// CHECK-SAME: %[[TUP:.*]]: tuple<i1, i8>,
// CHECK-SAME: %[[I1:.*]]: i1,
// CHECK-SAME: %[[I8:.*]]: i8
func @insf_tuples(%tup : tuple<i1, i8>, %i1 : i1, %i8 : i8) {
    // CHECK-NEXT: %{{.*}} = llhd.insf %[[I1]], %[[TUP]], 0 : i1 into tuple<i1, i8>
    %0 = llhd.insf %i1, %tup, 0 : i1 into tuple<i1, i8>
    // CHECK-NEXT: %{{.*}} = llhd.insf %[[I8]], %[[TUP]], 1 : i8 into tuple<i1, i8>
    %1 = llhd.insf %i8, %tup, 1 : i8 into tuple<i1, i8>

    return
}

// CHECK-LABEL: @insf_vectors
// CHECK-SAME: %[[V1:.*]]: vector<4xi1>,
// CHECK-SAME: %[[V8:.*]]: vector<4xi8>,
// CHECK-SAME: %[[I1:.*]]: i1,
// CHECK-SAME: %[[I8:.*]]: i8
func @insf_vectors(%v1 : vector<4xi1>, %v8 : vector<4xi8>, %i1 : i1, %i8 : i8) {
    // CHECK-NEXT: %{{.*}} = llhd.insf %[[I1]], %[[V1]], 0 : i1 into vector<4xi1>
    %0 = llhd.insf %i1, %v1, 0 : i1 into vector<4xi1>
    // CHECK-NEXT: %{{.*}} = llhd.insf %[[I8]], %[[V8]], 2 : i8 into vector<4xi8>
    %1 = llhd.insf %i8, %v8, 2 : i8 into vector<4xi8>

    return
}

// -----

func @illegal_kind(%c : i32, %vec : vector<2xi32>) {
    // expected-error @+1 {{failed to verify that 'target' and 'slice' have to be both either signless integers or vectors with the same element type}}
    %0 = llhd.inss %c, %vec, 0 : i32 into vector<2xi32>

    return
}

// -----

func @illegal_elemental_type(%slice : vector<1xi1>, %vec : vector<2xi32>) {
    // expected-error @+1 {{failed to verify that 'target' and 'slice' have to be both either signless integers or vectors with the same element type}}
    %0 = llhd.inss %slice, %vec, 0 : vector<1xi1> into vector<2xi32>

    return
}

// -----

func @inss_illegal_start_index_int(%slice : i16, %c : i32) {
     // expected-error @+1 {{failed to verify that 'start' + size of the 'slice' have to be smaller or equal to the 'target' size}}
    %0 = llhd.inss %slice, %c, 20 : i16 into i32

     return
 }

// -----

func @inss_illegal_start_index_vector(%slice : vector<2xi1>, %vec : vector<3xi1>) {
     // expected-error @+1 {{failed to verify that 'start' + size of the 'slice' have to be smaller or equal to the 'target' size}}
    %0 = llhd.inss %slice, %vec, 2 : vector<2xi1> into vector<3xi1>

     return
 }

// -----

func @insf_index_out_of_bounds(%e : i1, %vec : vector<3xi1>) {
    // expected-error @+1 {{failed to verify that 'index' has to be smaller than the 'target' size}}
    %0 = llhd.insf %e, %vec, 3 : i1 into vector<3xi1>

    return
}

// -----

func @insf_type_mismatch_vector(%e : i2, %vec : vector<3xi1>) {
    // expected-error @+1 {{failed to verify that 'element' type has to match type at 'index' of 'target'}}
    %0 = llhd.insf %e, %vec, 0 : i2 into vector<3xi1>

    return
}

// -----

func @insf_type_mismatch_tuple(%e : i2, %tup : tuple<i2, i1, i2>) {
    // expected-error @+1 {{failed to verify that 'element' type has to match type at 'index' of 'target'}}
    %0 = llhd.insf %e, %tup, 1 : i2 into tuple<i2, i1, i2>

    return
}
