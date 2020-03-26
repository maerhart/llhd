// RUN: llhdc %s -verify-diagnostics | FileCheck %s

func @test_time_type() {
    // CHECK: %[[CONST:.*]] = "time_result"() : () -> !llhd.time 
    %0 = "time_result"() : () -> !llhd.time
    // CHECK-NEXT: "time_const_arg"(%[[CONST:.*]]) : (!llhd.time) -> ()
    "time_const_arg"(%0) : (!llhd.time) -> ()
    return
}

func @test_time_attr() {
    "time_attr"() {
        // CHECK: time0 = #llhd.time<1ns>
        time0 = #llhd.time<1ns>,
        // CHECK-SAME: time1 = #llhd.time<1ns, 2d>
        time1 = #llhd.time<1ns, 2d>,
        // CHECK-SAME: time2 = #llhd.time<1ns, 2d, 3e>
        time2 = #llhd.time<1ns, 2d, 3e>,
        // CHECK-SAME: time3 = #llhd.time<10ns, 5d>
        time3 = #llhd.time<10ns, 5d, 0e>,
        //CHECK-SAME: time4 = #llhd.time<256ns>
        time4 = #llhd.time<256ns, 0d, 0e>
    } : () -> ()
}
