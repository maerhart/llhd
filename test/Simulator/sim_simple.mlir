// RUN: llhd-sim %s -n 10 | FileCheck %s

// CHECK: 0ns  Foo/toggle  0
// CHECK-NEXT: 1ns  Foo/toggle  1
// CHECK-NEXT: 2ns  Foo/toggle  0
// CHECK-NEXT: 3ns  Foo/toggle  1
// CHECK-NEXT: 4ns  Foo/toggle  0
// CHECK-NEXT: 5ns  Foo/toggle  1
// CHECK-NEXT: 6ns  Foo/toggle  0
// CHECK-NEXT: 7ns  Foo/toggle  1
// CHECK-NEXT: 8ns  Foo/toggle  0
// CHECK-NEXT: 9ns  Foo/toggle  1
llhd.entity @Foo () -> () {
    %0 = llhd.const 0 : i1
    %toggle = llhd.sig "toggle" %0 : i1 -> !llhd.sig<i1>
    %1 = llhd.prb %toggle : !llhd.sig<i1> -> i1
    %2 = llhd.not %1 : i1
    %dt = llhd.const #llhd.time<1ns, 0d, 0e> : !llhd.time
    llhd.drv %toggle, %2, %dt : !llhd.sig<i1>, i1, !llhd.time
}
