//RUN: llhdc %s --convert-llhd-to-llvm --split-input-file | FileCheck %s

// CHECK: llvm.func @convert_entity(%{{.*}}:  !llvm<"i8*">) {
// CHECK-NEXT: llvm.return
// CHECK-NEXT: }
llhd.entity @convert_entity () -> () {}
