//RUN: llhdc %s | llhdc | FileCheck %s

// CHECK-LABEL: func @lap
func @lap() -> i64 {
// CHECK-NEXT: %{{.*}} = llhd.const 5 : i64 
// CHECK-NEXT: return %{{.*}} : i64
// CHECK-NEXT: }
    %0 = "llhd.const"() {value = 5 : i64} : () -> i64
    return %0 : i64
    }
