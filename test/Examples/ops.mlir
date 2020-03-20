func @test() -> i64 {
    %0 = "llhd.const"() { value = 5 : i64 } : () -> i64
    %1 = llhd.const 6 : i64
    %2 = "llhd.add"(%0, %0) : (i64, i64) -> i64
    %3 = llhd.add(%0, %0) : (i64, i64) -> i64
    %base = llhd.const 15 : i4
    %hidden = llhd.const 0 : i4
    %amount = llhd.const 2 : i2
    %4 = llhd.shl(%base, %hidden, %amount) : (i4, i4, i2) -> i4
    return %3 : i64
}