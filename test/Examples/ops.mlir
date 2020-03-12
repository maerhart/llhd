func @test() -> i64 {
    %0 = "llhd.const"() { value = 5 : i64 } : () -> i64
    return %0 : i64
}