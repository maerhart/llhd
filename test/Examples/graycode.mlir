llhd.entity @binary_to_gray.param2 (%A : !llhd.sig<i20>) -> (%Z : !llhd.sig<i20>) {
    %A1 = llhd.prb %A : !llhd.sig<i20>
    %A2 = llhd.prb %A : !llhd.sig<i20>
    %0 = llhd.const 1 : i32
    %1 = llhd.const 0 : i20
    %2 = llhd.shr %A2, %1, %0 : (i20, i20, i32) -> i20
    %3 = llhd.xor %A1, %2 : i20
    %4 = llhd.const #llhd.time<0s, 0d, 1e> : !llhd.time
    llhd.drv %Z, %3 after %4 : !llhd.sig<i20>
}

llhd.entity @graycode.param1 (%A : !llhd.sig<i20>) -> (%G : !llhd.sig<i20>, %Z : !llhd.sig<i20>) {
    %A1 = llhd.prb %A : !llhd.sig<i20>
    %0 = llhd.const 0 : i20
    %1 = llhd.sig "sig_0" %0 : i20
    %2 = llhd.const #llhd.time<0s, 0d, 1e> : !llhd.time
    llhd.drv %1, %A1 after %2  : !llhd.sig<i20>
    llhd.inst "binary_to_gray" @binary_to_gray.param2 (%1) -> (%G) : (!llhd.sig<i20>) -> (!llhd.sig<i20>)
}

llhd.entity @root () -> () {
    %0 = llhd.const 0 : i20
    %a = llhd.sig "sig0" %0 : i20
    %1 = llhd.const 0 : i20
    %b = llhd.sig "graycode_output" %1 : i20
    %2 = llhd.const 0 : i20
    %c = llhd.sig "sig2" %2 : i20
    %3 = llhd.const 0 : i20
    %bp = llhd.sig "sig3" %3 : i20
    %a1 = llhd.prb %a : !llhd.sig<i20>
    %4 = llhd.const 123 : i20
    %5 = llhd.sig "graycode_input" %4 : i20
    %6 = llhd.const #llhd.time<0s, 0d, 1e> : !llhd.time
    llhd.drv %5, %a1 after %6 : !llhd.sig<i20>
    llhd.inst "graycode" @graycode.param1 (%5) -> (%b, %c) : (!llhd.sig<i20>) -> (!llhd.sig<i20>, !llhd.sig<i20>)
}