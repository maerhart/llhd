llhd.entity @binary_to_gray.param1 (%A : !llhd.sig<i8>) -> (%Z : !llhd.sig<i8>) {
    %A1 = llhd.prb %A : !llhd.sig<i8> -> i8
    %A2 = llhd.prb %A : !llhd.sig<i8> -> i8
    %0 = llhd.const 1 : i32
    %1 = llhd.const 0 : i8
    %2 = llhd.shr %A2, %1, %0 : (i8, i8, i32) -> i8
    %3 = llhd.xor %A1, %2 : i8
    %4 = llhd.const #llhd.time<0s, 0d, 1e> : !llhd.time
    llhd.drv %Z, %3, %4 : !llhd.sig<i8>, i8, !llhd.time
}

llhd.entity @graycode (%A : !llhd.sig<i8>) -> (%G : !llhd.sig<i8>, %Z : !llhd.sig<i8>) {
    llhd.inst @binary_to_gray.param1 (%A) -> (%G) : (!llhd.sig<i8>) -> !llhd.sig<i8>
}
