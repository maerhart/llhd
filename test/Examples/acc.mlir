llhd.entity @acc (%clk : !llhd.sig<i1>, %x : !llhd.sig<i32>, %en : !llhd.sig<i1>) -> (%q : !llhd.sig<i32>) {
    %zero = llhd.const 0 : i32
    %d = llhd.sig "d" %zero : i32
    // Output signals also need to be created in the entity instantiating this entity
    // We can only drive the signal to zero here, to make sure it has the right value
    // %q = llhd.sig %zero : i32 -> !llhd.sig<i32>
    llhd.inst @acc_ff(%clk, %d) -> (%q) : (!llhd.sig<i1>, !llhd.sig<i32>) -> !llhd.sig<i32>
    llhd.inst @acc_comb(%q, %x, %en) -> (%d) : (!llhd.sig<i32>, !llhd.sig<i32>, !llhd.sig<i1>) -> !llhd.sig<i32>
}

llhd.proc @acc_ff(%clk : !llhd.sig<i1>, %d : !llhd.sig<i32>) -> (%q : !llhd.sig<i32>) {
    br ^init
^init:
    %clk0 = llhd.prb %clk : !llhd.sig<i1>
    llhd.wait (%clk : !llhd.sig<i1>), ^check
^check:
    %clk1 = llhd.prb %clk : !llhd.sig<i1>
    %chg = cmpi "ne", %clk0, %clk1 : i1
    %posedge = llhd.and %chg, %clk1 : i1
    cond_br %posedge, ^init, ^event
^event:
    %dp = llhd.prb %d : !llhd.sig<i32>
    %delay = llhd.const #llhd.time<1ns, 0d, 0e> : !llhd.time
    llhd.drv %q, %dp after %delay : !llhd.sig<i32>
    br ^init
}

llhd.proc @acc_comb(%q : !llhd.sig<i32>, %x : !llhd.sig<i32>, %en : !llhd.sig<i1>) -> (%d : !llhd.sig<i32>) {
    br ^entry
^entry:
    %qp = llhd.prb %q : !llhd.sig<i32>
    %enp = llhd.prb %en : !llhd.sig<i1>
    %delay = llhd.const #llhd.time<2ns, 0d, 0e> : !llhd.time
    llhd.drv %d, %qp after %delay : !llhd.sig<i32>
    cond_br %enp, ^final, ^enabled
^enabled:
    %xp = llhd.prb %x : !llhd.sig<i32>
    %sum = addi %qp, %xp : i32
    llhd.drv %d, %sum after %delay : !llhd.sig<i32>
    br ^final
^final:
    llhd.wait (%q, %x, %en : !llhd.sig<i32>, !llhd.sig<i32>, !llhd.sig<i1>), ^entry
}
