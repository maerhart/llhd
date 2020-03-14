#ifndef LLHD_OPS
#define LLHD_OPS

#ifndef LLHD_BASE
include "Dialect/LLHD/LLHDBase.td"
#endif // LLHD_BASE


def LLHD_ConstOp : LLHD_Op<"const", [NoSideEffect]> {
    let summary = "Constant";

    let description = [{
        The const instruction is used to introduce a constant value into the IR. The first version constructs a constant integer value, the second a constant integer signal, and the third a constant time value.
        ```
        %result = const iN <int>
        %result = const iN$ <int>
        %result = const time <time>
        ```
        `int` is an integer literal such as `0b0101`, `0o1247`, `129`, or `0x14F3E`
        time is a time literal such as `1s`, `1s 2d`, or `1s 2d 3e`, where the real component may carry an SI prefix such as `as`, `fs`, `ps`, `ns`, `us`, `ms`.
    }];

    let arguments = (ins I64Attr:$value);
    
    let results = (outs LLHD_Type);


    let builders = [
        // Build a constant with a given constant integer value.
        OpBuilder<"Builder *builder, OperationState &state, int value">
    ];
}

#endif // LLHD_OPS
