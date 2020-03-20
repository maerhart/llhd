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
        %result = llhd.const <int> : iN
        %result = llhd.const <int> : !llhd.sig<iN>
        %result = llhd.const <time> : time
        ```
        `int` is an integer literal such as `0b0101`, `0o1247`, `129`, or `0x14F3E`
        time is a time literal such as `1s`, `1s 2d`, or `1s 2d 3e`, where the real component may carry an SI prefix such as `as`, `fs`, `ps`, `ns`, `us`, `ms`.
    }];

    let arguments = (ins I64Attr:$value);
    let results = (outs LLHD_Type:$out);

    let assemblyFormat = "$value attr-dict `:` type($out)";

    let builders = [
        // Build a constant with a given constant integer value.
        OpBuilder<"Builder *builder, OperationState &state, int64_t value">
    ];
}

def LLHD_SigOp : LLHD_Op<"sig", [NoSideEffect]> {
    let summary = "Create a signal.";

    // TODO: udpate description
    let description = [{
        ```
        %result = llhd.sig %init : T
        ```

        The `sig` instruction creates a signal in an entity with the initial value `%init` and returns that signal.

        * `T` may be any type.  
        * `%init` is the initial value of the signal and must be of type `T`.  
        * `%result` is of type `T$`.  
    }];

    let arguments = (ins LLHD_ElementType:$init);
    let results = (outs LLHD_SigType);

    let verifier = [{ return ::verify(*this); }];
}

def LLHD_PrbOp : LLHD_Op<"prb", []> {
    let summary = "Probe a signal.";

    // TODO: update description
    let description = [{
        ```
        %result = prb T$ %sig
        ```
        The `prb` instruction probes the current value of a signal `%sig`.

        * `T` may be any type.
        * `%sig` must be of type `T$`.
        * `%result` is of type `T`.
    }];

    let arguments = (ins LLHD_SigType:$signal);
    let results = (outs LLHD_ElementType:$output);

    let verifier = [{ return ::verify(*this); }];
}

def LLHD_DrvOp : LLHD_Op<"drv"> {
    let summary = "Drive a value into a signal.";

    // TODO: update description
    let description = [{
        ```
        drv T$ %sig, %value
        ```
        The drv instruction drives a `%value` onto a signal `%sig`.

        * `T` may be any type.
        * `%sig` must be of type `T$`.
        * `%value` must be of type `T`.
    }];

    let arguments = (ins LLHD_SigType:$signal, LLHD_ElementType:$value);

    let verifier = [{ return ::verify(*this); }];
}

def LLHD_TerminatorOp : LLHD_Op<"terminator", [Terminator]> {
    let summary = "Dummy terminator";
    // TODO: add description
    let parser = ?;
    let printer = ?;
}


//===----------------------------------------------------------------------===//
//=== Bitwise Operations
//===----------------------------------------------------------------------===//

def LLHD_NotOp : LLHD_Op<"not", [NoSideEffect, SameOperandsAndResultType]> {
    let summary = "Flips each bit of a value.";
    let description = [{
        ```
        %result = llhd.not(%value) : (T) -> T
        ```
        * `T` must be `iN` or `lN`.
        * `%value` is the input argument of type `T`.
        * `%result` is of type `T`.

        Truth Table for `iN`:

        | `not` |  0  |  1  |
        |:-----:|:---:|:---:|
        |       |  1  |  0  |

        Truth Table for `lN`:
        TODO
        | `not` |  U  |  X  |  0  |  1  |  Z  |  W  |  L  |  H  |  -  |
        |:-----:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
        |       |     |     |     |     |     |     |     |     |     |
    }];

    let arguments = (ins AnyTypeOf<[LLHD_iN, LLHD_lN]>:$value);
    let results = (outs AnyTypeOf<[LLHD_iN, LLHD_lN]>);

    let assemblyFormat = "`(` operands `)` attr-dict `:` `(` type(operands) `)` `->` type(results)";
}

def LLHD_AndOp : LLHD_Op<"and", [NoSideEffect, Commutative, SameOperandsAndResultType]> {
    let summary = "Computes the bitwise AND of two values.";
    let description = [{
        ```
        %result = llhd.and(%lhs, %rhs) : (T, T) -> T
        ```
        * `T` must be `iN` or `lN`.
        * `%lhs` and `%rhs` are the input arguments of type `T`.
        * `%result` is of type `T`.

        Truth Table for `iN`:

        | `and` |  0  |  1  |
        |:-----:|:---:|:---:|
        |   0   |  0  |  0  |
        |   1   |  0  |  1  |

        Truth Table for `lN`:
        TODO
        | `and` |  U  |  X  |  0  |  1  |  Z  |  W  |  L  |  H  |  -  |
        |:-----:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
        |   U   |     |     |     |     |     |     |     |     |     |
        |   X   |     |     |     |     |     |     |     |     |     |
        |   0   |     |     |     |     |     |     |     |     |     |
        |   1   |     |     |     |     |     |     |     |     |     |
        |   Z   |     |     |     |     |     |     |     |     |     |
        |   W   |     |     |     |     |     |     |     |     |     |
        |   L   |     |     |     |     |     |     |     |     |     |
        |   H   |     |     |     |     |     |     |     |     |     |
        |   -   |     |     |     |     |     |     |     |     |     |
    }];

    let arguments = (ins AnyTypeOf<[LLHD_iN, LLHD_lN]>:$lhs, AnyTypeOf<[LLHD_iN, LLHD_lN]>:$rhs);
    let results = (outs AnyTypeOf<[LLHD_iN, LLHD_lN]>);

    let assemblyFormat = "`(` operands `)` attr-dict `:` `(` type(operands) `)` `->` type(results)";
}

def LLHD_OrOp : LLHD_Op<"or", [NoSideEffect, Commutative, SameOperandsAndResultType]> {
    let summary = "Computes the bitwise OR of two values.";
    let description = [{
        ```
        %result = llhd.or(%lhs, %rhs) : (T, T) -> T
        ```
        * `T` must be `iN` or `lN`.
        * `%lhs` and `%rhs` are the input arguments of type `T`.
        * `%result` is of type `T`.

        Truth Table for `iN`:

        | `or`  |  0  |  1  |
        |:-----:|:---:|:---:|
        |   0   |  0  |  0  |
        |   1   |  0  |  1  |

        Truth Table for `lN`:
        TODO
        | `or`  |  U  |  X  |  0  |  1  |  Z  |  W  |  L  |  H  |  -  |
        |:-----:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
        |   U   |     |     |     |     |     |     |     |     |     |
        |   X   |     |     |     |     |     |     |     |     |     |
        |   0   |     |     |     |     |     |     |     |     |     |
        |   1   |     |     |     |     |     |     |     |     |     |
        |   Z   |     |     |     |     |     |     |     |     |     |
        |   W   |     |     |     |     |     |     |     |     |     |
        |   L   |     |     |     |     |     |     |     |     |     |
        |   H   |     |     |     |     |     |     |     |     |     |
        |   -   |     |     |     |     |     |     |     |     |     |
    }];

    let arguments = (ins AnyTypeOf<[LLHD_iN, LLHD_lN]>:$lhs, AnyTypeOf<[LLHD_iN, LLHD_lN]>:$rhs);
    let results = (outs AnyTypeOf<[LLHD_iN, LLHD_lN]>);

    let assemblyFormat = "`(` operands `)` attr-dict `:` `(` type(operands) `)` `->` type(results)";
}

def LLHD_XorOp : LLHD_Op<"xor", [NoSideEffect, Commutative, SameOperandsAndResultType]> {
    let summary = "Computes the bitwise XOR of two values.";
    let description = [{
        ```
        %result = llhd.xor(%lhs, %rhs) : (T, T) -> T
        ```
        * `T` must be `iN` or `lN`.
        * `%lhs` and `%rhs` are the input arguments of type `T`.
        * `%result` is of type `T`.

        Truth Table for `iN`:

        | `xor` |  0  |  1  |
        |:-----:|:---:|:---:|
        |   0   |  0  |  0  |
        |   1   |  0  |  1  |

        Truth Table for `lN`:
        TODO
        | `xor` |  U  |  X  |  0  |  1  |  Z  |  W  |  L  |  H  |  -  |
        |:-----:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
        |   U   |     |     |     |     |     |     |     |     |     |
        |   X   |     |     |     |     |     |     |     |     |     |
        |   0   |     |     |     |     |     |     |     |     |     |
        |   1   |     |     |     |     |     |     |     |     |     |
        |   Z   |     |     |     |     |     |     |     |     |     |
        |   W   |     |     |     |     |     |     |     |     |     |
        |   L   |     |     |     |     |     |     |     |     |     |
        |   H   |     |     |     |     |     |     |     |     |     |
        |   -   |     |     |     |     |     |     |     |     |     |
    }];

    let arguments = (ins AnyTypeOf<[LLHD_iN, LLHD_lN]>:$lhs, AnyTypeOf<[LLHD_iN, LLHD_lN]>:$rhs);
    let results = (outs AnyTypeOf<[LLHD_iN, LLHD_lN]>);

    let assemblyFormat = "`(` operands `)` attr-dict `:` `(` type(operands) `)` `->` type(results)";
}

def LLHD_ShlOp : LLHD_Op<"shl", [NoSideEffect]> {
    let summary = "Shifts a value to the left by a given amount.";
    let description = [{
        The instruction is transparent to signals and pointers. For example, \
        passing a signal as argument will shift the underlying value and return \
        a signal to the shifted value.

        ```
        %result = llhd.shl(%base, %hidden, %amount) : (T, Th, Ta) -> T
        ```
        * `T` must be `iN` or `lN`, or an array; or a signal/pointer thereof.
        * `Th` must be of the same type as T, but may have a different number \
          of bits of elements.
        * The maximum shift amount is determined by the number of bits or \
          elements in `Th`.
        * `Ta` must be `iN`.
        * `%base` is the base value that is produced if the shift amount is 0, \
          and must be of type `T`.
        * `%hidden` is the hidden value that is uncovered by non-zero shift \
          amounts, and must be of type `Th`.
        * `%amount` is the unsigned shift amount and determines by how many \
          positions the value is to be shifted. Must be of type `Ta`. Behavior \
          for values `%amount > N` is undefined.
        * `%result` is of type `T`.
    }];

    // TODO: adjust type T and Th to include arrays and pointers
    let arguments = (ins AnyTypeOf<[LLHD_iN, LLHD_lN, LLHD_SigType]>:$base, 
                         AnyTypeOf<[LLHD_iN, LLHD_lN, LLHD_SigType]>:$hidden,
                         LLHD_iN:$amount);
    let results = (outs AnyTypeOf<[LLHD_iN, LLHD_lN, LLHD_SigType]>:$result);

    let assemblyFormat = "`(` operands `)` attr-dict `:` `(` type(operands) `)` `->` type(results)";

    let verifier = [{ return ::verify(*this); }];
}

def LLHD_ShrOp : LLHD_Op<"shr", [NoSideEffect]> {
    let summary = "Shifts a value to the right by a given amount.";
    let description = [{
        The instruction is transparent to signals and pointers. For example, \
        passing a signal as argument will shift the underlying value and return \
        a signal to the shifted value.

        ```
        %result = llhd.shr(%base, %hidden, %amount) : (T, Th, Ta) -> T
        ```
        * `T` must be `iN` or `lN`, or an array; or a signal/pointer thereof.
        * `Th` must be of the same type as T, but may have a different number \
          of bits of elements.
        * The maximum shift amount is determined by the number of bits or \
          elements in `Th`.
        * `Ta` must be `iN`.
        * `%base` is the base value that is produced if the shift amount is 0, \
          and must be of type `T`.
        * `%hidden` is the hidden value that is uncovered by non-zero shift \
          amounts, and must be of type `Th`.
        * `%amount` is the unsigned shift amount and determines by how many \
          positions the value is to be shifted. Must be of type `Ta`. Behavior \
          for values `%amount > N` is undefined.
        * `%result` is of type `T`.
    }];

    // TODO: adjust type T and Th to include arrays and pointers
    let arguments = (ins AnyTypeOf<[LLHD_iN, LLHD_lN, LLHD_SigType]>:$base, 
                         AnyTypeOf<[LLHD_iN, LLHD_lN, LLHD_SigType]>:$hidden,
                         AnySignlessInteger:$amount);
    let results = (outs AnyTypeOf<[LLHD_iN, LLHD_lN, LLHD_SigType]>:$result);

    let assemblyFormat = "`(` operands `)` attr-dict `:` `(` type(operands) `)` `->` type(results)";

    let verifier = [{ return ::verify(*this); }];
}


//===----------------------------------------------------------------------===//
//=== Arithmetic Operations
//===----------------------------------------------------------------------===//

def LLHD_NegOp : LLHD_Op<"neg", [NoSideEffect, SameOperandsAndResultType]> {
    let summary = "Negate a value.";
    let description = [{
        ```
        %result = llhd.neg(%value) : (T) -> T
        ```
        * `T` must be `iN`
        * `%value` is the input argument of type `T`.
        * `%result` is of type `T`.
    }];

    let arguments = (ins LLHD_iN:$value);
    let results = (outs LLHD_iN);

    let assemblyFormat = "`(` operands `)` attr-dict `:` `(` type(operands) `)` `->` type(results)";
}

def LLHD_AddOp : LLHD_Op<"add", [NoSideEffect, Commutative, SameOperandsAndResultType]> {
    let summary = "Add two values.";
    let description = [{
        ```
        %result = llhd.add(%lhs, %rhs) : (T, T) -> T
        ```
        * `T` must be `iN`
        * `%lhs` and `%rhs` are the left- and right-hand side arguments and must be of type `T`.
        * `%result` is of type `T`.
    }];

    let arguments = (ins LLHD_iN:$lhs, LLHD_iN:$rhs);
    let results = (outs LLHD_iN);

    let assemblyFormat = "`(` operands `)` attr-dict `:` `(` type(operands) `)` `->` type(results)";
}

def LLHD_SubOp : LLHD_Op<"sub", [NoSideEffect, SameOperandsAndResultType]> {
    let summary = "Subtract two values.";
    let description = [{
        ```
        %result = llhd.sub(%lhs, %rhs) : (T, T) -> T
        ```
        * `T` must be `iN`
        * `%lhs` and `%rhs` are the left- and right-hand side arguments and must be of type `T`.
        * `%result` is of type `T`.
    }];

    let arguments = (ins LLHD_iN:$lhs, LLHD_iN:$rhs);
    let results = (outs LLHD_iN);

    let assemblyFormat = "`(` operands `)` attr-dict `:` `(` type(operands) `)` `->` type(results)";
}

def LLHD_SMulOp : LLHD_Op<"smul", [NoSideEffect, Commutative, SameOperandsAndResultType]> {
    let summary = "Signed multiplication of two values.";
    let description = [{
        ```
        %result = llhd.smul(%lhs, %rhs) : (T, T) -> T
        ```
        * `T` must be `iN`
        * `%lhs` and `%rhs` are the left- and right-hand side arguments and must be of type `T`.
        * `%result` is of type `T`.
    }];

    let arguments = (ins LLHD_iN:$lhs, LLHD_iN:$rhs);
    let results = (outs LLHD_iN);

    let assemblyFormat = "`(` operands `)` attr-dict `:` `(` type(operands) `)` `->` type(results)";
}

def LLHD_UMulOp : LLHD_Op<"umul", [NoSideEffect, Commutative, SameOperandsAndResultType]> {
    let summary = "Unsigned multiplication of two values.";
    let description = [{
        ```
        %result = llhd.umul(%lhs, %rhs) : (T, T) -> T
        ```
        * `T` must be `iN`
        * `%lhs` and `%rhs` are the left- and right-hand side arguments and must be of type `T`.
        * `%result` is of type `T`.
    }];

    let arguments = (ins LLHD_iN:$lhs, LLHD_iN:$rhs);
    let results = (outs LLHD_iN);

    let assemblyFormat = "`(` operands `)` attr-dict `:` `(` type(operands) `)` `->` type(results)";
}

def LLHD_SDivOp : LLHD_Op<"sdiv", [NoSideEffect, SameOperandsAndResultType]> {
    let summary = "Signed division of two values.";
    let description = [{
        ```
        %result = llhd.sdiv(%lhs, %rhs) : (T, T) -> T
        ```
        * `T` must be `iN`
        * `%lhs` and `%rhs` are the left- and right-hand side arguments and must be of type `T`.
        * `%result` is of type `T`.
    }];

    let arguments = (ins LLHD_iN:$lhs, LLHD_iN:$rhs);
    let results = (outs LLHD_iN);

    let assemblyFormat = "`(` operands `)` attr-dict `:` `(` type(operands) `)` `->` type(results)";
}

def LLHD_UDivOp : LLHD_Op<"udiv", [NoSideEffect, SameOperandsAndResultType]> {
    let summary = "Unsigned division of two values.";
    let description = [{
        ```
        %result = llhd.udiv(%lhs, %rhs) : (T, T) -> T
        ```
        * `T` must be `iN`
        * `%lhs` and `%rhs` are the left- and right-hand side arguments and must be of type `T`.
        * `%result` is of type `T`.
    }];

    let arguments = (ins LLHD_iN:$lhs, LLHD_iN:$rhs);
    let results = (outs LLHD_iN);

    let assemblyFormat = "`(` operands `)` attr-dict `:` `(` type(operands) `)` `->` type(results)";
}

def LLHD_SModOp : LLHD_Op<"smod", [NoSideEffect, SameOperandsAndResultType]> {
    let summary = "Signed modulo of the signed division of two values.";
    let description = [{
        ```
        %result = llhd.smod(%lhs, %rhs) : (T, T) -> T
        ```
        * `T` must be `iN`
        * `%lhs` and `%rhs` are the left- and right-hand side arguments and must be of type `T`.
        * `%result` is of type `T`.
    }];

    let arguments = (ins LLHD_iN:$lhs, LLHD_iN:$rhs);
    let results = (outs LLHD_iN);

    let assemblyFormat = "`(` operands `)` attr-dict `:` `(` type(operands) `)` `->` type(results)";
}

def LLHD_UModOp : LLHD_Op<"umod", [NoSideEffect, SameOperandsAndResultType]> {
    let summary = "Unsigned modulo of the unsigned division of two values.";
    let description = [{
        ```
        %result = llhd.umod(%lhs, %rhs) : (T, T) -> T
        ```
        * `T` must be `iN`
        * `%lhs` and `%rhs` are the left- and right-hand side arguments and must be of type `T`.
        * `%result` is of type `T`.
    }];

    let arguments = (ins LLHD_iN:$lhs, LLHD_iN:$rhs);
    let results = (outs LLHD_iN);

    let assemblyFormat = "`(` operands `)` attr-dict `:` `(` type(operands) `)` `->` type(results)";
}

def LLHD_SRemOp : LLHD_Op<"srem", [NoSideEffect, SameOperandsAndResultType]> {
    let summary = "Signed reminder of the signed division two values.";
    let description = [{
        ```
        %result = llhd.srem(%lhs, %rhs) : (T, T) -> T
        ```
        * `T` must be `iN`
        * `%lhs` and `%rhs` are the left- and right-hand side arguments and must be of type `T`.
        * `%result` is of type `T`.
    }];

    let arguments = (ins LLHD_iN:$lhs, LLHD_iN:$rhs);
    let results = (outs LLHD_iN);

    let assemblyFormat = "`(` operands `)` attr-dict `:` `(` type(operands) `)` `->` type(results)";
}

def LLHD_URemOp : LLHD_Op<"urem", [NoSideEffect, SameOperandsAndResultType]> {
    let summary = "Unsigned reminder of the unsigned division two values.";
    let description = [{
        ```
        %result = llhd.urem(%lhs, %rhs) : (T, T) -> T
        ```
        * `T` must be `iN`
        * `%lhs` and `%rhs` are the left- and right-hand side arguments and must be of type `T`.
        * `%result` is of type `T`.
    }];

    let arguments = (ins LLHD_iN:$lhs, LLHD_iN:$rhs);
    let results = (outs LLHD_iN);

    let assemblyFormat = "`(` operands `)` attr-dict `:` `(` type(operands) `)` `->` type(results)";
}

//===----------------------------------------------------------------------===//
//=== Units
//===----------------------------------------------------------------------===//

def LLHD_EntityOp : LLHD_Op<"entity", [Symbol, IsolatedFromAbove, SingleBlockImplicitTerminator<"TerminatorOp">]> {
    let summary = "Define an LLHD entity.";
    // TODO: add description
    let arguments = (ins I64Attr:$ins);
    let regions = (region SizedRegion<1>:$body);
}


#endif // LLHD_OPS
