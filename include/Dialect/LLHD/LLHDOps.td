#ifndef LLHD_OPS
#define LLHD_OPS

include "Dialect/LLHD/LLHDBase.td"

//===----------------------------------------------------------------------===//
// Constant Operation
//===----------------------------------------------------------------------===//

def LLHD_ConstOp : LLHD_Op<"const", [NoSideEffect]> {
    let summary = "Introduce a new constant.";

    let description = [{
        The `llhd.const` instruction introduces a new constant value as an 
        SSA-operator.  
        Legal types are integers and time. Note: Signals 
        are not legal to define using `llhd.const`, use the `llhd.sig` 
        instruction for that.

        **Custom syntax:**

        ```
        const-op ::= <ssa-id> `=` `llhd.const` attr-dict <attribute-value> `:` <result-type>
        ```

        **Examples:**
        
        ```mlir
        %0 = llhd.const 1 : i64
        %1 = llhd.const #llhd.time<1ns, 2d, 3d> : !llhd.time
        ```
    }];

    let arguments = (ins AnyAttr:$value);
    let results = (outs LLHD_ElementType:$out);
}

//===----------------------------------------------------------------------===//
//=== Signal Operations
//===----------------------------------------------------------------------===//

def LLHD_SigOp : LLHD_Op<"sig", [NoSideEffect]> {
    let summary = "Create a signal.";

    let description = [{
       The `llhd.sig` instruction introduces a new signal in the IR. The input 
       operand determines the initial value carried by the signal, while the
       result type will always be a signal carrying the type of the init operand.

       **Custom syntax:**

       ```
       sig-op ::= <ssa-id> `=` `llhd.sig` <ssa-use> attr-dict `:` <init-type> `->` !llhd.sig< <init-type> >
       ```

       **Examples:**

       ```mlir
       %init_i64 = llhd.const 123 : i64
       %sig_i64 = llhd.sig %init_64 : i64 -> !llhd.sig<i64>

       %init_i1 = llhd.const 1 : i1
       %sig_i1 = llhd.sig %init_i1 : i1 -> !llhd.sig<i1>
       ```
       The first `llhd.sig` instruction creates a new signal carrying an `i64` 
       type with initial value of 123, while the second one creates a new signal
       carrying an `i1` value with initial value of 1.
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

//===----------------------------------------------------------------------===//
//=== Control Flow Operations
//===----------------------------------------------------------------------===//

def LLHD_TerminatorOp : LLHD_Op<"terminator", [Terminator]> {
    let summary = "Dummy terminator";
    // TODO: add description
    let parser = ?;
    let printer = ?;
}

def LLHD_WaitOp : LLHD_Op<"wait", 
        [Terminator, AttrSizedOperandSegments, HasParent<"ProcOp">]> {
    let summary = "Suspends execution of a process.";
    // TODO: improve description
    let description = [{
        The `wait` instruction suspends execution of a process until any of the 
        observed signals change or optionally a fixed time interval has passed.
    }];

    let arguments = (ins Variadic<LLHD_SigType>:$obs, 
                         Variadic<LLHD_Type>:$destOps, 
                         I32ElementsAttr:$operand_segment_sizes);
    let successors = (successor AnySuccessor:$dest);

//    let assemblyFormat = [{
//        $obs `[` $dest (`(` $destOps^ `:` type($destOps) `)`)? `]` attr-dict `:` type($obs)
//    }];
}

def LLHD_HaltOp : LLHD_Op<"halt", [Terminator, HasParent<"ProcOp">]> {
    let summary = "Terminates execution of a process.";
    let description = [{
        The `halt` instruction terminates execution of a process. All processes 
        must eventually halt or consist of an infinite loop.
        * This is a terminator instruction
    }];

    let assemblyFormat = "attr-dict";
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

        | `not` |  U  |  X  |  0  |  1  |  Z  |  W  |  L  |  H  |  -  |
        |:-----:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
        |       |  U  |  X  |  1  |  0  |  X  |  X  |  1  |  0  |  X  |
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

        | `and` |  U  |  X  |  0  |  1  |  Z  |  W  |  L  |  H  |  -  |
        |:-----:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
        |   U   |  U  |  U  |  0  |  U  |  U  |  U  |  0  |  U  |  U  |
        |   X   |  U  |  X  |  0  |  X  |  X  |  X  |  0  |  X  |  X  |
        |   0   |  0  |  0  |  0  |  0  |  0  |  0  |  0  |  0  |  0  |
        |   1   |  U  |  X  |  0  |  1  |  X  |  X  |  0  |  1  |  X  |
        |   Z   |  U  |  X  |  0  |  X  |  X  |  X  |  0  |  X  |  X  |
        |   W   |  U  |  X  |  0  |  X  |  X  |  X  |  0  |  X  |  X  |
        |   L   |  0  |  0  |  0  |  0  |  0  |  0  |  0  |  0  |  0  |
        |   H   |  U  |  X  |  0  |  1  |  X  |  X  |  0  |  1  |  X  |
        |   -   |  U  |  X  |  0  |  X  |  X  |  X  |  0  |  X  |  X  |
    }];

    let arguments = (ins AnyTypeOf<[LLHD_iN, LLHD_lN]>:$lhs, 
                         AnyTypeOf<[LLHD_iN, LLHD_lN]>:$rhs);
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

        | `or`  |  U  |  X  |  0  |  1  |  Z  |  W  |  L  |  H  |  -  |
        |:-----:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
        |   U   |  U  |  U  |  U  |  1  |  U  |  U  |  U  |  1  |  U  |
        |   X   |  U  |  X  |  X  |  1  |  X  |  X  |  X  |  1  |  X  |
        |   0   |  U  |  X  |  0  |  1  |  X  |  X  |  0  |  1  |  X  |
        |   1   |  1  |  1  |  1  |  1  |  1  |  1  |  1  |  1  |  1  |
        |   Z   |  U  |  X  |  X  |  1  |  X  |  X  |  X  |  1  |  X  |
        |   W   |  U  |  X  |  X  |  1  |  X  |  X  |  X  |  1  |  X  |
        |   L   |  U  |  X  |  0  |  1  |  X  |  X  |  0  |  1  |  X  |
        |   H   |  1  |  1  |  1  |  1  |  1  |  1  |  1  |  1  |  1  |
        |   -   |  U  |  X  |  X  |  1  |  X  |  X  |  X  |  1  |  X  |
    }];

    let arguments = (ins AnyTypeOf<[LLHD_iN, LLHD_lN]>:$lhs,
                         AnyTypeOf<[LLHD_iN, LLHD_lN]>:$rhs);
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

        | `xor` |  U  |  X  |  0  |  1  |  Z  |  W  |  L  |  H  |  -  |
        |:-----:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
        |   U   |  U  |  U  |  U  |  U  |  U  |  U  |  U  |  U  |  U  |
        |   X   |  U  |  X  |  X  |  X  |  X  |  X  |  X  |  X  |  X  |
        |   0   |  U  |  X  |  0  |  1  |  X  |  X  |  0  |  1  |  X  |
        |   1   |  U  |  X  |  1  |  0  |  X  |  X  |  1  |  0  |  X  |
        |   Z   |  U  |  X  |  X  |  X  |  X  |  X  |  X  |  X  |  X  |
        |   W   |  U  |  X  |  X  |  X  |  X  |  X  |  X  |  X  |  X  |
        |   L   |  U  |  X  |  0  |  1  |  X  |  X  |  0  |  1  |  X  |
        |   H   |  U  |  X  |  1  |  0  |  X  |  X  |  1  |  0  |  X  |
        |   -   |  U  |  X  |  X  |  X  |  X  |  X  |  X  |  X  |  X  |
    }];

    let arguments = (ins AnyTypeOf<[LLHD_iN, LLHD_lN]>:$lhs, 
                         AnyTypeOf<[LLHD_iN, LLHD_lN]>:$rhs);
    let results = (outs AnyTypeOf<[LLHD_iN, LLHD_lN]>);

    let assemblyFormat = "`(` operands `)` attr-dict `:` `(` type(operands) `)` `->` type(results)";
}

def LLHD_ShlOp : LLHD_Op<"shl", [NoSideEffect]> {
    let summary = "Shifts a value to the left by a given amount.";
    let description = [{
        The instruction is transparent to signals and pointers. For example,
        passing a signal as argument will shift the underlying value and return 
        a signal to the shifted value.

        ```
        %result = llhd.shl(%base, %hidden, %amount) : (T, Th, Ta) -> T
        ```
        * `T` must be `iN` or `lN`, or an array; or a signal/pointer thereof.
        * `Th` must be of the same type as T, but may have a different number 
          of bits of elements.
        * The maximum shift amount is determined by the number of bits or 
          elements in `Th`.
        * `Ta` must be `iN`.
        * `%base` is the base value that is produced if the shift amount is 0, 
          and must be of type `T`.
        * `%hidden` is the hidden value that is uncovered by non-zero shift 
          amounts, and must be of type `Th`.
        * `%amount` is the unsigned shift amount and determines by how many 
          positions the value is to be shifted. Must be of type `Ta`. Behavior 
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
        The instruction is transparent to signals and pointers. For example, 
        passing a signal as argument will shift the underlying value and return 
        a signal to the shifted value.

        ```
        %result = llhd.shr(%base, %hidden, %amount) : (T, Th, Ta) -> T
        ```
        * `T` must be `iN` or `lN`, or an array; or a signal/pointer thereof.
        * `Th` must be of the same type as T, but may have a different number 
          of bits of elements.
        * The maximum shift amount is determined by the number of bits or 
          elements in `Th`.
        * `Ta` must be `iN`.
        * `%base` is the base value that is produced if the shift amount is 0, 
          and must be of type `T`.
        * `%hidden` is the hidden value that is uncovered by non-zero shift 
          amounts, and must be of type `Th`.
        * `%amount` is the unsigned shift amount and determines by how many 
          positions the value is to be shifted. Must be of type `Ta`. Behavior 
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
    // TODO: add descriptioand change the main cmake file to install the td and def filesn
    let arguments = (ins I64Attr:$ins);
    let regions = (region SizedRegion<1>:$body);
}

def LLHD_ProcOp : LLHD_Op<"proc", [Symbol, FunctionLike, IsolatedFromAbove]> {
    let summary = "";
    // TODO: add description
    let description = [{

    }];

    let arguments = (ins I64Attr:$ins);
    let regions = (region AnyRegion:$body);

    let extraClassDeclaration = [{
        friend class OpTrait::FunctionLike<ProcOp>;

        /// Hooks for the input/output type enumeration in FunctionLike.
        unsigned getNumFuncArguments() { return getType().getNumInputs(); }
        unsigned getNumFuncResults() { return getType().getNumResults(); }

        /// Hook for FunctionLike verifier.
        LogicalResult verifyType();

        /// Verifies the body of the function.
        LogicalResult verifyBody();
    }];

    let verifier = [{ return ::verify(*this); }];
}


#endif // LLHD_OPS
