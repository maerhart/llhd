#ifndef LLHD_OPS
#define LLHD_OPS

include "mlir/Interfaces/SideEffects.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "LLHDBase.td"

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
        const-op ::= ssa-id `=` `llhd.const` attribute-value attr-dict `:` result-type
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
       sig-op ::= ssa-id `=` `llhd.sig` ssa-init attr-dict `:` init-type `->` !llhd.sig<init-type>
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

    let assemblyFormat = "$init attr-dict `:` type($init) `->` type(results)";

    let verifier = [{ return ::verify(*this); }];
}

def LLHD_PrbOp : LLHD_Op<"prb", []> {
    let summary = "Probe a signal.";

    let description = [{
       The `llhd.prb` instruction probes a signal and returns the value it
       currently carries as a new SSA operand. The result type is always
       the type carried by the signal.

        **Custom syntax:**

        ```
        prb-op ::= ssa-id `=` `llhd.prb` ssa-sig attr-dict `:` !llhd.sig<type> `->` type
        ```

        **Examples:***
        
        ```mlir
        %const_i1 = llhd.const 1 : i1
        %sig_i1 = llhd.sig %const_i1 : i1 -> !llhd.sig<i1>
        %prbd = llhd.prb %sig_i1 : !llhd.sig<i1> -> i1
        ```
    }];

    let arguments = (ins LLHD_SigType:$signal);
    let results = (outs LLHD_ElementType);

    let assemblyFormat = "$signal attr-dict `:` type($signal) `->` type(results)";

    let verifier = [{ return ::verify(*this); }];
}

def LLHD_DrvOp : LLHD_Op<"drv"> {
    let summary = "Drive a value into a signal.";

    let description = [{
        The `llhd.drv` operation drives a new value onto a signal. A time operand
        also has to be passed, which specifies the frequency at which the drive
        will be performed. This operation does not define any new SSA operands.

        **Custom Syntax:**

        ```
        drv-op ::= `llhd.drv` ssa-signal `,` ssa-const `,` ssa-time `:` !llhd.sig<const-type> `,` const-type `,` time-type
        ```

        **Examples:**

        ```mlir
        %init = llhd.const 1 : i1
        %time = llhd.const #llhd.time<1ns, 0d, 0e> : !llhd.time
        %sig = llhd.sig %init : i1 -> !llhd.sig<i1>
        %new = llhd.not %init : i1

        llhd.drv %sig, %new, %time : !llhd.sig<i1>, i1, !llhd.time
        ```
    }];

    let arguments = (ins LLHD_SigType:$signal, LLHD_ElementType:$value, LLHD_TimeType:$time);

    let assemblyFormat = "operands attr-dict `:` type(operands)";

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
        [Terminator, AttrSizedOperandSegments, HasParent<"ProcOp">, DeclareOpInterfaceMethods<BranchOpInterface>]> {
    let summary = "Suspends execution of a process.";
    let description = [{
        The `wait` instruction suspends execution of a process until any of the 
        observed signals change or a fixed time interval has passed. Execution
        resumes at the specified basic block with the passed arguments.
        * This is a terminator instruction.
        * This instruction is only allowed in processes (`llhd.proc`).

        **Syntax:**
        ```
        wait-op ::= `llhd.wait` ssa-list-obs (`for` ssa-time)? `,` successor-dest ( `(` ssa-list-dest-arguments `:` type-list-dest-arguments `)` )? `:` type-list-obs (`,` type-time)?
        ```
        Notes: 
        * `ssa-list-obs`, `ssa-list-dest-arguments`, `type-list-dest-arguments` and `type-list-obs` are comma-separated lists of 0 or more elements.
        * In case there is no optional time and `type-list-obs` has zero elements, the last colon is omitted as well.

        **Examples:**
        ```
        llhd.wait ^bb1
        llhd.wait for %time, ^bb1(%time : !llhd.time) : !llhd.time
        llhd.wait %0, %1, ^bb1(%1 : !llhd.sig<i1>) : !llhd.sig<i64>, !llhd.sig<i1>
        llhd.wait %0, %1 for %time, ^bb1(%1, %0 : !llhd.sig<i1>, !llhd.sig<i64>) : !llhd.sig<i64>, !llhd.sig<i1>, !llhd.time
        ```
    }];

    let arguments = (ins Variadic<LLHD_SigType>:$obs, 
                         Variadic<LLHD_TimeType>:$time,
                         Variadic<AnyType>:$destOps, 
                         I32ElementsAttr:$operand_segment_sizes); 
     
    let successors = (successor AnySuccessor:$dest);

    let verifier = [{ return ::verify(*this); }];
}

def LLHD_HaltOp : LLHD_Op<"halt", [Terminator, HasParent<"ProcOp">]> {
    let summary = "Terminates execution of a process.";
    let description = [{
        The `halt` instruction terminates execution of a process. All processes 
        must halt eventually or consist of an infinite loop.
        * This is a terminator instruction
        * This instruction is only allowed in processes (`llhd.proc`).

        **Syntax:**
        ```
        halt-op ::= `llhd.halt`
        ```

        **Examples:**
        ```
        llhd.halt
        ```
    }];

    let assemblyFormat = "attr-dict";
}

//===----------------------------------------------------------------------===//
//=== Bitwise Operations
//===----------------------------------------------------------------------===//

def LLHD_NotOp : LLHD_ArithmeticOrBitwiseOp<"not", []> {
    let summary = "Bitwise NOT";
    let description = [{
        Takes an integer of any width or a nine-valued-logic (IEEE 1164) value 
        of any width as input. Flips each bit of a value. The result always has 
        the exact same type.

        **Syntax:**
        ```
        not-op ::= ssa-id `=` `llhd.not` ssa-value attr-dict `:` type
        ```

        **Examples:**
        ```
        %0 = llhd.const 0 : i32
        %1 = llhd.not %0 : i32 
        ```

        **Truth Table for `iN`:**

        | `not` |  0  |  1  |
        |:-----:|:---:|:---:|
        |       |  1  |  0  |

        **Truth Table for `lN`:**

        | `not` |  U  |  X  |  0  |  1  |  Z  |  W  |  L  |  H  |  -  |
        |:-----:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
        |       |  U  |  X  |  1  |  0  |  X  |  X  |  1  |  0  |  X  |
    }];

    let arguments = (ins AnyTypeOf<[LLHD_iN, LLHD_lN]>:$value);
}

def LLHD_AndOp : LLHD_ArithmeticOrBitwiseOp<"and", [Commutative]> {
    let summary = "Bitwise AND";
    let description = [{
        Takes two integers of the same width or two nine-valued-logic (IEEE 1164)
        values of the same width as input. Calculates the bitwise AND. The 
        result is always of the exact same type as the two inputs.

        **Syntax:**
        ```
        and-op ::= ssa-id `=` `llhd.and` ssa-lhs `,` ssa-rhs attr-dict `:` type
        ```

        **Examples:**
        ```
        %0 = llhd.const 0 : i32
        %1 = llhd.and %0, %0 : i32 
        ```

        **Truth Table for `iN`:**

        | `and` |  0  |  1  |
        |:-----:|:---:|:---:|
        |   0   |  0  |  0  |
        |   1   |  0  |  1  |

        **Truth Table for `lN`:**

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
}

def LLHD_OrOp : LLHD_ArithmeticOrBitwiseOp<"or", [Commutative]> {
    let summary = "Bitwise OR";
    let description = [{
        Takes two integers of the same width or two nine-valued-logic (IEEE 1164)
        values of the same width as input. Calculates the bitwise OR. The 
        result is always of the exact same type as the two inputs.

        **Syntax:**
        ```
        or-op ::= ssa-id `=` `llhd.or` ssa-lhs `,` ssa-rhs attr-dict `:` type
        ```

        **Examples:**
        ```
        %0 = llhd.const 0 : i32
        %1 = llhd.or %0, %0 : i32 
        ```

        **Truth Table for `iN`:**

        | `or`  |  0  |  1  |
        |:-----:|:---:|:---:|
        |   0   |  0  |  0  |
        |   1   |  0  |  1  |

        **Truth Table for `lN`:**

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
}

def LLHD_XorOp : LLHD_ArithmeticOrBitwiseOp<"xor", [Commutative]> {
    let summary = "Bitwise XOR";
    let description = [{
        Takes two integers of the same width or two nine-valued-logic (IEEE 1164)
        values of the same width as input. Calculates the bitwise XOR. The 
        result is always of the exact same type as the two inputs.

        **Syntax:**
        ```
        xor-op ::= ssa-id `=` `llhd.xor` ssa-lhs `,` ssa-rhs attr-dict `:` type
        ```

        **Examples:**
        ```
        %0 = llhd.const 0 : i32
        %1 = llhd.xor %0, %0 : i32 
        ```

        **Truth Table for `iN`:**

        | `xor` |  0  |  1  |
        |:-----:|:---:|:---:|
        |   0   |  0  |  0  |
        |   1   |  0  |  1  |

        **Truth Table for `lN`:**

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
}

def LLHD_ShlOp : LLHD_Op<"shl", [NoSideEffect]> {
    let summary = "Shifts a value to the left by a given amount.";
    let description = [{
        The type of the base value and the hidden value must be the same, but
        may differ in the number of bits or elements. The result always has the
        same type (including width) of the base value.
        The instruction is transparent to signals and pointers. For example,
        passing a signal as argument will shift the underlying value and return 
        a signal to the shifted value.
        Allowed (underlying) types are signless integers, nine-valued-logic values
        and arrays. The shift amount has to be a signless integer. A shift amount
        bigger than the number of bits or elements of the hidden value is undefined.
        The hidden value is uncovered by non-zero shift amounts. E.g. consider
        the four bit values `base = 0xf`, `hidden = 0xc` shifted by an amount of
        three result in `0xe`.

        **Syntax:**
        ```
        shl-op ::= ssa-id `=` `llhd.shl` ssa-base `,` ssa-hidden `,` ssa-amount attr-dict `:` `(` base-type `,` hidden-type `,` amount-type `)` `->` result-type
        ```

        **Examples:**
        ```
        %0 = llhd.shl %base, %hidden, %amount : (i4, i2, i2) -> i4
        ```
    }];

    // TODO: adjust type T and Th to include arrays and pointers
    let arguments = (ins AnyTypeOf<[LLHD_iN, LLHD_lN, LLHD_SigType]>:$base, 
                         AnyTypeOf<[LLHD_iN, LLHD_lN, LLHD_SigType]>:$hidden,
                         AnySignlessInteger:$amount);
    let results = (outs AnyTypeOf<[LLHD_iN, LLHD_lN, LLHD_SigType]>:$result);

    let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";

    let verifier = [{ return ::verify(*this); }];
}

def LLHD_ShrOp : LLHD_Op<"shr", [NoSideEffect]> {
    let summary = "Shifts a value to the right by a given amount.";
    let description = [{
        The type of the base value and the hidden value must be the same, but
        may differ in the number of bits or elements. The result always has the
        same type (including width) of the base value.
        The instruction is transparent to signals and pointers. For example,
        passing a signal as argument will shift the underlying value and return 
        a signal to the shifted value.
        Allowed (underlying) types are signless integers, nine-valued-logic values
        and arrays. The shift amount has to be a signless integer. A shift amount
        bigger than the number of bits or elements of the hidden value is undefined.
        The hidden value is uncovered by non-zero shift amounts. E.g. consider
        the four bit values `base = 0xf`, `hidden = 0xc` shifted by an amount of
        three result in `0x9`.

        **Syntax:**
        ```
        shr-op ::= ssa-id `=` `llhd.shr` ssa-base `,` ssa-hidden `,` ssa-amount attr-dict `:` `(` base-type `,` hidden-type `,` amount-type `)` `->` result-type
        ```

        **Examples:**
        ```
        %0 = llhd.shr %base, %hidden, %amount : (i4, i2, i2) -> i4
        ```
    }];

    // TODO: adjust type T and Th to include arrays and pointers
    let arguments = (ins AnyTypeOf<[LLHD_iN, LLHD_lN, LLHD_SigType]>:$base, 
                         AnyTypeOf<[LLHD_iN, LLHD_lN, LLHD_SigType]>:$hidden,
                         AnySignlessInteger:$amount);
    let results = (outs AnyTypeOf<[LLHD_iN, LLHD_lN, LLHD_SigType]>:$result);

    let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";

    let verifier = [{ return ::verify(*this); }];
}


//===----------------------------------------------------------------------===//
//=== Arithmetic Operations
//===----------------------------------------------------------------------===//

def LLHD_NegOp : LLHD_ArithmeticOrBitwiseOp<"neg", []> {
    let summary = "Negate a value.";
    let description = [{
        The operand and result always have the same type. The type has to be a
        signless integer of any width. Although, only signless integers are
        allowed, this instruction applies two's complement negation of the
        integer, basically treating it as a signed integer.

        **Syntax:**
        ```
        neg-op ::= ssa-id `=` `llhd.neg` ssa-value attr-dict `:` type
        ```

        **Examples:**
        ```
        %0 = llhd.const 42 : i32
        %1 = llhd.neg %0 : i32
        ```
    }];

    let arguments = (ins LLHD_iN:$value);
}

def LLHD_SModOp : LLHD_ArithmeticOrBitwiseOp<"smod", []> {
    let summary = "Signed modulo.";
    let description = [{
        This instruction computes the signed modulo of two signless integers of
        any width, treating the leading bit as sign. The operand and result
        types always have to be the same.
        To calculate the signed remainder of two integers, use `remi_signed`
        from the standard dialect.

        **Syntax:**
        ```
        smod-op ::= ssa-id `=` `llhd.smod` ssa-lhs `,` ssa-rhs attr-dict `:` type
        ```

        **Examples:**
        ```
        %0 = llhd.const 9 : i4
        %1 = llhd.const 4 : i4
        %2 = llhd.smod %0, %1 : i4
        ```
    }];

    let arguments = (ins LLHD_iN:$lhs, LLHD_iN:$rhs);
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
