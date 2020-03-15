#ifndef LLHD_BASE
#define LLHD_BASE

#ifndef OP_BASE
include "mlir/IR/OpBase.td"
#endif // OP_BASE

//===----------------------------------------------------------------------===//
// LLHD dialect definition
//===----------------------------------------------------------------------===//

def LLHD_Dialect : Dialect {
    let name = "llhd";

    let description = [{
        A low-level hardware description dialect in MLIR.
    }];

    let cppNamespace = "llhd";
}

//===----------------------------------------------------------------------===//
// LLHD type definitions
//===----------------------------------------------------------------------===//
// LLHD sig type
def LLHD_SigType : Type<CPred<"$_self.isa<SigType>()">, "LLHD sig type">;

// LLHD elementary typse
def LLHD_ElementType : AnyTypeOf<[I64]>;

// LLHD full list of types
def LLHD_Type : AnyTypeOf<[LLHD_ElementType, LLHD_SigType]>;

//===----------------------------------------------------------------------===//
// LLHD op definition
//===----------------------------------------------------------------------===//

// Base class for all LLHD ops.
class LLHD_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<LLHD_Dialect, mnemonic, traits> {

    // For each LLHD op, the following static functions need to be defined in
    // LLHDOps.cpp:
    //
    // * static ParseResult parse<op-c++-class-name>(OpAsmParser &parser,
    //                                               OperationState &state);
    // * static void print(OpAsmPrinter &p, <op-c++-class-name> op)
    // * static LogicalResult verify(<op-c++-class-name> op)
    let parser = [{ return ::parse$cppClass(parser, result); }];
    let printer = [{ ::print(p, *this); }];
    let verifier = [{ return ::verify(*this); }];
}


#endif // LLHD_BASE
