#ifndef LLHD_BASE
#define LLHD_BASE

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// LLHD dialect definition
//===----------------------------------------------------------------------===//

def LLHD_Dialect : Dialect {
    let name = "llhd";

    let description = [{
        A low-level hardware description dialect in MLIR.
    }];

    let cppNamespace = "llhd";

    let hasConstantMaterializer = 1;
}

//===----------------------------------------------------------------------===//
// LLHD type definitions
//===----------------------------------------------------------------------===//

// LLHD sig type
def LLHD_SigType : Type<CPred<"$_self.isa<SigType>()">, "LLHD sig type">;

// LLHD Time Type
def LLHD_TimeType : Type<CPred<"$_self.isa<TimeType>()">, "LLHD time type">;

//===----------------------------------------------------------------------===//
// LLDH attribute definitions
//===----------------------------------------------------------------------===//

// LLHD time attr
def LLHD_TimeAttr : Attr<CPred<"$_self.isa<TimeAttr>()">, "LLHD time attribute"> {
    let storageType= [{ TimeAttr }];
    let returnType = [{ llvm::ArrayRef<unsigned> }];
    let valueType = LLHD_TimeType;
}

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
    let parser = [{ return ::parse$cppClass(parser, result); }];
    let printer = [{ ::print(p, *this); }];
}

class LLHD_ArithmeticOrBitwiseOp<string mnemonic, list<OpTrait> traits = []> :
    Op<LLHD_Dialect, mnemonic, 
        !listconcat(traits, [NoSideEffect, SameOperandsAndResultType])> {

    let results = (outs AnySignlessInteger);

    let parser = [{
        return impl::parseOneResultSameOperandTypeOp(parser, result);
    }];

    let printer = [{
        impl::printOneResultOp(this->getOperation(), p);
    }];
}


#endif // LLHD_BASE
