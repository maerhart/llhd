#include "Dialect/LLHD/LLHDOps.h"
#include "Dialect/LLHD/LLHDDialect.h"
#include "llvm/ADT/ArrayRef.h"
// #include "llvm/ADT/STLExtras.h"
// #include "llvm/ADT/iterator_range.h"
// #include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Block.h"
// #include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/FunctionImplementation.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/Functional.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/STLExtras.h"
// #include <bits/stdint-intn.h>
#include <cstddef>
#include <functional>
#include <iterator>
//#include <llvm-9/llvm/ADT/STLExtras.h>

using namespace mlir;

//===---------------------------------------------------------------------===//
// LLHD Operations
//===---------------------------------------------------------------------===//

// ConstOp

/// Parse an LLHD const operation with following syntax:
/// op ::= llhd.const literal : type
// static ParseResult parseConstOp(OpAsmParser &parser, OperationState &result)
// {
//     IntegerAttr value;
//     Type type;
//     if (parser.parseAttribute(value, parser.getBuilder().getIntegerType(64),
//                               "value", result.attributes) ||
//         parser.parseColonType(type))
//         return failure();

//     // auto res = type.dyn_cast<IntegerType>();
//     // if (!res) return failure();
//     result.addTypes(type);
//     return success();
// }

// /// print an LLHD const operation
// static void print(OpAsmPrinter &printer, llhd::ConstOp op) {
//     printer << op.getOperationName() << " ";
//     printer << op.value() << " : ";
//     printer.printType(op.getType());
// }

static LogicalResult verify(llhd::ConstOp op) { return success(); }

// Sig Op

/// Parse an LLHD sig operation with the following syntax:
/// op ::= llhd.sig %operand : type
static ParseResult parseSigOp(OpAsmParser &parser, OperationState &result) {
    OpAsmParser::OperandType operand;
    Type sigType;
    if (parser.parseOperand(operand) || parser.parseColonType(sigType) ||
        parser.parseOptionalAttrDict(result.attributes))
        return failure();

    if (parser.resolveOperand(operand, sigType, result.operands))
        return failure();

    result.addTypes(llhd::SigType::get(sigType));

    return success();
}

/// Print an LLHD sig operation
static void print(OpAsmPrinter &printer, llhd::SigOp op) {
    // get the resulting signal type
    llhd::SigType opType = op.getType().dyn_cast<llhd::SigType>();
    printer << op.getOperationName() << " ";
    printer.printOperand(op.init());
    printer << " : ";
    printer.printType(opType.getUnderlyingType());
}

static LogicalResult verify(llhd::SigOp op) { return success(); }

// Prb Op

/// Parse an LLHD prb operation with the following syntax:
/// op ::= llhd.prb %operand : !llhd.sig<type>
static ParseResult parsePrbOp(OpAsmParser &parser, OperationState &result) {
    llhd::SigType sigType;
    OpAsmParser::OperandType operand;
    if (parser.parseOperand(operand) || parser.parseColonType(sigType))
        return failure();
    if (parser.resolveOperand(operand, sigType, result.operands))
        return failure();

    result.addTypes(sigType.getUnderlyingType());
    return success();
}

/// Print an LLHD prb operation
static void print(OpAsmPrinter &printer, llhd::PrbOp op) {
    printer << op.getOperationName() << " ";
    printer.printOperand(op.signal());
    printer << " : ";
    printer.printType(llhd::SigType::get(op.getType()));
}

static LogicalResult verify(llhd::PrbOp op) { return success(); }

// Drv Op

/// Parse an LLHD drv operation with the following syntax:
/// op ::= llhd.drv %signal, %value : !llhd.sig<type>
static ParseResult parseDrvOp(OpAsmParser &parser, OperationState &result) {
    llvm::SmallVector<OpAsmParser::OperandType, 2> operands;
    llhd::SigType sigType;
    if (parser.parseOperandList(operands, 2) || parser.parseColon() ||
        parser.parseType(sigType))
        return failure();
    if (parser.resolveOperand(operands[0], sigType, result.operands) ||
        parser.resolveOperand(operands[1], sigType.getUnderlyingType(),
                              result.operands))
        return failure();

    return success();
}

/// Print an LLHD drv operation
static void print(OpAsmPrinter &printer, llhd::DrvOp op) {
    printer << op.getOperationName() << " ";
    printer << op.getOperands() << " : ";
    printer.printType(op.signal().getType());
}

static LogicalResult verify(llhd::DrvOp op) {
    Type opType = op.value().getType();
    llhd::SigType sigType = op.signal().getType().dyn_cast<llhd::SigType>();

    if (!sigType) {
        op.emitError("Expected signal type, got ") << op.signal().getType();
        return failure();
    }
    if (opType != sigType.getUnderlyingType()) {
        op.emitError("The operand to drive has to be the same type as the "
                     "signal's underlying type, got ")
            << opType;
    }

    return success();
}

// Entity Op

/// Parse an argument list of an entity operation.
/// The argument list and argument types are returned in args and argTypes
/// respectively.
static ParseResult
parseArgumentList(OpAsmParser &parser,
                  SmallVector<OpAsmParser::OperandType, 4> &args,
                  SmallVector<Type, 4> &argTypes) {
    if (parser.parseLParen())
        return failure();
    do {
        OpAsmParser::OperandType argument;
        Type argType;
        if (succeeded(parser.parseOptionalRegionArgument(argument))) {
            if (!argument.name.empty() &&
                succeeded(parser.parseColonType(argType))) {
                args.push_back(argument);
                argTypes.push_back(argType);
            }
        }
    } while (succeeded(parser.parseOptionalComma()));
    if (parser.parseRParen())
        return failure();

    return success();
}

/// parse an entity signature with syntax:
/// (%arg0 : T0, %arg1 : T1, <...>) -> (%out0 : T0, %out1 : T1, <...>)
static ParseResult
parseEntitySignature(OpAsmParser &parser, OperationState &result,
                     SmallVector<OpAsmParser::OperandType, 4> &args,
                     SmallVector<Type, 4> &argTypes) {
    if (parseArgumentList(parser, args, argTypes))
        return failure();
    // create integer attribute for number of inputs. Take directly from the
    // size of the argument list parsed so far.
    IntegerAttr insAttr = IntegerAttr::get(
        IntegerType::get(64, result.getContext()), args.size());
    result.addAttribute("ins", insAttr);
    if (parser.parseArrow() || parseArgumentList(parser, args, argTypes))
        return failure();

    return success();
}

static ParseResult parseEntityOp(OpAsmParser &parser, OperationState &result) {
    StringAttr entityName;
    SmallVector<OpAsmParser::OperandType, 4> args;
    SmallVector<Type, 4> argTypes;

    if (parser.parseSymbolName(entityName, SymbolTable::getSymbolAttrName(),
                               result.attributes))
        return failure();

    parseEntitySignature(parser, result, args, argTypes);

    auto *body = result.addRegion();
    parser.parseOptionalRegion(*body, args, argTypes);
    llhd::EntityOp::ensureTerminator(*body, parser.getBuilder(),
                                     result.location);
    return success();
}

static void printArgumentList(OpAsmPrinter &printer,
                              std::vector<BlockArgument> args) {
    printer << "(";
    for (size_t i = 0; i < args.size(); i++) {
        printer << args[i] << " : ";
        printer.printType(args[i].getType());
        if (i < args.size() - 1)
            printer << ", ";
    }
    printer << ")";
}

static void print(OpAsmPrinter &printer, llhd::EntityOp op) {
    std::vector<BlockArgument> ins, outs;
    int64_t n_ins = op.insAttr().getInt();
    for (int64_t i = 0; i < op.body().front().getArguments().size(); ++i) {
        // no furter verification for the attribute type is required, already
        // handled by verify.
        if (i < n_ins) {
            ins.push_back(op.body().front().getArguments()[i]);
        } else {
            outs.push_back(op.body().front().getArguments()[i]);
        }
    }
    auto entityName =
        op.getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName())
            .getValue();
    printer << op.getOperationName() << " ";
    printer.printSymbolName(entityName);
    printer << " ";
    printArgumentList(printer, ins);
    printer << " -> ";
    printArgumentList(printer, outs);
    printer.printRegion(op.body(), false, false);
}

static LogicalResult verify(llhd::EntityOp op) {
    Block &body = op.body().front();
    int64_t nIns = op.insAttr().getInt();
    // check that there is exactly one flag for each argument
    if (body.getArguments().size() < nIns) {
        op.emitError(
            "Cannot have more inputs than arguments, expected at most ")
            << body.getArguments().size() << " but got: " << nIns;
        return failure();
    }
    return success();
}

// Bitwise Operations
static LogicalResult verify(llhd::NotOp op) { return success(); }
static LogicalResult verify(llhd::AndOp op) { return success(); }
static LogicalResult verify(llhd::OrOp op) { return success(); }
static LogicalResult verify(llhd::XorOp op) { return success(); }

static LogicalResult verify(llhd::ShlOp op) {
    if (op.base().getType() != op.result().getType()) {
        op.emitError("The output of the Shl operation is required to have the "
                     "same type as the base value (first operand), (")
            << op.base().getType() << " vs. " << op.result().getType() << ")";
        return failure();
    }

    // TODO: verify that T and Th only differ in the number of bits or elements

    return success();
}

static LogicalResult verify(llhd::ShrOp op) {
    if (op.base().getType() != op.result().getType()) {
        op.emitError("The output of the Shl operation is required to have the "
                     "same type as the base value (first operand), (")
            << op.base().getType() << " vs. " << op.result().getType() << ")";
        return failure();
    }

    // TODO: verify that T and Th only differ in the number of bits or elements

    return success();
}

// Arithmetic Operations
static LogicalResult verify(llhd::NegOp op) { return success(); }
static LogicalResult verify(llhd::AddOp op) { return success(); }
static LogicalResult verify(llhd::SubOp op) { return success(); }
static LogicalResult verify(llhd::SMulOp op) { return success(); }
static LogicalResult verify(llhd::UMulOp op) { return success(); }

namespace mlir {
namespace llhd {
#define GET_OP_CLASSES
#include "Dialect/LLHD/LLHDOps.cpp.inc"
}    // namespace llhd
}    // namespace mlir
