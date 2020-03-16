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
/// op ::= llhd.const type literal
static ParseResult parseConstOp(OpAsmParser &parser, OperationState &result) {
    IntegerAttr value;
    Type type;
    if (parser.parseType(type) ||
        parser.parseAttribute(value, "value", result.attributes))
        return failure();

    // auto res = type.dyn_cast<IntegerType>();
    // if (!res) return failure();
    result.addTypes(type);
    return success();
}

/// print an LLHD const operation
static void print(OpAsmPrinter &printer, llhd::ConstOp op) {
    printer << op.getOperationName() << " ";
    printer.printType(op.getType());
    printer << " " << op.value();
}

static LogicalResult verify(llhd::ConstOp op) { return success(); }

// Sig Op

/// Parse an LLHD sig operation with the following syntax:
/// op ::= llhd.sig type %operand
static ParseResult parseSigOp(OpAsmParser &parser, OperationState &result) {
    OpAsmParser::OperandType operand;
    Type sigType;
    if (parser.parseType(sigType) || parser.parseOperand(operand) ||
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
    printer.printType(opType.getUnderlyingType());
    printer << " ";
    printer.printOperand(op.init());
}

static LogicalResult verify(llhd::SigOp op) { return success(); }

// Prb Op

/// Parse an LLHD prb operation with the following syntax:
/// op ::= llhd.prb !llhd.sig<type> %operand
static ParseResult parsePrbOp(OpAsmParser &parser, OperationState &result) {
    llhd::SigType sigType;
    OpAsmParser::OperandType operand;
    if (parser.parseType(sigType) || parser.parseOperand(operand))
        return failure();
    if (parser.resolveOperand(operand, sigType, result.operands))
        return failure();

    result.addTypes(sigType.getUnderlyingType());
    return success();
}

/// Print an LLHD prb operation
static void print(OpAsmPrinter &printer, llhd::PrbOp op) {
    printer << op.getOperationName() << " ";
    printer.printType(llhd::SigType::get(op.getType()));
    printer << " ";
    printer.printOperand(op.signal());
}

static LogicalResult verify(llhd::PrbOp op) { return success(); }

// Drv Op

/// Parse an LLHD drv operation with the following syntax:
/// op ::= llhd.drv !llhd.sig<type> %signal, %value
static ParseResult parseDrvOp(OpAsmParser &parser, OperationState &result) {
    llvm::SmallVector<OpAsmParser::OperandType, 2> operands;
    llhd::SigType sigType;
    if (parser.parseType(sigType) || parser.parseOperandList(operands, 2))
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
    printer.printType(op.signal().getType());
    printer << " " << op.getOperands();
}

static LogicalResult verify(llhd::DrvOp op) { return success(); }

namespace mlir {
namespace llhd {
#define GET_OP_CLASSES
#include "Dialect/LLHD/LLHDOps.cpp.inc"
}    // namespace llhd
}    // namespace mlir
