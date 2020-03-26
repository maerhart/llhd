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

// Wait Terminator
static ParseResult parseWaitOp(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::OperandType, 4> obsOperands;
  llvm::SMLoc obsOperandsLoc = parser.getCurrentLocation();
  (void)obsOperandsLoc;
  SmallVector<OpAsmParser::OperandType, 4> destOpsOperands;
  llvm::SMLoc destOpsOperandsLoc = parser.getCurrentLocation();
  (void)destOpsOperandsLoc;
  SmallVector<Type, 1> destOpsTypes;
  SmallVector<Type, 1> obsTypes;

  if (parser.parseOperandList(obsOperands))
    return failure();
  if (parser.parseLSquare())
    return failure();

  Block *destSuccessor = nullptr;
  if (parser.parseSuccessor(destSuccessor))
    return failure();
  if (succeeded(parser.parseOptionalLParen())) {

  if (parser.parseOperandList(destOpsOperands))
    return failure();
  if (parser.parseColon())
    return failure();

  if (parser.parseTypeList(destOpsTypes))
    return failure();
  if (parser.parseRParen())
    return failure();
  }
  if (parser.parseRSquare())
    return failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();
  if (parser.parseColon())
    return failure();

  if (parser.parseTypeList(obsTypes))
    return failure();
  if (parser.resolveOperands(obsOperands, obsTypes, obsOperandsLoc, result.operands))
    return failure();
  if (parser.resolveOperands(destOpsOperands, destOpsTypes, destOpsOperandsLoc, result.operands))
    return failure();
  result.addSuccessors(destSuccessor);
  result.addAttribute("operand_segment_sizes", parser.getBuilder().getI32VectorAttr({static_cast<int32_t>(obsOperands.size()), static_cast<int32_t>(destOpsOperands.size())}));
  return success();
}

static void print(OpAsmPrinter &p, llhd::WaitOp op) {
  p << "llhd.wait";
  p << " ";
  p << op.obs();
  p << " ";
  p << "[";
  p << op.dest();
  if (!op.destOps().empty()) {
  p << "(";
  p << op.destOps();
  p << " " << ":";
  p << " ";
  p << op.destOps().getTypes();
  p << ")";
  }
  p << "]";
  p.printOptionalAttrDict(op.getAttrs(), /*elidedAttrs=*/{"operand_segment_sizes", });
  p << " " << ":";
  p << " ";
  p << op.obs().getTypes();
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
      if (!argument.name.empty() && succeeded(parser.parseColonType(argType))) {
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
  IntegerAttr insAttr =
      IntegerAttr::get(IntegerType::get(64, result.getContext()), args.size());
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
  llhd::EntityOp::ensureTerminator(*body, parser.getBuilder(), result.location);
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
      op.getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName()).getValue();
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
    op.emitError("Cannot have more inputs than arguments, expected at most ")
        << body.getArguments().size() << " but got: " << nIns;
    return failure();
  }
  return success();
}

// Proc Operation
LogicalResult mlir::llhd::ProcOp::verifyType() {
  // Fail if function returns more than zero values. This is because the outputs
  // of a process are specially marked arguments.
  if (this->getNumResults() > 0) {
    this->emitOpError(
        "process has more than zero return types, this is not allowed");
    return failure();
  }
  // Check that all operands are of signal type
  for (int i = 0, n = this->getNumFuncArguments(); i < n; i++) {
    if (!llhd::SigType::kindof(this->getArgument(i).getType().getKind())) {
      this->emitOpError("usage of invalid argument type, was ")
          << this->getArgument(i).getType() << ", expected LLHD signal type";
      return failure();
    }
  }
  return success();
}

LogicalResult mlir::llhd::ProcOp::verifyBody() {
  // Body must not be empty, this indicates an external process. We use another
  // instruction to reference external processes.
  if (this->isExternal()) {
    this->emitOpError("defining external processes with the proc instruction "
                      "is not allowed, use the intended instruction instead.");
    return failure();
  }
  return success();
}

static LogicalResult verify(llhd::ProcOp op) {
  // Check that the ins attribute is smaller or equal the number of
  // arguments
  uint64_t numArgs = op.getNumArguments();
  uint64_t numIns = op.insAttr().getInt();
  if (numArgs < numIns) {
    op.emitOpError("Cannot have more inputs than arguments, expected at most ")
        << numArgs << ", got " << numIns;
    return failure();
  }
  return success();
}

static ParseResult
parseProcArgumentList(OpAsmParser &parser, SmallVectorImpl<Type> &argTypes,
                      SmallVectorImpl<OpAsmParser::OperandType> &argNames) {
  if (parser.parseLParen())
    return failure();

  // The argument list either has to consistently have ssa-id's followed by
  // types, or just be a type list.  It isn't ok to sometimes have SSA ID's and
  // sometimes not.
  auto parseArgument = [&]() -> ParseResult {
    llvm::SMLoc loc = parser.getCurrentLocation();

    // Parse argument name if present.
    OpAsmParser::OperandType argument;
    Type argumentType;
    if (succeeded(parser.parseOptionalRegionArgument(argument)) &&
        !argument.name.empty()) {
      // Reject this if the preceding argument was missing a name.
      if (argNames.empty() && !argTypes.empty())
        return parser.emitError(loc, "expected type instead of SSA identifier");
      argNames.push_back(argument);

      if (parser.parseColonType(argumentType))
        return failure();
    } else if (!argNames.empty()) {
      // Reject this if the preceding argument had a name.
      return parser.emitError(loc, "expected SSA identifier");
    } else if (parser.parseType(argumentType)) {
      return failure();
    }

    // Add the argument type.
    argTypes.push_back(argumentType);

    return success();
  };

  // Parse the function arguments.
  if (failed(parser.parseOptionalRParen())) {
    do {
      unsigned numTypedArguments = argTypes.size();
      if (parseArgument())
        return failure();

      llvm::SMLoc loc = parser.getCurrentLocation();
      if (argTypes.size() == numTypedArguments &&
          succeeded(parser.parseOptionalComma()))
        return parser.emitError(loc, "variadic arguments are not allowed");
    } while (succeeded(parser.parseOptionalComma()));
    parser.parseRParen();
  }

  return success();
}

static ParseResult parseProcOp(OpAsmParser &parser, OperationState &result) {
  StringAttr procName;
  SmallVector<OpAsmParser::OperandType, 8> argNames;
  SmallVector<Type, 8> argTypes;
  Builder &builder = parser.getBuilder();

  if (parser.parseSymbolName(procName, SymbolTable::getSymbolAttrName(),
                             result.attributes))
    return failure();

  if (parseProcArgumentList(parser, argTypes, argNames))
    return failure();

  result.addAttribute("ins", builder.getI64IntegerAttr(argTypes.size()));
  parser.parseArrow();

  if (parseProcArgumentList(parser, argTypes, argNames))
    return failure();

  auto type = builder.getFunctionType(argTypes, llvm::None);
  result.addAttribute(mlir::llhd::ProcOp::getTypeAttrName(),
                      TypeAttr::get(type));

  auto *body = result.addRegion();
  parser.parseRegion(*body, argNames, argNames.empty() ? ArrayRef<Type>() : argTypes);

  return success();
}

/// Print the signature of the `proc` unit. Assumes that it passed the
/// verification.
static void printProcArguments(OpAsmPrinter &p, Operation *op,
                               ArrayRef<Type> types, uint64_t numIns) {
  Region &body = op->getRegion(0);
  auto printList = [&](unsigned i, unsigned max) -> void {
    for (; i < max; ++i) {
      p.printOperand(body.front().getArgument(i));
      p << " : ";

      p.printType(types[i]);
      p.printOptionalAttrDict(::mlir::impl::getArgAttrs(op, i));

      if (i < max - 1)
        p << ", ";
    }
  };

  p << '(';
  printList(0, numIns);
  p << ") -> (";
  printList(numIns, types.size());
  p << ')';
}

static void print(OpAsmPrinter &printer, llhd::ProcOp op) {
  FunctionType type = op.getType();
  printer << op.getOperationName() << ' ';
  printer.printSymbolName(op.getName());
  printProcArguments(printer, op.getOperation(), type.getInputs(),
                     op.insAttr().getInt());
  printer.printRegion(op.body(), false, true);
}

// Shift Operations
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
    op.emitError("The output of the Shr operation is required to have the "
                 "same type as the base value (first operand), (")
        << op.base().getType() << " vs. " << op.result().getType() << ")";
    return failure();
  }

  // TODO: verify that T and Th only differ in the number of bits or elements

  return success();
}

namespace mlir {
namespace llhd {
#define GET_OP_CLASSES
#include "Dialect/LLHD/LLHDOps.cpp.inc"
} // namespace llhd
} // namespace mlir
