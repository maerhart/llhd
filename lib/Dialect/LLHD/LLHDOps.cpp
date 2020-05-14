#include "Dialect/LLHD/LLHDOps.h"
#include "Dialect/LLHD/LLHDDialect.h"
#include "mlir/Dialect/CommonFolders.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"

using namespace mlir;

//===---------------------------------------------------------------------===//
// LLHD Operations
//===---------------------------------------------------------------------===//

// Const Op

static ParseResult parseConstOp(OpAsmParser &parser, OperationState &result) {
  Attribute val;
  Type type;
  if (parser.parseAttribute(val, "value", result.attributes) ||
      parser.parseOptionalAttrDict(result.attributes))
    return failure();
  type = val.getType();
  return parser.addTypeToList(val.getType(), result.types);
}

static void print(OpAsmPrinter &printer, llhd::ConstOp op) {
  printer << op.getOperationName() << " ";
  // The custom time attribute is not printing the attribute type by default for
  // some reason. Work around by printing the attribute without type, explicitly
  // followed by the operation type
  printer.printAttributeWithoutType(op.valueAttr());
  printer.printOptionalAttrDict(op.getAttrs(), {"value"});
  printer << " : ";
  printer.printType(op.getType());
}

OpFoldResult llhd::ConstOp::fold(ArrayRef<Attribute> operands) {
  assert(operands.empty() && "const has no operands");
  return value();
}

//===----------------------------------------------------------------------===//
// NegOp
//===----------------------------------------------------------------------===//

OpFoldResult llhd::NegOp::fold(ArrayRef<Attribute> operands) {
  return constFoldUnaryOp<IntegerAttr>(operands, [](APInt a) { return -a; });
}

//===----------------------------------------------------------------------===//
// SModOp
//===----------------------------------------------------------------------===//

OpFoldResult llhd::SModOp::fold(ArrayRef<Attribute> operands) {
  /// llhd.smod(x, 1) -> 0
  if (matchPattern(rhs(), m_One()))
    return Builder(getContext()).getZeroAttr(getType());

  /// llhd.smod(0, x) -> 0
  if (matchPattern(lhs(), m_Zero()))
    return Builder(getContext()).getZeroAttr(getType());

  /// llhs.smod(x,x) -> 0
  if (lhs() == rhs())
    return Builder(getContext()).getZeroAttr(getType());

  return constFoldBinaryOp<IntegerAttr>(operands, [](APInt lhs, APInt rhs) {
    APInt result = lhs.srem(rhs);
    if ((lhs.isNegative() && rhs.isNonNegative()) ||
        (lhs.isNonNegative() && rhs.isNegative())) {
      result += rhs;
    }
    return result;
  });
}

//===----------------------------------------------------------------------===//
// NotOp
//===----------------------------------------------------------------------===//

OpFoldResult llhd::NotOp::fold(ArrayRef<Attribute> operands) {
  return constFoldUnaryOp<IntegerAttr>(operands, [](APInt a) { return ~a; });
}

//===----------------------------------------------------------------------===//
// AndOp
//===----------------------------------------------------------------------===//

OpFoldResult llhd::AndOp::fold(ArrayRef<Attribute> operands) {
  /// llhd.and(x, 0) -> 0
  if (matchPattern(rhs(), m_Zero()))
    return rhs();

  /// llhd.and(x, all_bits_set) -> x
  if (matchPattern(rhs(), constant_int_all_ones_matcher()))
    return lhs();

  // llhd.and(x, x) -> x
  if (rhs() == lhs())
    return rhs();

  return constFoldBinaryOp<IntegerAttr>(operands,
                                        [](APInt a, APInt b) { return a & b; });
}

//===----------------------------------------------------------------------===//
// OrOp
//===----------------------------------------------------------------------===//

OpFoldResult llhd::OrOp::fold(ArrayRef<Attribute> operands) {
  /// llhd.or(x, 0) -> x
  if (matchPattern(rhs(), m_Zero()))
    return lhs();

  /// llhd.or(x, all_bits_set) -> all_bits_set
  if (matchPattern(rhs(), constant_int_all_ones_matcher()))
    return rhs();

  // llhd.or(x, x) -> x
  if (rhs() == lhs())
    return rhs();

  return constFoldBinaryOp<IntegerAttr>(operands,
                                        [](APInt a, APInt b) { return a | b; });
}

//===----------------------------------------------------------------------===//
// XorOp
//===----------------------------------------------------------------------===//

OpFoldResult llhd::XorOp::fold(ArrayRef<Attribute> operands) {
  /// llhd.xor(x, 0) -> x
  if (matchPattern(rhs(), m_Zero()))
    return lhs();

  /// llhs.xor(x,x) -> 0
  if (lhs() == rhs())
    return Builder(getContext()).getZeroAttr(getType());

  return constFoldBinaryOp<IntegerAttr>(operands,
                                        [](APInt a, APInt b) { return a ^ b; });
}

//===----------------------------------------------------------------------===//
// ShlOp
//===----------------------------------------------------------------------===//

OpFoldResult llhd::ShlOp::fold(ArrayRef<Attribute> operands) {
  /// llhd.shl(base, hidden, 0) -> base
  if (matchPattern(amount(), m_Zero()))
    return base();

  return constFoldTernaryOp<IntegerAttr>(
      operands, [](APInt base, APInt hidden, APInt amt) {
        base <<= amt;
        base += hidden.getHiBits(amt.getZExtValue());
        return base;
      });
}

//===----------------------------------------------------------------------===//
// ShrOp
//===----------------------------------------------------------------------===//

OpFoldResult llhd::ShrOp::fold(ArrayRef<Attribute> operands) {
  /// llhd.shl(base, hidden, 0) -> base
  if (matchPattern(amount(), m_Zero()))
    return base();

  return constFoldTernaryOp<IntegerAttr>(
      operands, [](APInt base, APInt hidden, APInt amt) {
        base = base.getHiBits(base.getBitWidth() - amt.getZExtValue());
        hidden = hidden.getLoBits(amt.getZExtValue());
        hidden <<= base.getBitWidth() - amt.getZExtValue();
        return base + hidden;
      });
}

// Wait Terminator

// Implement this operation for the BranchOpInterface
Optional<MutableOperandRange>
llhd::WaitOp::getMutableSuccessorOperands(unsigned index) {
  assert(index == 0 && "invalid successor index");
  return destOpsMutable();
}

// Entity Op

/// Parse an argument list of an entity operation.
/// The argument list and argument types are returned in args and argTypes
/// respectively.
static ParseResult
parseArgumentList(OpAsmParser &parser,
                  SmallVectorImpl<OpAsmParser::OperandType> &args,
                  SmallVectorImpl<Type> &argTypes) {
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
                     SmallVectorImpl<OpAsmParser::OperandType> &args,
                     SmallVectorImpl<Type> &argTypes) {
  if (parseArgumentList(parser, args, argTypes))
    return failure();
  // create the integer attribute with the number of inputs.
  IntegerAttr insAttr = parser.getBuilder().getI64IntegerAttr(args.size());
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

  if (parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return failure();

  auto type = parser.getBuilder().getFunctionType(argTypes, llvm::None);
  result.addAttribute(mlir::llhd::EntityOp::getTypeAttrName(),
                      TypeAttr::get(type));

  auto *body = result.addRegion();
  parser.parseRegion(*body, args, argTypes);
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
  printer.printOptionalAttrDictWithKeyword(
      op.getAttrs(),
      /*elidedAttrs =*/{SymbolTable::getSymbolAttrName(),
                        llhd::EntityOp::getTypeAttrName(), "ins"});
  printer.printRegion(op.body(), false, false);
}

static LogicalResult verify(llhd::EntityOp op) {
  Block &body = op.body().front();
  int64_t nIns = op.insAttr().getInt();
  // check that there is at most one flag for each argument
  if (body.getArguments().size() < nIns) {
    op.emitError("Cannot have more inputs than arguments, expected at most ")
        << body.getArguments().size() << " but got: " << nIns;
    return failure();
  }
  return success();
}

LogicalResult mlir::llhd::EntityOp::verifyType() {
  // Fail if function returns any values. An entity's outputs are specially
  // marked arguments.
  if (this->getNumResults() > 0) {
    this->emitOpError("an entity cannot have return types.");
    return failure();
  }
  // Check that all operands are of signal type
  for (int i = 0, n = this->getNumFuncArguments(); i < n; i++) {
    if (!llhd::SigType::kindof(this->getArgument(i).getType().getKind())) {
      this->emitOpError("usage of invalid argument type. Got ")
          << this->getArgument(i).getType() << ", expected LLHD signal type";
      return failure();
    }
  }
  return success();
}

LogicalResult mlir::llhd::EntityOp::verifyBody() {
  // Body must not be empty.
  if (this->isExternal()) {
    this->emitOpError("defining external entity with the entity instruction "
                      "is not allowed, use the intended instruction instead.");
    return failure();
  }

  // check signal names are unique
  llvm::StringMap<bool> sigMap;
  auto walkResult = walk([&sigMap](Operation *op) -> WalkResult {
    if (auto sigOp = dyn_cast<SigOp>(op)) {
      if (sigMap[sigOp.name()]) {
        return sigOp.emitError("Redefinition of signal named '")
               << sigOp.name() << "'!";
      }
      sigMap.insert_or_assign(sigOp.name(), true);
    }
    return WalkResult::advance();
  });

  return failure(walkResult.wasInterrupted());
}

Region *llhd::EntityOp::getCallableRegion() {
  return isExternal() ? nullptr : &getBody();
}

ArrayRef<Type> llhd::EntityOp::getCallableResults() {
  return getType().getResults();
}

// Proc Operation
LogicalResult mlir::llhd::ProcOp::verifyType() {
  // Fail if function returns more than zero values. This is because the
  // outputs of a process are specially marked arguments.
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
  // Body must not be empty, this indicates an external process. We use
  // another instruction to reference external processes.
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
  // types, or just be a type list.  It isn't ok to sometimes have SSA ID's
  // and sometimes not.
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
  parser.parseRegion(*body, argNames,
                     argNames.empty() ? ArrayRef<Type>() : argTypes);

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

Region *llhd::ProcOp::getCallableRegion() {
  return isExternal() ? nullptr : &getBody();
}

ArrayRef<Type> llhd::ProcOp::getCallableResults() {
  return getType().getResults();
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

static LogicalResult verify(llhd::InstOp op) {
  // Check that the callee attribute was specified.
  auto calleeAttr = op.getAttrOfType<FlatSymbolRefAttr>("callee");
  if (!calleeAttr)
    return op.emitOpError("requires a 'callee' symbol reference attribute");
  auto proc = op.getParentOfType<ModuleOp>().lookupSymbol<llhd::ProcOp>(
      calleeAttr.getValue());
  auto entity = op.getParentOfType<ModuleOp>().lookupSymbol<llhd::EntityOp>(
      calleeAttr.getValue());

  // Verify that the input and output types match the callee.
  if (proc) {
    auto type = proc.getType();

    if (proc.ins() != op.inputs().size())
      return op.emitOpError(
          "incorrect number of inputs for proc instantiation");

    if (type.getNumInputs() != op.getNumOperands())
      return op.emitOpError(
          "incorrect number of outputs for proc instantiation");

    for (unsigned i = 0, e = type.getNumInputs(); i != e; ++i)
      if (op.getOperand(i).getType() != type.getInput(i))
        return op.emitOpError("operand type mismatch");

    return success();
  } else if (entity) {
    auto type = entity.getType();

    if (entity.ins() != op.inputs().size())
      return op.emitOpError(
          "incorrect number of inputs for entity instantiation");

    if (type.getNumInputs() != op.getNumOperands())
      return op.emitOpError(
          "incorrect number of outputs for entity instantiation");

    for (unsigned i = 0, e = type.getNumInputs(); i != e; ++i)
      if (op.getOperand(i).getType() != type.getInput(i))
        return op.emitOpError("operand type mismatch");

    return success();
  } else {
    return op.emitOpError() << "'" << calleeAttr.getValue()
                            << "' does not reference a valid proc or entity";
  }
}

FunctionType llhd::InstOp::getCalleeType() {
  SmallVector<Type, 8> argTypes(getOperandTypes());
  return FunctionType::get(argTypes, ArrayRef<Type>(), getContext());
}

static ParseResult parseRegOp(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::OperandType init;
  Type initType;
  SmallVector<OpAsmParser::OperandType, 8> valueOperands;
  SmallVector<OpAsmParser::OperandType, 8> triggerOperands;
  SmallVector<OpAsmParser::OperandType, 8> gateOperands;
  SmallVector<Type, 8> valueTypes;
  SmallVector<Type, 8> triggerTypes;
  SmallVector<Type, 8> gateTypes;
  llvm::SmallVector<int64_t, 8> modesArray;
  llvm::SmallVector<int64_t, 8> gateMask;
  int64_t gateCount = 0;

  if (parser.parseOperand(init))
    return failure();
  while (succeeded(parser.parseOptionalComma())) {
    OpAsmParser::OperandType value;
    OpAsmParser::OperandType trigger;
    OpAsmParser::OperandType gate;
    Type valueType;
    Type triggerType;
    Type gateType;
    StringAttr modeAttr;
    NamedAttrList attrStorage;
    bool hasGate = false;

    if (parser.parseLParen())
      return failure();
    if (parser.parseOperand(value) || parser.parseComma())
      return failure();
    if (parser.parseAttribute(modeAttr, parser.getBuilder().getNoneType(),
                              "modes", attrStorage))
      return failure();
    auto attrOptional = llhd::symbolizeRegMode(modeAttr.getValue());
    if (!attrOptional)
      return parser.emitError(parser.getCurrentLocation(),
                              "invalid string attribute");
    modesArray.push_back(static_cast<int64_t>(attrOptional.getValue()));
    if (parser.parseOperand(trigger))
      return failure();
    if (succeeded(parser.parseOptionalKeyword("if"))) {
      hasGate = true;
      gateMask.push_back(++gateCount);
      if (parser.parseOperand(gate))
        return failure();
    } else {
      gateMask.push_back(0);
    }
    if (parser.parseColon() || parser.parseType(valueType) ||
        parser.parseComma() || parser.parseType(triggerType))
      return failure();
    if (succeeded(parser.parseOptionalComma())) {
      if (!hasGate)
        return parser.emitError(
            parser.getCurrentLocation(),
            "Got optional gate type, but none was expected!");
      if (parser.parseType(gateType))
        return failure();
    } else {
      if (hasGate)
        return parser.emitError(parser.getCurrentLocation(),
                                "Optional gate type expected!");
    }
    valueOperands.push_back(value);
    triggerOperands.push_back(trigger);
    valueTypes.push_back(valueType);
    triggerTypes.push_back(triggerType);
    if (hasGate) {
      gateOperands.push_back(gate);
      gateTypes.push_back(gateType);
    }
    if (parser.parseRParen())
      return failure();
  }
  if (parser.parseColon() || parser.parseType(initType))
    return failure();
  if (parser.resolveOperand(init, initType, result.operands))
    return failure();
  if (parser.resolveOperands(valueOperands, valueTypes,
                             parser.getCurrentLocation(), result.operands))
    return failure();
  if (parser.resolveOperands(triggerOperands, triggerTypes,
                             parser.getCurrentLocation(), result.operands))
    return failure();
  if (parser.resolveOperands(gateOperands, gateTypes,
                             parser.getCurrentLocation(), result.operands))
    return failure();
  result.addAttribute("gateMask",
                      parser.getBuilder().getI64ArrayAttr(gateMask));
  result.addAttribute("modes", parser.getBuilder().getI64ArrayAttr(modesArray));
  llvm::SmallVector<int32_t, 3> operandSizes;
  operandSizes.push_back(1);
  operandSizes.push_back(valueOperands.size());
  operandSizes.push_back(triggerOperands.size());
  operandSizes.push_back(gateOperands.size());
  result.addAttribute("operand_segment_sizes",
                      parser.getBuilder().getI32VectorAttr(operandSizes));
  result.addTypes(llhd::SigType::get(initType));
}

static void print(OpAsmPrinter &printer, llhd::RegOp op) {
  printer << op.getOperationName() << " " << op.init();
  for (unsigned i = 0; i < op.values().size(); ++i) {
    Optional<llhd::RegMode> mode = llhd::symbolizeRegMode(
        op.modes().getValue()[i].cast<IntegerAttr>().getInt());
    if (!mode)
      op.emitError("invalid RegMode");
    printer << ", (" << op.values()[i] << ", \""
            << llhd::stringifyRegMode(mode.getValue()) << "\" "
            << op.triggers()[i];
    if (op.hasGate(i))
      printer << " if " << op.getGateAt(i);
    printer << " : " << op.values()[i].getType() << ", "
            << op.triggers()[i].getType();
    if (op.hasGate(i))
      printer << ", " << op.getGateAt(i).getType();
    printer << ")";
  }
  printer << " : " << op.init().getType();
}

static LogicalResult verify(llhd::RegOp op) {
  // Values variadic operand must have the same size as the triggers variadic
  if (op.values().size() != op.triggers().size())
    return op.emitOpError("Number of 'values' is not equal to the number of "
                          "'triggers', got ")
           << op.values().size() << " modes, but " << op.triggers().size()
           << " triggers!";

  // Array Attribute of RegModes must have the same number of elements as the
  // variadics
  if (op.modes().size() != op.triggers().size())
    return op.emitOpError("Number of 'modes' is not equal to the number of "
                          "'triggers', got ")
           << op.modes().size() << " modes, but " << op.triggers().size()
           << " triggers!";

  // Array Attribute 'gateMask' must have the same number of elements as the
  // triggers and values variadics
  if (op.gateMask().size() != op.triggers().size())
    return op.emitOpError("Size of 'gateMask' is not equal to the size of "
                          "'triggers', got ")
           << op.gateMask().size() << " modes, but " << op.triggers().size()
           << " triggers!";

  // Number of non-zero elements in 'gateMask' has to be the same as the size of
  // the gates variadic, also each number from 1 to size-1 has to occur only
  // once and in increasing order
  unsigned counter = 0;
  unsigned prevElement = 0;
  for (Attribute maskElem : op.gateMask().getValue()) {
    int64_t val = maskElem.cast<IntegerAttr>().getInt();
    if (val < 0)
      return op.emitError("Element in 'gateMask' must not be negative!");
    if (val == 0)
      continue;
    if (val != ++prevElement)
      return op.emitError(
          "'gateMask' has to contain every number from 1 to the "
          "number of gates minus one exactly once in increasing order "
          "(may have zeros in-between).");
    counter++;
  }
  if (op.gates().size() != counter)
    return op.emitError("The number of non-zero elements in 'gateMask' and the "
                        "size of the 'gates' variadic have to match.");

  // Each value must be either the same type as the init or a signal type with
  // the same type as the init as underlying type.
  for (auto val : op.values()) {
    if (val.getType() != op.init().getType() &&
        val.getType() != llhd::SigType::get(op.init().getType())) {
      op.emitOpError("type of each 'value' has to be either the same as the "
                     "type of 'init' of a signal with the type of 'init' as "
                     "it's underlying type!");
      return failure();
    }
  }
}

#include "Dialect/LLHD/LLHDOpsEnums.cpp.inc"
namespace mlir {
namespace llhd {
#define GET_OP_CLASSES
#include "Dialect/LLHD/LLHDOps.cpp.inc"
} // namespace llhd
} // namespace mlir
