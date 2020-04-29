#include "Dialect/LLHD/LLHDOps.h"
#include "mlir/Dialect/CommonFolders.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/ArrayRef.h"

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

// Sig Op

/// Verify the construction invariants of a sig operation.
static LogicalResult verify(llhd::SigOp op) {
  // cast the result type to sig
  auto resultType = op.getType().dyn_cast<llhd::SigType>();

  // check the operand type matches the result type
  if (op.init().getType() != resultType.getUnderlyingType())
    return op.emitError(
               "The operand type is not equal to the signal type. Expected ")
           << resultType.getUnderlyingType() << " but got "
           << op.init().getType();

  return success();
}

// Prb Op

/// Verify the construction invariants of a llhd.prb instruction
static LogicalResult verify(llhd::PrbOp op) {
  auto sigType = op.signal().getType().dyn_cast<llhd::SigType>();

  // check the type carried by the signal matches the result type
  if (sigType.getUnderlyingType() != op.getType())
    return op.emitError(
               "The operand type is not equal to the signal type. Expected ")
           << sigType.getUnderlyingType() << " but got " << op.getType();

  return success();
}

// Drv Op

/// Verify construction invariants of a llhd.drv operation
static LogicalResult verify(llhd::DrvOp op) {
  auto sigType = op.signal().getType().dyn_cast<llhd::SigType>();

  // check the type of the new value matches the type carried by the signal
  if (sigType.getUnderlyingType() != op.value().getType())
    return op.emitError("The new value's type is not equal to the signal type. "
                        "Expected ")
           << sigType.getUnderlyingType() << " but got "
           << op.value().getType();

  return success();
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
static LogicalResult verify(llhd::WaitOp op) {
  // Check that there is not more than one time operand.
  if (op.time().size() > 1) {
    op.emitOpError("It is not allowed to have more than one time operand.");
    return failure();
  }
  return success();
}

static ParseResult parseWaitOp(OpAsmParser &parser, OperationState &result) {
  llvm::SMLoc loc = parser.getCurrentLocation();
  SmallVector<OpAsmParser::OperandType, 4> obsOperands;
  SmallVector<OpAsmParser::OperandType, 4> destOpsOperands;
  SmallVector<OpAsmParser::OperandType, 1> timeOperands;
  SmallVector<Type, 4> obsTypes;
  SmallVector<Type, 4> destOpsTypes;
  SmallVector<Type, 1> timeTypes;

  bool hasOptionalTime = false;
  Block *destSuccessor = nullptr;

  // Consider the case where there is not signal to observe
  if (succeeded(parser.parseOptionalKeyword("for")))
    hasOptionalTime = true;

  // Parse the signal observe list until a for or a successor can be parsed.
  while (!hasOptionalTime &&
         !parser.parseOptionalSuccessor(destSuccessor).hasValue()) {
    OpAsmParser::OperandType operand;
    if (parser.parseOperand(operand))
      return failure();
    obsOperands.push_back(operand);
    if (failed(parser.parseOptionalComma())) {
      if (parser.parseKeyword("for"))
        return failure();
      hasOptionalTime = true;
      break;
    }
  }

  // Parse the optional time operand
  if (hasOptionalTime) {
    OpAsmParser::OperandType operand;
    if (parser.parseOperand(operand))
      return failure();
    timeOperands.push_back(operand);

    if (parser.parseComma())
      return failure();

    if (parser.parseSuccessor(destSuccessor))
      return failure();
  }

  // Parse the arguments of the target basic block
  if (succeeded(parser.parseOptionalLParen())) {
    if (parser.parseOperandList(destOpsOperands, OpAsmParser::Delimiter::None))
      return failure();
    if (parser.parseColonTypeList(destOpsTypes))
      return failure();
    if (parser.parseRParen())
      return failure();
  }

  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  // Parse the types, notice that the colon is only parsed if there is at least
  // one type to parse
  if (!obsOperands.empty() || !timeOperands.empty()) {
    if (parser.parseColon())
      return failure();

    // Parse the types of the observed signals, we cannot use the parseTypeList
    // function from the parser because there we cannot specify the amount of
    // types in the list to be parsed
    for (unsigned i = 0, n = obsOperands.size(); i < n; ++i) {
      Type type;
      if (parser.parseType(type))
        return failure();

      obsTypes.push_back(type);

      if (i < n - 1) {
        if (parser.parseComma())
          return failure();
      }
    }

    // Parse the optional time type at the end of the list
    if (!timeOperands.empty()) {
      if (!obsOperands.empty()) {
        if (parser.parseComma())
          return failure();
      }
      if (parser.parseTypeList(timeTypes))
        return failure();
    }
  }
  if (parser.resolveOperands(obsOperands, obsTypes, loc, result.operands))
    return failure();
  if (parser.resolveOperands(timeOperands, timeTypes, loc, result.operands))
    return failure();
  if (parser.resolveOperands(destOpsOperands, destOpsTypes, loc,
                             result.operands))
    return failure();

  result.addSuccessors(destSuccessor);
  result.addAttribute("operand_segment_sizes",
                      parser.getBuilder().getI32VectorAttr(
                          {static_cast<int32_t>(obsOperands.size()),
                           static_cast<int32_t>(timeOperands.size()),
                           static_cast<int32_t>(destOpsOperands.size())}));
  return success();
}

static void print(OpAsmPrinter &p, llhd::WaitOp op) {
  p << op.getOperationName() << " ";
  p << op.obs();
  if (!op.time().empty())
    p << (op.obs().empty() ? "" : " ") << "for " << op.time() << ", ";
  else if (!op.obs().empty())
    p << ", ";
  // else: there are no observed signals and no time argument, so don't print
  // the comma
  p << op.dest();
  if (!op.destOps().empty())
    p << "(" << op.destOps() << " : " << op.destOps().getTypes() << ")";
  p.printOptionalAttrDict(op.getAttrs(),
                          /* elidedAttrs= */ {"operand_segment_sizes"});
  if (op.obs().empty() && op.time().empty())
    return;
  p << " : " << op.obs().getTypes()
    << (op.time().empty() || op.obs().empty() ? "" : ", ")
    << op.time().getTypes();
}

// Implement this operation for the BranchOpInterface
Optional<OperandRange> llhd::WaitOp::getSuccessorOperands(unsigned index) {
  assert(index == 0 && "invalid successor index");
  return destOps();
}

// Implement this operation for the BranchOpInterface
bool llhd::WaitOp::canEraseSuccessorOperand() { return true; }

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
  return success();
}

Region *llhd::EntityOp::getCallableRegion() {
  return isExternal() ? nullptr : &getBody();
}

ArrayRef<Type> llhd::EntityOp::getCallableResults() {
  return getType().getResults();
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

namespace mlir {
namespace llhd {
#define GET_OP_CLASSES
#include "Dialect/LLHD/LLHDOps.cpp.inc"
} // namespace llhd
} // namespace mlir
