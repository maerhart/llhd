#include "Dialect/LLHD/LLHDDialect.h"
#include "Dialect/LLHD/LLHDOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

using namespace mlir;
using namespace mlir::llhd;

//===----------------------------------------------------------------------===//
// LLHDDialect Interfaces
//===----------------------------------------------------------------------===//

namespace {
/// This class defines the interface for handling inlining with LLHD operations.
struct LLHDInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  //===--------------------------------------------------------------------===//
  // Analysis Hooks
  //===--------------------------------------------------------------------===//

  /// All operations within LLHD can be inlined.
  bool isLegalToInline(Operation *, Region *,
                       BlockAndValueMapping &) const final {
    return true;
  }

  bool isLegalToInline(Region *, Region *, BlockAndValueMapping &) const final {
    return true;
  }
};
} // end anonymous namespace

//===----------------------------------------------------------------------===//
// LLHD Dialect
//===----------------------------------------------------------------------===//

LLHDDialect::LLHDDialect(mlir::MLIRContext *context)
    : Dialect(getDialectNamespace(), context) {
  addTypes<SigType, TimeType>();
  addAttributes<TimeAttr>();
  addOperations<
#define GET_OP_LIST
#include "Dialect/LLHD/LLHDOps.cpp.inc"
      >();
  addInterfaces<LLHDInlinerInterface>();
}

Operation *LLHDDialect::materializeConstant(OpBuilder &builder, Attribute value,
                                            Type type, Location loc) {
  return builder.create<llhd::ConstOp>(loc, type, value);
}

//===----------------------------------------------------------------------===//
// Type parsing
//===----------------------------------------------------------------------===//

/// Parse a signal type.
/// Syntax: sig ::= !llhd.sig<type>
Type parseSigType(DialectAsmParser &parser) {
  Type underlyingType;
  if (parser.parseLess())
    return Type();

  llvm::SMLoc loc = parser.getCurrentLocation();
  if (parser.parseType(underlyingType)) {
    parser.emitError(loc, "No signal type found. Signal needs an underlying "
                          "type.");
    return nullptr;
  }
  loc = parser.getCurrentLocation();
  if (!underlyingType.isa<IntegerType>() && !underlyingType.isa<TimeType>()) {
    parser.emitError(loc, "Illegal signal type: ") << underlyingType;
    return Type();
  }
  if (parser.parseGreater())
    return Type();
  return SigType::get(underlyingType);
}

Type LLHDDialect::parseType(DialectAsmParser &parser) const {
  llvm::StringRef typeKeyword;
  // parse the type keyword first
  llvm::SMLoc loc = parser.getCurrentLocation();
  if (parser.parseKeyword(&typeKeyword))
    return Type();
  if (typeKeyword == SigType::getKeyword()) {
    return parseSigType(parser);
  }
  if (typeKeyword == TimeType::getKeyword())
    return TimeType::get(getContext());
  return Type();
}

//===----------------------------------------------------------------------===//
// Type printing
//===----------------------------------------------------------------------===//

/// Print a signal type with custom syntax:
/// type ::= !sig.type<underlying-type>
void printSigType(SigType sig, DialectAsmPrinter &printer) {
  printer << sig.getKeyword() << "<";
  printer.printType(sig.getUnderlyingType());
  printer << ">";
}

void LLHDDialect::printType(Type type, DialectAsmPrinter &printer) const {
  switch (type.getKind()) {
  case LLHDTypes::Sig: {
    SigType sig = type.dyn_cast<SigType>();
    printSigType(sig, printer);
    break;
  }
  case LLHDTypes::Time: {
    TimeType time = type.dyn_cast<TimeType>();
    printer << time.getKeyword();
    break;
  }

  default:
    break;
  }
}

//===----------------------------------------------------------------------===//
// Attribute parsing
//===----------------------------------------------------------------------===//

/// Parse a time attribute with the custom syntax:
/// time ::= #llhd.time<time time_unit, delta d, epsilon e>
Attribute parseTimeAttribute(DialectAsmParser &parser, Type type) {
  if (parser.parseLess())
    return Attribute();

  // values to parse
  llvm::SmallVector<unsigned, 3> values;
  llvm::StringRef timeUnit;
  unsigned time = 0;
  unsigned delta = 0;
  unsigned eps = 0;

  // parse the time value
  if (parser.parseInteger(time) || parser.parseKeyword(&timeUnit))
    return {};
  values.push_back(time);

  // parse the delta step value
  if (parser.parseComma() || parser.parseInteger(delta) ||
      parser.parseKeyword("d"))
    return {};
  values.push_back(delta);

  // parse the epsilon value
  if (parser.parseComma() || parser.parseInteger(eps) ||
      parser.parseKeyword("e") || parser.parseGreater())
    return Attribute();
  values.push_back(eps);

  // return a new instance of time attribute
  return TimeAttr::get(type, values, timeUnit);
}

/// Parse an LLHD attribute
Attribute LLHDDialect::parseAttribute(DialectAsmParser &parser,
                                      Type type) const {
  llvm::StringRef attrKeyword;
  // parse keyword first
  if (parser.parseKeyword(&attrKeyword))
    return Attribute();
  if (attrKeyword == TimeAttr::getKeyword()) {
    return parseTimeAttribute(parser, type);
  }
  return Attribute();
}

//===----------------------------------------------------------------------===//
// Attribute printing
//===----------------------------------------------------------------------===//

/// Print an LLHD time attribute.
void printTimeAttribute(TimeAttr attr, DialectAsmPrinter &printer) {
  printer << attr.getKeyword() << "<";
  printer << attr.getTime() << attr.getTimeUnit() << ", ";
  printer << attr.getDelta() << "d, ";
  printer << attr.getEps() << "e>";
}

void LLHDDialect::printAttribute(Attribute attr,
                                 DialectAsmPrinter &printer) const {
  switch (attr.getKind()) {
  case LLHDAttrs::Time: {
    TimeAttr time = attr.dyn_cast<TimeAttr>();
    printTimeAttribute(time, printer);
    break;
  }
  default:
    break;
  }
}

namespace mlir {
namespace llhd {
namespace detail {

//===----------------------------------------------------------------------===//
// Type storage
//===----------------------------------------------------------------------===//

// Sig Type Storage

/// Storage struct implementation for LLHD's sig type. The sig type only
/// contains one underlying llhd type.
struct SigTypeStorage : public mlir::TypeStorage {
  using KeyTy = mlir::Type;
  /// construcor for sig type's storage.
  /// Takes the underlying type as the only argument
  SigTypeStorage(mlir::Type underlyingType) : underlyingType(underlyingType) {}

  /// compare sig type instances on the underlying type
  bool operator==(const KeyTy &key) const { return key == getUnderlyingType(); }

  /// return the KeyTy for sig type
  static KeyTy getKey(mlir::Type underlyingType) {
    return KeyTy(underlyingType);
  }

  /// construction method for creating a new instance of the sig type
  /// storage
  static SigTypeStorage *construct(mlir::TypeStorageAllocator &allocator,
                                   const KeyTy &key) {
    return new (allocator.allocate<SigTypeStorage>()) SigTypeStorage(key);
  }

  /// get the underlying type
  mlir::Type getUnderlyingType() const { return underlyingType; }

private:
  mlir::Type underlyingType;
};

//===----------------------------------------------------------------------===//
// Attribute storage
//===----------------------------------------------------------------------===//

struct TimeAttrStorage : public mlir::AttributeStorage {
public:
  // use the ArrayRef containign the timing attributes for uniquing
  using KeyTy = std::tuple<Type, llvm::ArrayRef<unsigned>, llvm::StringRef>;

  /// Construct a time attribute storage
  TimeAttrStorage(Type type, llvm::ArrayRef<unsigned> timeValues,
                  llvm::StringRef timeUnit)
      : AttributeStorage(type), timeValues(timeValues), timeUnit(timeUnit) {}

  /// Compare two istances of the time attribute. Hashing and equality are done
  /// only on the time values and time unit. The time type is implicitly always
  /// equal.
  bool operator==(const KeyTy &key) const {
    return (std::get<1>(key) == timeValues && std::get<2>(key) == timeUnit);
  }

  /// Generate hash key for uniquing.
  static unsigned hashKey(const KeyTy &key) {
    auto vals = std::get<1>(key);
    auto unit = std::get<2>(key);
    return llvm::hash_combine(vals, unit);
  }

  /// Construction method for llhd's time attribute
  static TimeAttrStorage *construct(mlir::AttributeStorageAllocator &allocator,
                                    const KeyTy &key) {
    auto keyValues = std::get<1>(key);
    auto values = allocator.copyInto(keyValues);
    auto keyUnit = std::get<2>(key);
    auto unit = allocator.copyInto(keyUnit);

    return new (allocator.allocate<TimeAttrStorage>())
        TimeAttrStorage(std::get<0>(key), values, unit);
  }

  llvm::ArrayRef<unsigned> getValue() const { return timeValues; }

  unsigned getTime() { return timeValues[0]; }

  unsigned getDelta() { return timeValues[1]; }

  unsigned getEps() { return timeValues[2]; }

  llvm::StringRef getTimeUnit() { return timeUnit; }

private:
  llvm::ArrayRef<unsigned> timeValues;
  llvm::StringRef timeUnit;
};

} // namespace detail
} // namespace llhd
} // namespace mlir

//===----------------------------------------------------------------------===//
// LLHD Types
//===----------------------------------------------------------------------===//

// Sig Type

SigType SigType::get(mlir::Type underlyingType) {
  return Base::get(underlyingType.getContext(), LLHDTypes::Sig, underlyingType);
};

LogicalResult SigType::verifyConstructionInvariants(Location loc,
                                                    Type underlyingType) {
  // check whether the given type is legal
  if (!underlyingType.isa<IntegerType>() && !underlyingType.isa<TimeType>())
    return emitError(loc) << "The provided signal type " << underlyingType
                          << " is not legal";
  return success();
}

mlir::Type SigType::getUnderlyingType() {
  return getImpl()->getUnderlyingType();
}

// Time Type

TimeType TimeType::get(MLIRContext *context) {
  return Base::get(context, LLHDTypes::Time);
}

//===----------------------------------------------------------------------===//
// LLHD Attribtues
//===----------------------------------------------------------------------===//

// Time Attribute

TimeAttr TimeAttr::get(Type type, llvm::ArrayRef<unsigned> timeValues,
                       llvm::StringRef timeUnit) {
  return Base::get(type.getContext(), LLHDAttrs::Time, type, timeValues,
                   timeUnit);
}

LogicalResult
TimeAttr::verifyConstructionInvariants(Location loc, Type type,
                                       llvm::ArrayRef<unsigned> timeValues,
                                       llvm::StringRef timeUnit) {
  // Check the attribute type is of TimeType.
  if (!type.isa<TimeType>())
    return emitError(loc) << "Time attribute type has to be TimeType, but got "
                          << type;

  // Check the time unit is a legal SI unit
  std::vector<std::string> legalUnits{"ys", "zs", "as", "fs", "ps",
                                      "ns", "us", "ms", "s"};
  if (std::find(legalUnits.begin(), legalUnits.end(), timeUnit) ==
      legalUnits.end())
    return emitError(loc) << "Illegal time unit.";

  // Check there are exactly 3 time values
  if (timeValues.size() != 3)
    return emitError(loc) << "Got a wrong number of time values. Expected "
                             "exactly 3, but got "
                          << timeValues.size();

  // Check the time values are positive or zero integers.
  if (timeValues[0] < 0 || timeValues[1] < 0 || timeValues[2] < 0)
    return emitError(loc) << "Received a negative time value.";

  return success();
}

llvm::ArrayRef<unsigned> TimeAttr::getValue() const {
  return getImpl()->getValue();
}

unsigned TimeAttr::getTime() const { return getImpl()->getTime(); }

unsigned TimeAttr::getDelta() const { return getImpl()->getDelta(); }

unsigned TimeAttr::getEps() const { return getImpl()->getEps(); }

llvm::StringRef TimeAttr::getTimeUnit() const {
  return getImpl()->getTimeUnit();
}
