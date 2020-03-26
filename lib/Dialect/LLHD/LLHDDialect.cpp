#include "Dialect/LLHD/LLHDDialect.h"
#include "Dialect/LLHD/LLHDOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Module.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/STLExtras.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/None.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorHandling.h"
#include <algorithm>
// #include <bits/stdint-intn.h>
#include <cstddef>
#include <string>

using namespace mlir;
using namespace mlir::llhd;

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
}

//===----------------------------------------------------------------------===//
// LLHD Type parsing/printing
//===----------------------------------------------------------------------===//

Type LLHDDialect::parseType(DialectAsmParser &parser) const {
  StringRef typeKeyword;
  // parse the type keyword first
  if (parser.parseKeyword(&typeKeyword))
    return Type();
  if (typeKeyword == "sig") {
    Type underlyignType;
    if (parser.parseLess())
      return Type();
    llvm::SMLoc currLoc = parser.getCurrentLocation();
    if (parser.parseType(underlyignType)) {
      parser.emitError(currLoc,
                       "No signal type found. Signal needs an underlying "
                       "type.");
      return nullptr;
    }
    if (!underlyignType.isa<IntegerType>()) {
      parser.emitError(currLoc, "Illegal signal type: ") << underlyignType;
      return Type();
    }
    if (parser.parseGreater())
      return Type();
    return SigType::get(underlyignType);
  }
  if (typeKeyword == "time")
    return TimeType::get(getContext());
  return Type();
}

void LLHDDialect::printType(Type type, DialectAsmPrinter &printer) const {
  // for now print and parse using the format: sig ::= !llhd.sig<type>.
  switch (type.getKind()) {
  case LLHDTypes::Sig: {
    SigType sig = type.cast<SigType>();
    printer << "sig<";
    printer.printType(sig.getUnderlyingType());
    printer << ">";
    break;
  }
  case LLHDTypes::Time: {
    printer << "time";
    break;
  }

  default:
    break;
  }
}

//===----------------------------------------------------------------------===//
// LLHD Attribute parsing/printing
//===----------------------------------------------------------------------===//

Attribute LLHDDialect::parseAttribute(DialectAsmParser &parser,
                                      Type type) const {
  StringRef attrKeyword;
  // parse keyword first
  if (parser.parseKeyword(&attrKeyword))
    return Attribute();
  if (attrKeyword == "time") {
    if (parser.parseLess())
      return Attribute();

    unsigned time;
    unsigned delta = 0;
    unsigned eps = 0;
    if (parser.parseInteger(time) || parser.parseKeyword("ns"))
      return Attribute();
    if (succeeded(parser.parseOptionalComma())) {
      if (parser.parseInteger(delta) || parser.parseKeyword("d"))
        return Attribute();
    }
    if (succeeded(parser.parseOptionalComma())) {
      if (parser.parseInteger(eps) || parser.parseKeyword("e"))
        return Attribute();
    }
    if (parser.parseGreater())
      return Attribute();
    return TimeAttr::get(getContext(), time, delta, eps);
  }
  return Attribute();
}

void LLHDDialect::printAttribute(Attribute attr,
                                 DialectAsmPrinter &printer) const {
  switch (attr.getKind()) {
  case LLHDAttrs::Time: {
    TimeAttr time = attr.dyn_cast<TimeAttr>();

    printer << "time<" << time.getTime() << "ns";
    if (time.getDelta() > 0)
      printer << ", " << time.getDelta() << "d";
    if (time.getEps() > 0)
      printer << ", " << time.getEps() << "e";
    printer << ">";
    break;
  }
  default:
    break;
  }
}
//===----------------------------------------------------------------------===//
// LLHD Types
//===----------------------------------------------------------------------===//

namespace mlir {
namespace llhd {
namespace detail {
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

struct TimeAttrStorage : public mlir::AttributeStorage {
public:
  // use the ArrayRef containign the timing attributes for uniquing
  using KeyTy = std::tuple<unsigned, unsigned, unsigned>;

  /// Construct a time attribute storage
  TimeAttrStorage(unsigned time, unsigned delta, unsigned eps)
      : time(time), delta(delta), eps(eps) {}

  /// Compare two istances of the time attribute. Equality is checked on all
  /// the values contained in the attributes.
  bool operator==(const KeyTy &key) const {
    return key == KeyTy(time, delta, eps);
  }

  /// Construction method for llhd's time attribute
  static TimeAttrStorage *construct(mlir::AttributeStorageAllocator &allocator,
                                    const KeyTy &key) {
    return new (allocator.allocate<TimeAttrStorage>())
        TimeAttrStorage(std::get<0>(key), std::get<1>(key), std::get<2>(key));
  }

  /// Return the time value.
  unsigned getTime() { return time; }

  /// Return the delta step value. Defaults to 0 when not defined.
  unsigned getDelta() { return delta; }

  /// Return the epsilon value. Defaults to 0 when not defined.
  unsigned getEps() { return eps; }

private:
  // llvm::ArrayRef<unsigned> timeValues;
  unsigned time;
  unsigned delta;
  unsigned eps;
};

} // namespace detail
} // namespace llhd
} // namespace mlir

// Sig Type

SigType SigType::get(mlir::Type underlyingType) {
  return Base::get(underlyingType.getContext(), LLHDTypes::Sig, underlyingType);
};

SigType SigType::getChecked(mlir::Type underlyingType, Location location) {
  return Base::getChecked(location, LLHDTypes::Sig, underlyingType);
}
LogicalResult SigType::VerifyConstructionInvariants(Location loc,
                                                    Type underlyingType) {
  // check whether the given type is legal
  if (!underlyingType.isa<IntegerType>())
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
TimeAttr TimeAttr::get(MLIRContext *context, unsigned time, unsigned delta,
                       unsigned epsilon) {
  return Base::get(context, LLHDAttrs::Time, time, delta, epsilon);
}
LogicalResult TimeAttr::VerifyConstructionInvariants(Location loc,
                                                     unsigned time,
                                                     unsigned delta,
                                                     unsigned eps) {
  if (delta == 0 && eps > 0)
    return emitError(loc) << "Cannot have an epsilon value defined without a "
                             "delta step value.";

  return success();
}

unsigned TimeAttr::getTime() { return getImpl()->getTime(); }

unsigned TimeAttr::getDelta() { return getImpl()->getDelta(); }

unsigned TimeAttr::getEps() { return getImpl()->getEps(); }
