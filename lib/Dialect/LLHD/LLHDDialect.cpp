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
  addTypes<SigType>();
  addOperations<
#define GET_OP_LIST
#include "Dialect/LLHD/LLHDOps.cpp.inc"
      >();
}

Type LLHDDialect::parseType(DialectAsmParser &parser) const {
  Type underlyignType;
  if (parser.parseKeyword("sig") || parser.parseLess())
    Type();
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

void LLHDDialect::printType(Type type, DialectAsmPrinter &printer) const {
  // for now print and parse using the format: sig ::= !llhd.sig<type>.
  SigType sig = type.cast<SigType>();
  printer << "sig<";
  printer.printType(sig.getUnderlyingType());
  printer << ">";
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

struct TimeTypeStorage : public mlir::TypeStorage {
public:
  // use a SmallVector containing the three timing attribute for uniquing
  using KeyTy = std::pair<Type, llvm::ArrayRef<unsigned>>;

  TimeTypeStorage(Type timeType, llvm::ArrayRef<unsigned> additionalAttrs)
      : timeType(timeType), additionalAttrs(additionalAttrs) {}

  /// Compare the time elements contained in the key's SmallVector for uniquing.
  bool operator==(const KeyTy &key) const {
    return key == getKey(timeType, additionalAttrs);
  }

  /// Build the time type key.
  static KeyTy getKey(Type timeType, llvm::ArrayRef<unsigned> additionalAttrs) {
    return KeyTy(timeType, additionalAttrs);
  }

  /// Construct an istance of the Time Type.
  static TimeTypeStorage *construct(mlir::TypeStorageAllocator &allocator,
                                    const KeyTy &key) {
    assert(key.second.size() == 3);
    return new (allocator.allocate<TimeTypeStorage>())
        TimeTypeStorage(key.first, key.second);
  }

  Type getTimeType() const { return timeType; }
  /// Return the stored delta value.
  unsigned getDelta() const {
    return additionalAttrs.size() > 0 ? additionalAttrs[0] : 0;
  }
  /// Return the stored eps value.
  unsigned getEps() const {
    return additionalAttrs.size() == 2 ? additionalAttrs[1] : 0;
  }

private:
  // Use unsigned integers for the three timing elements.
  Type timeType;
  llvm::ArrayRef<unsigned> additionalAttrs;
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

TimeType TimeType::get(Type timeType,
                       llvm::ArrayRef<unsigned> additionalAttributes) {
  return Base::get(timeType.getContext(), LLHDTypes::Time, timeType,
                   additionalAttributes);
}

LogicalResult TimeType::VerifyConstructionInvariants(
    Location loc, Type timeType, llvm::ArrayRef<unsigned> additionalAttrs) {
  if (additionalAttrs.size() > 2)
    return emitError(loc) << "Too many additional attributes provided. "
                             "Expected at most 2, but got "
                          << additionalAttrs.size();
  if (!timeType.isa<IntegerType>())
    return emitError(loc) << "Expected IntegerType, got " << timeType;
  return success();
}

Type TimeType::getTimeType() { return getImpl()->getTimeType(); }
unsigned TimeType::getDelta() { return getImpl()->getDelta(); }
unsigned TimeType::getEps() { return getImpl()->getEps(); }