#ifndef MLIR_DIALECT_LLHD_LLHDDIALECT_H
#define MLIR_DIALECT_LLHD_LLHDDIALECT_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/Function.h"

namespace mlir {
namespace llhd {
namespace detail {
struct SigTypeStorage;
struct TimeAttrStorage;
} // namespace detail

class LLHDDialect : public Dialect {
public:
  explicit LLHDDialect(MLIRContext *context);

  /// Returns the prefix used in the textual IR to refer to LLHD operations
  static StringRef getDialectNamespace() { return "llhd"; }

  /// Parses a type registered to this dialect
  Type parseType(DialectAsmParser &parser) const override;

  /// Print a type registered to this dialect
  void printType(Type type, DialectAsmPrinter &printer) const override;

  /// Parse an attribute regustered to this dialect
  Attribute parseAttribute(DialectAsmParser &parser, Type type) const override;

  /// Print an attribute registered to this dialect
  void printAttribute(Attribute attr,
                      DialectAsmPrinter &printer) const override;
};

//===----------------------------------------------------------------------===//
// LLHD Types
//===----------------------------------------------------------------------===//
namespace LLHDTypes {
enum Kinds {
  Sig = mlir::Type::FIRST_LLHD_TYPE,
  Time,
};
} // namespace LLHDTypes

class SigType
    : public mlir::Type::TypeBase<SigType, mlir::Type, detail::SigTypeStorage> {
public:
  using Base::Base;
  /// Return whether the given kind is of type Sig
  static bool kindof(unsigned kind) { return kind == LLHDTypes::Sig; }
  /// Get a new instance of llhd sig type
  static SigType get(mlir::Type underlyingType);
  /// Get a new instance of llhd sig type defined at the given location
  static SigType getChecked(mlir::Type underlyingType, Location location);
  /// Verify construction invariants, passed to get and getChecked
  static LogicalResult VerifyConstructionInvariants(Location loc,
                                                    Type underlyingType);
  /// The underlying type of the sig type
  Type getUnderlyingType();
};

class TimeType : public Type::TypeBase<TimeType, Type> {
public:
  using Base::Base;

  /// Return whether the given kind is of type Time
  static bool kindof(unsigned kind) { return kind == LLHDTypes::Time; }

  /// Get a new isntance of type Time
  static TimeType get(MLIRContext *context);
};

//===----------------------------------------------------------------------===//
// LLHD Attributes
//===----------------------------------------------------------------------===//

namespace LLHDAttrs {
enum Kinds {
  Time = mlir::Attribute::FIRST_LLHD_ATTR,
};
} // namespace LLHDAttrs

class TimeAttr
    : public Attribute::AttrBase<TimeAttr, Attribute, detail::TimeAttrStorage> {
public:
  using Base::Base;

  /// Returns whether the passed argument is of kind Time.
  static bool kindof(unsigned kind) { return kind == LLHDAttrs::Time; }

  /// Get a new instance of Time attribute.
  static TimeAttr get(MLIRContext *context, unsigned time, unsigned delta,
                      unsigned eps);
  static LogicalResult VerifyConstructionInvariants(Location loc, unsigned time,
                                                    unsigned delta,
                                                    unsigned eps);

  /// Get the time value stored in the attribute.
  unsigned getTime();

  /// Get the delta value stored in the attribute. Returns 0 if no delta value
  /// is defined.
  unsigned getDelta();

  /// Get the eps value stored in the attribute. Returns 0 if no epsilon value
  /// is defined.
  unsigned getEps();
};

} // namespace llhd
} // namespace mlir

#endif // MLIR_DIALECT_LLHD_LLHDDIALECT_H
