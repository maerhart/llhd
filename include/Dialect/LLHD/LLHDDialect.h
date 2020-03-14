#ifndef MLIR_DIALECT_LLHD_LLHDDIALECT_H
#define MLIR_DIALECT_LLHD_LLHDDIALECT_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/Function.h"

namespace mlir {
namespace llhd {
namespace detail {
struct SigTypeStorage;
}    // namespace detail

class LLHDDialect : public Dialect {
   public:
    explicit LLHDDialect(MLIRContext *context);

    /// Returns the prefix used in the textual IR to refer to LLHD operations
    static StringRef getDialectNamespace() { return "llhd"; }

    /// Parses a type registered to this dialect
    Type parseType(DialectAsmParser &parser) const override;

    /// Print a type registered to this dialect
    void printType(Type type, DialectAsmPrinter &printer) const override;
};

//===----------------------------------------------------------------------===//
// LLHD Types
//===----------------------------------------------------------------------===//
namespace LLHDTypes {
enum Kinds {
    Sig = mlir::Type::FIRST_LLHD_TYPE,
};
}    // namespace LLHDTypes

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
    /// Verify construction invariants passed to get and getChecked
    static LogicalResult VerifyConstructionInvariants(Location loc,
                                                      Type underlyingType);
    /// The underlying type of the sig type
    Type getUnderlyingType();
};
}    // namespace llhd
}    // namespace mlir

#endif    // MLIR_DIALECT_LLHD_LLHDDIALECT_H
