#ifndef MLIR_DIALECT_LLHD_LLHDDIALECT_H
#define MLIR_DIALECT_LLHD_LLHDDIALECT_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/Function.h"

namespace mlir {
namespace llhd {

class LLHDDialect : public Dialect {
public:
  explicit LLHDDialect(MLIRContext *context);

  /// Returns the prefix used in the textual IR to refer to LLHD operations
  static StringRef getDialectNamespace() { return "llhd"; }

  /// Parses a type registered to this dialect
  //Type parseType(DialectAsmParser &parser) const override;

  /// Print a type registered to this dialect
  //void printType(Type type, DialectAsmPrinter &printer) const override;
};


} // namespace llhd
} // namespace mlir

#endif // MLIR_DIALECT_LLHD_LLHDDIALECT_H
