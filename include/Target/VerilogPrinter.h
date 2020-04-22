#ifndef TARGET_VERILOGPRINTER_H
#define TARGET_VERILOGPRINTER_H

#include "Dialect/LLHD/LLHDOps.h"

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Module.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/FormattedStream.h"

using namespace llvm;

namespace mlir {
namespace llhd {

class VerilogPrinter {
public:
  VerilogPrinter(formatted_raw_ostream &output) : out(output) {}

  LogicalResult printModule(ModuleOp op);
  LogicalResult printOperation(Operation *op, unsigned indentAmount = 0);

private:
  LogicalResult printType(Type type);
  LogicalResult printUnaryOp(Operation *op, char opSymbol,
                             unsigned indentAmount = 0);
  LogicalResult printBinaryOp(Operation *op, char opSymbol,
                              unsigned indentAmount = 0);
  LogicalResult printSignedBinaryOp(Operation *op, char opSymbol,
                                    unsigned indentAmount = 0);

  /// Prints a SSA value. In case no mapping to a name exists yet, a new one is
  /// added.
  Twine getVariableName(Value value);

  /// Adds an alias for an existing SSA value. In case doesn't exist, it just
  /// adds the alias as a new value.
  void addAliasVariable(Value alias, Value existing);

  formatted_raw_ostream &out;
  unsigned nextValueNum = 0;
  DenseMap<Value, unsigned> mapValueToName;
  DenseMap<Value, unsigned> timeValueMap;
};

} // namespace llhd
} // namespace mlir

#endif // TARGET_VERILOGPRINTER_H
