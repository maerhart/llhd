#include "Dialect/LLHD/LLHDOps.h"
#include "Dialect/LLHD/LLHDDialect.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
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

// ConstOp

/// Parse an LLHD const operation with following syntax:
/// op ::= llhd.const type literal
static ParseResult parseConstOp(OpAsmParser &parser, OperationState &result) {
    IntegerAttr value;
    Type type;
    if (parser.parseType(type) ||
        parser.parseAttribute(value, "value", result.attributes))
        return failure();

    // auto res = type.dyn_cast<IntegerType>();
    // if (!res) return failure();
    result.addTypes(type);
    return success();
}

/// print an LLHD const operation
static void print(OpAsmPrinter &printer, llhd::ConstOp op) {
    printer << "llhd.const ";
    printer.printType(op.getType());
    printer << " " << op.value();
}

static LogicalResult verify(llhd::ConstOp op) {
    if (!op.getType().isa<IntegerType>() && !op.getType().isa<llhd::SigType>())
        return failure();
    return success();
}

namespace mlir {
namespace llhd {
#define GET_OP_CLASSES
#include "Dialect/LLHD/LLHDOps.cpp.inc"
}    // namespace llhd
}    // namespace mlir
