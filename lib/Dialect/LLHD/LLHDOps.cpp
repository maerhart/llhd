#include "Dialect/LLHD/LLHDOps.h"
#include "Dialect/LLHD/LLHDDialect.h"
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
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/raw_ostream.h"
// #include <bits/stdint-intn.h>
#include <cstddef>
#include <functional>
#include <iterator>
//#include <llvm-9/llvm/ADT/STLExtras.h>

using namespace mlir;


//===---------------------------------------------------------------------===//
// LLHD Operations
//===---------------------------------------------------------------------===//

//===---------------------------------------------------------------------===//
// ConstOp

static ParseResult parseConstOp(OpAsmParser &parser, OperationState &result) {
    IntegerAttr value;
    if (parser.parseOptionalAttrDict(result.attributes) ||
        parser.parseAttribute(value, "value", result.attributes))
        return failure();

    result.addTypes(value.getType());
    return success();
}

static void print(OpAsmPrinter &printer, llhd::ConstOp op) {
    printer << "llhd.const";
    printer.printOptionalAttrDict(op.getAttrs(), {"value"});
    printer << op.value();
}

static LogicalResult verify(llhd::ConstOp op) {
    return success();
}

namespace mlir {
namespace llhd {
#define GET_OP_CLASSES
#include "Dialect/LLHD/LLHDOps.cpp.inc"
} // namespace llhd
} // namespace mlir
