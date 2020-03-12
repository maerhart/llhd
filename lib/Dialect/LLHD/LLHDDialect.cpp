#include "Dialect/LLHD/LLHDDialect.h"
#include "Dialect/LLHD/LLHDOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Module.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/STLExtras.h"
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

    addOperations<
#define GET_OP_LIST
#include "Dialect/LLHD/LLHDOps.cpp.inc"
      >(); 

}

