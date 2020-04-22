#include "LLHDToLLVM/LLHDToLLVM.h"
#include "Dialect/LLHD/LLHDDialect.h"
#include "Dialect/LLHD/LLHDOps.h"

#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace llhd {
#define GEN_PASS_CLASSES
#include "LLHDToLLVM/Passes.h.inc"
} // namespace llhd
} // namespace mlir

using namespace mlir;
using namespace mlir::llhd;

namespace {
static int counter = 0;

//===----------------------------------------------------------------------===//
// Unit conversions
//===----------------------------------------------------------------------===//

/// Convert an `llhd.entity` unit to LLVM. The result is an `llvm.func` which
/// takes a pointer to the state as arguments.
struct EntityOpConversion : public ConvertToLLVMPattern {
  explicit EntityOpConversion(MLIRContext *ctx,
                              LLVMTypeConverter &typeConverter)
      : ConvertToLLVMPattern(llhd::EntityOp::getOperationName(), ctx,
                             typeConverter) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    // get adapted operands
    OperandAdaptor<EntityOp> transformed(operands);
    // get entity operation
    auto entityOp = cast<EntityOp>(op);

    // collect llvm types
    auto voidTy = getVoidType();
    auto i8PtrTy = getVoidPtrType();

    // get llvm function signature
    auto funcTy = LLVM::LLVMType::getFunctionTy(voidTy, i8PtrTy, false);
    // create the llvm function
    auto llvmFunc = rewriter.create<LLVM::LLVMFuncOp>(
        op->getLoc(), entityOp.getName(), funcTy);

    // inline the entity region in the new llvm function
    rewriter.inlineRegionBefore(entityOp.getBody(), llvmFunc.getBody(),
                                llvmFunc.end());

    // insert state argument in first position
    llvmFunc.getBody().front().insertArgument(
        llvmFunc.getBody().front().getArguments().begin(), i8PtrTy);

    // erase original operation
    rewriter.eraseOp(op);

    return success();
  }
};

/// Convert an `"llhd.terminator" operation to `llvm.return`.
struct TerminatorOpConversion : public ConvertToLLVMPattern {
  explicit TerminatorOpConversion(MLIRContext *ctx,
                                  LLVMTypeConverter &typeConverter)
      : ConvertToLLVMPattern(llhd::TerminatorOp::getOperationName(), ctx,
                             typeConverter) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    // just replace the original op with return void
    rewriter.replaceOpWithNewOp<LLVM::ReturnOp>(op, ValueRange());

    return success();
  }
};

//===----------------------------------------------------------------------===//
// Signal conversions
//===----------------------------------------------------------------------===//

/// Convert an `llhd.sig` operation to LLVM. The result is a library call to the
/// `@alloc_signal` function, which allocates a new signal slot in the state,
/// and returns its index (which is used to retrieve the signal by other
/// operations). The library call takes also takes the expected index of the
/// signal as argument, which is used to test wheter the signal has already been
/// allocated.
struct SigOpConversion : public ConvertToLLVMPattern {
  explicit SigOpConversion(MLIRContext *ctx, LLVMTypeConverter &typeConverter)
      : ConvertToLLVMPattern(llhd::SigOp::getOperationName(), ctx,
                             typeConverter) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    // get adapted opreands
    OperandAdaptor<SigOp> transformed(operands);
    // get sig operation
    auto sigOp = cast<SigOp>(op);
    // get parent module
    auto module = op->getParentOfType<ModuleOp>();

    // collect llvm types
    auto voidTy = getVoidType();
    auto i8PtrTy = getVoidPtrType();
    auto i1Ty = LLVM::LLVMType::getIntNTy(typeConverter.getDialect(), 1);
    auto i32Ty = LLVM::LLVMType::getInt32Ty(typeConverter.getDialect());

    // get or insert library call definition
    auto sigFunc = module.lookupSymbol<LLVM::LLVMFuncOp>(libCall);
    if (!sigFunc) {
      OpBuilder moduleBuilder(module.getBodyRegion());
      // alloc_signal function signature: (i8* %state, i32 %sig_index, i1
      // %value) -> i32 %sig_index
      auto allocSigFuncTy =
          LLVM::LLVMType::getFunctionTy(i32Ty, {i8PtrTy, i32Ty, i1Ty}, false);
      sigFunc = moduleBuilder.create<LLVM::LLVMFuncOp>(rewriter.getUnknownLoc(),
                                                       libCall, allocSigFuncTy);
    }

    // get expected signal index as an attribute
    auto counterAttr = rewriter.getI32IntegerAttr(counter);
    // add signal index constant
    Value sigInd = rewriter.create<LLVM::ConstantOp>(
        op->getLoc(), LLVM::LLVMType::getInt32Ty(typeConverter.getDialect()),
        counterAttr);

    // get state
    Value statePtr = op->getParentOfType<LLVM::LLVMFuncOp>().getArgument(0);
    // build call arguments vector
    llvm::SmallVector<Value, 3> args({statePtr, sigInd, transformed.init()});
    // replace original operation with the library call
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(
        op, i32Ty, rewriter.getSymbolRefAttr(sigFunc), args);

    // increase index counter
    counter++;
    return success();
  }

private:
  const std::string libCall = "alloc_signal";
};

/// Convert an `llhd.prb` operation to LLVM. The result is a library call to the
/// `@probe_signal` function, which is then bitcast to the result type and
/// loaded. The library function declaration is inserted at the beginning of the
/// module if not already present.
struct PrbOpConversion : public ConvertToLLVMPattern {
  explicit PrbOpConversion(MLIRContext *ctx, LLVMTypeConverter &typeConverter)
      : ConvertToLLVMPattern(llhd::PrbOp::getOperationName(), ctx,
                             typeConverter) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    // get adapted operands
    OperandAdaptor<PrbOp> transformed(operands);
    // get probe operation
    auto prbOp = cast<PrbOp>(op);
    // get parent module
    auto module = op->getParentOfType<ModuleOp>();

    // collect llvm types
    auto voidTy = getVoidType();
    auto i8PtrTy = getVoidPtrType();
    auto i32Ty = LLVM::LLVMType::getInt32Ty(typeConverter.getDialect());

    // get or insert library call definition
    auto prbFunc = module.lookupSymbol<LLVM::LLVMFuncOp>(libCall);
    if (!prbFunc) {
      OpBuilder moduleBuilder(module.getBodyRegion());
      // `probe_signal` function signature: (i8* %state, i32 %sig_index) -> i8*
      // %value
      auto prbFuncTy =
          LLVM::LLVMType::getFunctionTy(i8PtrTy, {i8PtrTy, i32Ty}, false);
      prbFunc = moduleBuilder.create<LLVM::LLVMFuncOp>(rewriter.getUnknownLoc(),
                                                       libCall, prbFuncTy);
    }

    // get pointer to state from function arguments
    Value statePtr = op->getParentOfType<LLVM::LLVMFuncOp>().getArgument(0);
    // define library call arguments
    llvm::SmallVector<Value, 2> args({statePtr, transformed.signal()});
    // create library call
    auto probedPtr =
        rewriter
            .create<LLVM::CallOp>(op->getLoc(), getVoidPtrType(),
                                  rewriter.getSymbolRefAttr(prbFunc), args)
            .getResult(0);

    // get final type
    auto finalTy = typeConverter.convertType(prbOp.getType());
    // get pointer of final type
    auto finalTyPtr = finalTy.cast<LLVM::LLVMType>().getPointerTo();

    // create bitcast of library call result to pointer of final type
    auto bitcast =
        rewriter.create<LLVM::BitcastOp>(op->getLoc(), finalTyPtr, probedPtr);

    // create load of probed value
    rewriter.replaceOpWithNewOp<LLVM::LoadOp>(op, finalTy, bitcast);

    return success();
  }

private:
  const std::string libCall = "probe_signal";
};

/// Convert an `llhd.drv` operation to LLVM. The result is a library call to the
/// `@drive_signal` function, which declaration is inserted at the beginning of
/// the module if missing. New `llvm.mlir.constant`s operation are also
/// generated by extracting the required time values from the time operand. The
/// resulting operation should not use the original time constant operation,
/// making it possible to erase it.
struct DrvOpConversion : public ConvertToLLVMPattern {
  explicit DrvOpConversion(MLIRContext *ctx, LLVMTypeConverter &typeConverter)
      : ConvertToLLVMPattern(llhd::DrvOp::getOperationName(), ctx,
                             typeConverter) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    // get adapted operands;
    OperandAdaptor<DrvOp> transformed(operands);
    // get drive operation
    auto drvOp = cast<DrvOp>(op);
    // get parent module
    auto module = op->getParentOfType<ModuleOp>();

    // collect used llvm types
    auto voidTy = getVoidType();
    auto i8PtrTy = getVoidPtrType();
    auto i1Ty = LLVM::LLVMType::getIntNTy(typeConverter.getDialect(), 1);
    auto i32Ty = LLVM::LLVMType::getInt32Ty(typeConverter.getDialect());

    // get or insert drive library call
    auto drvFunc = module.lookupSymbol<LLVM::LLVMFuncOp>(libCall);
    if (!drvFunc) {
      OpBuilder moduleBuilder(module.getBodyRegion());
      // drv function signature: (i8* %state, i32 %sig_index, i1 %new_value, i32
      // %time) -> ()
      auto drvFuncTy = LLVM::LLVMType::getFunctionTy(
          voidTy, {i8PtrTy, i32Ty, i1Ty, i32Ty}, /*isVarArg=*/false);
      drvFunc = moduleBuilder.create<LLVM::LLVMFuncOp>(rewriter.getUnknownLoc(),
                                                       libCall, drvFuncTy);
    }

    // get constant time operation
    unsigned timeOp = cast<ConstOp>(drvOp.time().getDefiningOp())
                          .valueAttr()
                          .dyn_cast<TimeAttr>()
                          .getTime();
    // get real time as an attribute
    auto timeAttr = rewriter.getI32IntegerAttr(timeOp);
    // create new time const at the current operation location
    auto newTimeConst = rewriter.create<LLVM::ConstantOp>(
        op->getLoc(), LLVM::LLVMType::getInt32Ty(typeConverter.getDialect()),
        timeAttr);

    // get state pointer from function arguments
    Value statePtr = op->getParentOfType<LLVM::LLVMFuncOp>().getArgument(0);
    // define library call arguments
    llvm::SmallVector<Value, 3> args(
        {statePtr, transformed.signal(), transformed.value(), newTimeConst});
    // create library call
    rewriter.create<LLVM::CallOp>(op->getLoc(), voidTy,
                                  rewriter.getSymbolRefAttr(drvFunc), args);
    // erase original operation
    rewriter.eraseOp(op);

    return success();
  }

private:
  const std::string libCall = "drive_signal";
};

//===----------------------------------------------------------------------===//
// Arithmetic conversions
//===----------------------------------------------------------------------===//

/// Convert an `llhd.not` operation. The result is an `llvm.xor` operation,
/// xor-ing the operand with all ones.
struct NotOpConversion : public ConvertToLLVMPattern {
  explicit NotOpConversion(MLIRContext *ctx, LLVMTypeConverter &typeConverter)
      : ConvertToLLVMPattern(llhd::NotOp::getOperationName(), ctx,
                             typeConverter) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    // get adapted operands
    OperandAdaptor<NotOp> transformed(operands);
    // get `llhd.not` operation
    auto notOp = cast<NotOp>(op);
    // get llvm types
    auto i1Ty = LLVM::LLVMType::getIntNTy(typeConverter.getDialect(), 1);

    // get rhs operand (all ones)
    auto rhs = rewriter.getBoolAttr(true);
    auto rhsConst = rewriter.create<LLVM::ConstantOp>(op->getLoc(), i1Ty, rhs);

    // replace original op with llvm equivalent
    rewriter.replaceOpWithNewOp<LLVM::XOrOp>(
        op, typeConverter.convertType(notOp.getType()), transformed.value(),
        rhsConst);

    return success();
  }
};

//===----------------------------------------------------------------------===//
// Constant conversions
//===----------------------------------------------------------------------===//

/// Lower an LLHD constant operation to LLVM. Time constant are treated as a
/// special case, by just erasing them. Operations that use time constants are
/// assumed to extract and convert the elements they require. Remaining const
/// types are lowered to an equivalent `llvm.mlir.constant` operation.
struct ConstOpConversion : public ConvertToLLVMPattern {
  explicit ConstOpConversion(MLIRContext *ctx, LLVMTypeConverter &typeConverter)
      : ConvertToLLVMPattern(llhd::ConstOp::getOperationName(), ctx,
                             typeConverter) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operand,
                  ConversionPatternRewriter &rewriter) const override {
    // get ConstOp
    auto constOp = cast<ConstOp>(op);
    // get const's attribute
    auto attr = constOp.value();
    // treat time const special case
    if (!attr.getType().isa<IntegerType>()) {
      rewriter.eraseOp(op);
      return success();
    }
    // get llvm converted type
    auto intType = typeConverter.convertType(attr.getType());
    // replace op with llvm constant op
    rewriter.replaceOpWithNewOp<LLVM::ConstantOp>(op, intType,
                                                  constOp.valueAttr());

    return success();
  }
};

struct LLHDToLLVMLoweringPass
    : public ConvertLLHDToLLVMBase<LLHDToLLVMLoweringPass> {
  void runOnOperation() override;
};
} // namespace

void llhd::populateLLHDToLLVMConversionPatterns(
    LLVMTypeConverter &converter, OwningRewritePatternList &patterns) {
  MLIRContext *ctx = converter.getDialect()->getContext();

  // constant conversion patterns
  patterns.insert<ConstOpConversion>(ctx, converter);
  // arithmetic conversion patterns
  patterns.insert<NotOpConversion>(ctx, converter);
  // unit conversion patterns
  patterns.insert<EntityOpConversion, TerminatorOpConversion>(ctx, converter);
  // signal conversion patterns
  patterns.insert<SigOpConversion, PrbOpConversion, DrvOpConversion>(ctx,
                                                                     converter);
}

void LLHDToLLVMLoweringPass::runOnOperation() {
  auto module = getOperation();
  OwningRewritePatternList patterns;
  auto converter = mlir::LLVMTypeConverter(&getContext());

  // get conversion patterns
  populateStdToLLVMConversionPatterns(converter, patterns);
  populateLLHDToLLVMConversionPatterns(converter, patterns);

  // define target
  LLVMConversionTarget target(getContext());
  target.addLegalDialect<LLVM::LLVMDialect>();
  target.addLegalOp<ModuleOp, ModuleTerminatorOp>();

  // apply conversion
  if (failed(applyFullConversion(getOperation(), target, patterns, &converter)))
    signalPassFailure();
}

/// Create an LLHD to LLVM conversion pass.
std::unique_ptr<OperationPass<ModuleOp>>
mlir::llhd::createConvertLLHDToLLVMPass() {
  return std::make_unique<LLHDToLLVMLoweringPass>();
}