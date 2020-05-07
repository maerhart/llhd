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
//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

/// Get an existing global string
Value getGlobalString(Location loc, ConversionPatternRewriter &rewriter,
                      LLVMTypeConverter &typeConverter, LLVM::GlobalOp &str) {
  auto i8PtrTy = LLVM::LLVMType::getInt8PtrTy(typeConverter.getDialect());
  auto i32Ty = LLVM::LLVMType::getInt32Ty(typeConverter.getDialect());

  auto addr = rewriter.create<LLVM::AddressOfOp>(
      loc, str.getType().getPointerTo(), str.getName());
  auto idx = rewriter.create<LLVM::ConstantOp>(loc, i32Ty,
                                               rewriter.getI32IntegerAttr(0));
  llvm::SmallVector<Value, 2> idxs({idx, idx});
  auto gep = rewriter.create<LLVM::GEPOp>(loc, i8PtrTy, addr, idxs);
  return gep;
}

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

    // get or create owner name string
    Value owner;
    llvm::StringRef parent = op->getParentOfType<LLVM::LLVMFuncOp>().getName();
    auto parentSym =
        module.lookupSymbol<LLVM::GlobalOp>("entity." + parent.str());
    if (!parentSym) {
      owner = LLVM::createGlobalString(
          op->getLoc(), rewriter, "entity." + parent.str(), parent.str() + '\0',
          LLVM::Linkage::Internal, typeConverter.getDialect());
    } else {
      owner = getGlobalString(op->getLoc(), rewriter, typeConverter, parentSym);
    }

    // get or create signal name
    Value sigName;
    auto sigSym =
        module.lookupSymbol<LLVM::GlobalOp>("sig." + sigOp.name().str());
    if (!sigSym) {
      sigName = LLVM::createGlobalString(
          op->getLoc(), rewriter, "sig." + sigOp.name().str(),
          sigOp.name().str() + '\0', LLVM::Linkage::Internal,
          typeConverter.getDialect());
    } else {
      sigName = getGlobalString(op->getLoc(), rewriter, typeConverter, sigSym);
    }

    // get or insert library call definition
    auto sigFunc = module.lookupSymbol<LLVM::LLVMFuncOp>(libCall);
    if (!sigFunc) {
      OpBuilder moduleBuilder(module.getBodyRegion());
      // alloc_signal function signature: (i8* %state, i8* %sig_name, i8*
      // %sig_owner, i32 %value) -> i32 %sig_index
      auto allocSigFuncTy = LLVM::LLVMType::getFunctionTy(
          i32Ty, {i8PtrTy, i8PtrTy, i8PtrTy, i32Ty}, false);
      sigFunc = moduleBuilder.create<LLVM::LLVMFuncOp>(rewriter.getUnknownLoc(),
                                                       libCall, allocSigFuncTy);
    }

    // extend value to i32
    auto zext =
        rewriter.create<LLVM::ZExtOp>(op->getLoc(), i32Ty, transformed.init());
    // get state
    Value statePtr = op->getParentOfType<LLVM::LLVMFuncOp>().getArgument(0);
    // build call arguments vector
    llvm::SmallVector<Value, 4> args({statePtr, sigName, owner, zext});
    // replace original operation with the library call
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(
        op, i32Ty, rewriter.getSymbolRefAttr(sigFunc), args);

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
      // drv function signature: (i8* %state, i32 %sig_index, i32 %new_value,
      // i32 %time, i32 %delta, i32 %eps) -> ()
      auto drvFuncTy = LLVM::LLVMType::getFunctionTy(
          voidTy, {i8PtrTy, i32Ty, i32Ty, i32Ty, i32Ty, i32Ty},
          /*isVarArg=*/false);
      drvFunc = moduleBuilder.create<LLVM::LLVMFuncOp>(rewriter.getUnknownLoc(),
                                                       libCall, drvFuncTy);
    }

    // get constant time operation
    auto timeAttr = cast<llhd::ConstOp>(drvOp.time().getDefiningOp())
                        .valueAttr()
                        .dyn_cast<TimeAttr>();
    // get real time as an attribute
    auto realTimeAttr = rewriter.getI32IntegerAttr(timeAttr.getTime());
    // create new time const operation
    auto realTimeConst = rewriter.create<LLVM::ConstantOp>(
        op->getLoc(), LLVM::LLVMType::getInt32Ty(typeConverter.getDialect()),
        realTimeAttr);
    // get the delta step as an attribute
    auto deltaAttr = rewriter.getI32IntegerAttr(timeAttr.getDelta());
    // create new delta const operation
    auto deltaConst = rewriter.create<LLVM::ConstantOp>(
        op->getLoc(), LLVM::LLVMType::getInt32Ty(typeConverter.getDialect()),
        deltaAttr);
    // get the epsilon step as an attribute
    auto epsAttr = rewriter.getI32IntegerAttr(timeAttr.getEps());
    // create new eps const operation
    auto epsConst = rewriter.create<LLVM::ConstantOp>(
        op->getLoc(), LLVM::LLVMType::getInt32Ty(typeConverter.getDialect()),
        epsAttr);

    // extend value to i32
    auto zext =
        rewriter.create<LLVM::ZExtOp>(op->getLoc(), i32Ty, transformed.value());
    // get state pointer from function arguments
    Value statePtr = op->getParentOfType<LLVM::LLVMFuncOp>().getArgument(0);
    // define library call arguments
    llvm::SmallVector<Value, 6> args({statePtr, transformed.signal(), zext,
                                      realTimeConst, deltaConst, epsConst});
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
// Bitwise conversions
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
    // get integer width
    unsigned width = notOp.getType().getIntOrFloatBitWidth();
    // get llvm types
    auto iTy = LLVM::LLVMType::getIntNTy(typeConverter.getDialect(), width);

    // get mask operand
    APInt mask(width, 0);
    mask.setAllBits();
    auto rhs = rewriter.getIntegerAttr(rewriter.getIntegerType(width), mask);
    auto rhsConst = rewriter.create<LLVM::ConstantOp>(op->getLoc(), iTy, rhs);

    // replace original op with llvm equivalent
    rewriter.replaceOpWithNewOp<LLVM::XOrOp>(
        op, typeConverter.convertType(notOp.getType()), transformed.value(),
        rhsConst);

    return success();
  }
};

/// Convert an `llhd.shr` operation to llvm. All the operands are extended to
/// the width obtained by combining the hidden and base values. This combined
/// value is then shifted (exposing the hidden value) and truncated to the base
/// length
struct ShrOpConversion : public ConvertToLLVMPattern {
  explicit ShrOpConversion(MLIRContext *ctx, LLVMTypeConverter &typeConverter)
      : ConvertToLLVMPattern(llhd::ShrOp::getOperationName(), ctx,
                             typeConverter) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {

    OperandAdaptor<ShrOp> transformed(operands);
    auto shrOp = cast<ShrOp>(op);
    assert(!(shrOp.getType().getKind() == llhd::LLHDTypes::Sig) &&
           "sig not yet supported");

    // get widths
    auto baseWidth = shrOp.getType().getIntOrFloatBitWidth();
    auto hdnWidth = shrOp.hidden().getType().getIntOrFloatBitWidth();
    auto full = baseWidth + hdnWidth;

    auto tmpTy = LLVM::LLVMType::getIntNTy(typeConverter.getDialect(), full);

    // extend all operands to the base and hidden combined  width
    auto baseZext =
        rewriter.create<LLVM::ZExtOp>(op->getLoc(), tmpTy, transformed.base());
    auto hdnZext = rewriter.create<LLVM::ZExtOp>(op->getLoc(), tmpTy,
                                                 transformed.hidden());
    auto amntZext = rewriter.create<LLVM::ZExtOp>(op->getLoc(), tmpTy,
                                                  transformed.amount());

    // shift hidden operand to prepend to full value
    auto hdnShAmnt = rewriter.create<LLVM::ConstantOp>(
        op->getLoc(), tmpTy,
        rewriter.getIntegerAttr(rewriter.getIntegerType(full), baseWidth));
    auto hdnSh =
        rewriter.create<LLVM::ShlOp>(op->getLoc(), tmpTy, hdnZext, hdnShAmnt);

    // combine base and hidden operands
    auto combined =
        rewriter.create<LLVM::OrOp>(op->getLoc(), tmpTy, hdnSh, baseZext);

    // perform the right shift
    auto shifted =
        rewriter.create<LLVM::LShrOp>(op->getLoc(), tmpTy, combined, amntZext);

    // truncate to final width
    rewriter.replaceOpWithNewOp<LLVM::TruncOp>(op, transformed.base().getType(),
                                               shifted);

    return success();
  }
};

/// Convert an `llhd.shr` operation to llvm. All the operands are extended to
/// the width obtained by combining the hidden and base values. This combined
/// value is then shifted right by `hidden_width - amount` (exposing the hidden
/// value) and truncated to the base length
struct ShlOpConversion : public ConvertToLLVMPattern {
  explicit ShlOpConversion(MLIRContext *ctx, LLVMTypeConverter &typeConverter)
      : ConvertToLLVMPattern(llhd::ShlOp::getOperationName(), ctx,
                             typeConverter) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {

    OperandAdaptor<ShlOp> transformed(operands);
    auto shlOp = cast<ShlOp>(op);
    assert(!(shlOp.getType().getKind() == llhd::LLHDTypes::Sig) &&
           "sig not yet supported");

    // get widths
    auto baseWidth = shlOp.getType().getIntOrFloatBitWidth();
    auto hdnWidth = shlOp.hidden().getType().getIntOrFloatBitWidth();
    auto full = baseWidth + hdnWidth;

    auto tmpTy = LLVM::LLVMType::getIntNTy(typeConverter.getDialect(), full);

    // extend all operands to the base and hidden combined  width
    auto baseZext =
        rewriter.create<LLVM::ZExtOp>(op->getLoc(), tmpTy, transformed.base());
    auto hdnZext = rewriter.create<LLVM::ZExtOp>(op->getLoc(), tmpTy,
                                                 transformed.hidden());
    auto amntZext = rewriter.create<LLVM::ZExtOp>(op->getLoc(), tmpTy,
                                                  transformed.amount());

    // shift hidden operand to
    // prepend to full value
    auto hdnWidthConst = rewriter.create<LLVM::ConstantOp>(
        op->getLoc(), tmpTy,
        rewriter.getIntegerAttr(rewriter.getIntegerType(full), hdnWidth));
    auto baseSh = rewriter.create<LLVM::ShlOp>(op->getLoc(), tmpTy, baseZext,
                                               hdnWidthConst);

    // combine base and hidden operands
    auto combined =
        rewriter.create<LLVM::OrOp>(op->getLoc(), tmpTy, baseSh, hdnZext);

    // get the final right shift amount
    auto shrAmnt = rewriter.create<LLVM::SubOp>(op->getLoc(), tmpTy,
                                                hdnWidthConst, amntZext);

    // perform the right shift
    auto shifted =
        rewriter.create<LLVM::LShrOp>(op->getLoc(), tmpTy, combined, shrAmnt);

    // truncate to final width
    rewriter.replaceOpWithNewOp<LLVM::TruncOp>(op, transformed.base().getType(),
                                               shifted);

    return success();
  }
};

using AndOpConversion = OneToOneConvertToLLVMPattern<llhd::AndOp, LLVM::AndOp>;
using OrOpConversion = OneToOneConvertToLLVMPattern<llhd::OrOp, LLVM::OrOp>;
using XorOpConversion = OneToOneConvertToLLVMPattern<llhd::XorOp, LLVM::XOrOp>;

//===----------------------------------------------------------------------===//
// Constant conversions
//===----------------------------------------------------------------------===//

/// Lower an LLHD constant operation to LLVM. Time constant are treated as a
/// special case, by just erasing them. Operations that use time constants
/// are assumed to extract and convert the elements they require. Remaining
/// const types are lowered to an equivalent `llvm.mlir.constant` operation.
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
  // bitwise conversion patterns
  patterns.insert<NotOpConversion, ShrOpConversion, ShlOpConversion>(ctx,
                                                                     converter);
  // patterns.insert<NotOpConversion, AndOpConversion>(ctx, converter);
  patterns.insert<AndOpConversion, OrOpConversion, XorOpConversion>(converter);
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
