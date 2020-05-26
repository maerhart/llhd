#include "LLHDToLLVM/LLHDToLLVM.h"
#include "Dialect/LLHD/LLHDDialect.h"
#include "Dialect/LLHD/LLHDOps.h"

#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
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
// keep a counter to infer a signal's index in his entity's signal table
static int signalCounter = 0;
//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

/// Get an existing global string
Value getGlobalString(Location loc, OpBuilder &builder,
                      LLVMTypeConverter &typeConverter, LLVM::GlobalOp &str) {
  auto i8PtrTy = LLVM::LLVMType::getInt8PtrTy(typeConverter.getDialect());
  auto i32Ty = LLVM::LLVMType::getInt32Ty(typeConverter.getDialect());

  auto addr = builder.create<LLVM::AddressOfOp>(
      loc, str.getType().getPointerTo(), str.getName());
  auto idx = builder.create<LLVM::ConstantOp>(loc, i32Ty,
                                              builder.getI32IntegerAttr(0));
  llvm::SmallVector<Value, 2> idxs({idx, idx});
  auto gep = builder.create<LLVM::GEPOp>(loc, i8PtrTy, addr, idxs);
  return gep;
}

/// Looks up a symbol and inserts a new functino at the beginning of the
/// module's region in case the function does not exists. If
/// insertBodyAndTerminator is set, also adds the entry block and return
/// terminator
LLVM::LLVMFuncOp getOrInsertFunction(ModuleOp &module,
                                     ConversionPatternRewriter &rewriter,
                                     std::string name, LLVM::LLVMType signature,
                                     bool insertBodyAndTerminator = false) {
  auto func = module.lookupSymbol<LLVM::LLVMFuncOp>(name);
  if (!func) {
    OpBuilder moduleBuilder(module.getBodyRegion());
    func = moduleBuilder.create<LLVM::LLVMFuncOp>(rewriter.getUnknownLoc(),
                                                  name, signature);
    if (insertBodyAndTerminator) {
      func.addEntryBlock();
      OpBuilder b(func.getBody());
      auto ret =
          b.create<LLVM::ReturnOp>(rewriter.getUnknownLoc(), ValueRange());
    }
  }
  return func;
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
    // reset signal counter
    signalCounter = 0;
    // get adapted operands
    OperandAdaptor<EntityOp> transformed(operands);
    // get entity operation
    auto entityOp = cast<EntityOp>(op);

    // collect llvm types
    auto voidTy = getVoidType();
    auto i8PtrTy = getVoidPtrType();
    auto i32Ty = LLVM::LLVMType::getInt32Ty(&getDialect());

    // have an intermediate signature conversion to add the arguments for the
    // state, signal table and argument table poitner arguments
    LLVMTypeConverter::SignatureConversion intermediate(
        entityOp.getNumArguments());
    // add state, signal table and arguments table arguments
    intermediate.addInputs(
        ArrayRef<Type>({i8PtrTy, i32Ty.getPointerTo(), i32Ty.getPointerTo()}));
    for (int i = 0; i < entityOp.getNumArguments(); i++)
      intermediate.addInputs(i, voidTy);
    rewriter.applySignatureConversion(&entityOp.getBody(), intermediate);

    OpBuilder bodyBuilder =
        OpBuilder::atBlockBegin(&entityOp.getBlocks().front());
    LLVMTypeConverter::SignatureConversion final(
        intermediate.getConvertedTypes().size());
    final.addInputs(0, i8PtrTy);
    final.addInputs(1, i32Ty.getPointerTo());
    final.addInputs(2, i32Ty.getPointerTo());

    for (int i = 0; i < entityOp.getNumArguments(); i++) {
      // create gep and load operations from arguments table for each original
      // argument
      auto index = bodyBuilder.create<LLVM::ConstantOp>(
          rewriter.getUnknownLoc(), i32Ty, rewriter.getI32IntegerAttr(i));
      auto bitcast = bodyBuilder.create<LLVM::GEPOp>(
          rewriter.getUnknownLoc(), i32Ty.getPointerTo(),
          entityOp.getArgument(2), ArrayRef<Value>(index));
      auto load =
          bodyBuilder.create<LLVM::LoadOp>(rewriter.getUnknownLoc(), bitcast);
      // remap i-th original argument to the loaded value
      final.remapInput(i + 3, load.getResult());
    }

    rewriter.applySignatureConversion(&entityOp.getBody(), final);

    // converted entity signature
    auto funcTy = LLVM::LLVMType::getFunctionTy(
        voidTy, {i8PtrTy, i32Ty.getPointerTo(), i32Ty.getPointerTo()}, false);
    // // create the llvm function
    auto llvmFunc = rewriter.create<LLVM::LLVMFuncOp>(
        op->getLoc(), entityOp.getName(), funcTy);

    // inline the entity region in the new llvm function
    rewriter.inlineRegionBefore(entityOp.getBody(), llvmFunc.getBody(),
                                llvmFunc.end());

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

/// Lower an llhd.inst operation to llvm. This generates malloc calls and
/// alloc_signal calls (to store the pointer into the state) for each signal in
/// the instantiated entity.
struct InstOpConversion : public ConvertToLLVMPattern {
  explicit InstOpConversion(MLIRContext *ctx, LLVMTypeConverter &typeConverter)
      : ConvertToLLVMPattern(InstOp::getOperationName(), ctx, typeConverter) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    // get inst operation
    auto instOp = cast<InstOp>(op);
    // get parent module
    auto module = op->getParentOfType<ModuleOp>();

    auto voidTy = getVoidType();
    auto i8PtrTy = getVoidPtrType();
    auto i32Ty = LLVM::LLVMType::getInt32Ty(&getDialect());
    auto i64Ty = LLVM::LLVMType::getInt64Ty(&getDialect());

    // init function signature: (i8* %state) -> void
    auto initFuncTy = LLVM::LLVMType::getFunctionTy(voidTy, {i8PtrTy}, false);
    auto initFunc =
        getOrInsertFunction(module, rewriter, initCall, initFuncTy, true);

    //! get or insert malloc function definition
    // malloc function signature: (i64 %size) -> i8* %pointer
    auto mallocSigFuncTy =
        LLVM::LLVMType::getFunctionTy(i8PtrTy, {i64Ty}, false);
    auto mallFunc =
        getOrInsertFunction(module, rewriter, "malloc", mallocSigFuncTy);

    //! get or insert library call definition
    // alloc_signal function signature: (i8* %state, i8* %sig_name, i8*
    // %sig_owner, i32 %value) -> i32 %sig_index
    auto allocSigFuncTy = LLVM::LLVMType::getFunctionTy(
        i32Ty, {i8PtrTy, i32Ty, i8PtrTy, i8PtrTy, i64Ty}, false);
    auto sigFunc =
        getOrInsertFunction(module, rewriter, allocCall, allocSigFuncTy);

    // get builder with insertion point before the init function terminator
    OpBuilder initBuilder =
        OpBuilder::atBlockTerminator(&initFunc.getBody().getBlocks().front());

    if (auto child = module.lookupSymbol<EntityOp>(instOp.callee())) {
      // use the instance name to retrieve the signal table of the instance, and
      // insert the pointers in the global signal table
      auto ownerName = instOp.name();
      //! get or create owner name string
      Value owner;

      auto parentSym =
          module.lookupSymbol<LLVM::GlobalOp>("instance." + ownerName.str());
      if (!parentSym) {
        owner = LLVM::createGlobalString(
            rewriter.getUnknownLoc(), initBuilder,
            "instance." + ownerName.str(), ownerName.str() + '\0',
            LLVM::Linkage::Internal, typeConverter.getDialect());
        parentSym =
            module.lookupSymbol<LLVM::GlobalOp>("instance." + ownerName.str());
      } else {
        owner = getGlobalString(rewriter.getUnknownLoc(), initBuilder,
                                typeConverter, parentSym);
      }

      // walk over the unit and generate mallocs for each one of its signals
      // index of the signal in the unit's signal table
      int initCounter = 0;
      child.walk([&](Operation *op) -> WalkResult {
        if (auto sigOp = dyn_cast<SigOp>(op)) {
          // get index constant of the signal in the unit's signal table
          auto indexConst = initBuilder.create<LLVM::ConstantOp>(
              rewriter.getUnknownLoc(), i32Ty,
              rewriter.getI32IntegerAttr(initCounter));
          initCounter++;

          //! add signal allocation to the init function
          // clone and insert init's defining operation (assmued to be a
          // constant op)
          auto initDef =
              initBuilder.insert(sigOp.init().getDefiningOp()->clone())
                  ->getResult(0);
          // malloc required space
          int size =
              std::ceil(sigOp.init().getType().getIntOrFloatBitWidth() / 8.0);
          auto sizeConst = initBuilder.create<LLVM::ConstantOp>(
              rewriter.getUnknownLoc(), i64Ty,
              rewriter.getI64IntegerAttr(size));
          llvm::SmallVector<Value, 1> margs({sizeConst});
          auto mall = initBuilder
                          .create<LLVM::CallOp>(
                              rewriter.getUnknownLoc(), i8PtrTy,
                              rewriter.getSymbolRefAttr(mallFunc), margs)
                          .getResult(0);
          auto bitcast =
              initBuilder
                  .create<LLVM::BitcastOp>(
                      rewriter.getUnknownLoc(),
                      typeConverter.convertType(sigOp.init().getType())
                          .cast<LLVM::LLVMType>()
                          .getPointerTo(),
                      mall)
                  .getResult();
          auto initStore = initBuilder.create<LLVM::StoreOp>(
              rewriter.getUnknownLoc(), initDef, bitcast);
          Value initStatePtr = initFunc.getArgument(0);
          llvm::SmallVector<Value, 5> args(
              {initStatePtr, indexConst, owner, mall, sizeConst});
          initBuilder.create<LLVM::CallOp>(rewriter.getUnknownLoc(), i32Ty,
                                           rewriter.getSymbolRefAttr(sigFunc),
                                           args);
        }
        return WalkResult::advance();
      });
    }

    rewriter.eraseOp(op);
    return success();
  }

private:
  const std::string allocCall = "alloc_signal";
  const std::string initCall = "llhd_init";
};

//===----------------------------------------------------------------------===//
// Signal conversions
//===----------------------------------------------------------------------===//

/// Convert an `llhd.sig` operation to LLVM. The i-th signal of an entity get's
/// lowered to a load of the i-th element of the signal table, passed as an
/// argument.
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
    auto i64Ty = LLVM::LLVMType::getInt64Ty(typeConverter.getDialect());

    //! load the signal's index from the signal table
    // get signal table argument
    Value sigTablePtr = op->getParentOfType<LLVM::LLVMFuncOp>().getArgument(1);

    // get index in the signal table and increase counter
    auto indexConst = rewriter.create<LLVM::ConstantOp>(
        op->getLoc(), i32Ty, rewriter.getI32IntegerAttr(signalCounter));
    signalCounter++;

    // gep&load index
    auto gep =
        rewriter.create<LLVM::GEPOp>(op->getLoc(), i32Ty.getPointerTo(),
                                     sigTablePtr, ArrayRef<Value>(indexConst));
    rewriter.replaceOpWithNewOp<LLVM::LoadOp>(op, gep);

    return success();
  }
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
    auto i64Ty = LLVM::LLVMType::getInt64Ty(typeConverter.getDialect());

    // get or insert drive library call
    auto drvFunc = module.lookupSymbol<LLVM::LLVMFuncOp>(libCall);
    if (!drvFunc) {
      OpBuilder moduleBuilder(module.getBodyRegion());
      // drv function signature: (i8* %state, i32 %sig_index, i32 %new_value,
      // i32 %time, i32 %delta, i32 %eps) -> ()
      auto drvFuncTy = LLVM::LLVMType::getFunctionTy(
          voidTy, {i8PtrTy, i32Ty, i8PtrTy, i64Ty, i32Ty, i32Ty, i32Ty},
          /*isVarArg=*/false);
      drvFunc = moduleBuilder.create<LLVM::LLVMFuncOp>(rewriter.getUnknownLoc(),
                                                       libCall, drvFuncTy);
    }

    //! get new value pointer
    auto oneConst = rewriter.create<LLVM::ConstantOp>(
        op->getLoc(), i32Ty, rewriter.getI32IntegerAttr(1));
    auto alloca = rewriter.create<LLVM::AllocaOp>(
        op->getLoc(),
        transformed.value().getType().cast<LLVM::LLVMType>().getPointerTo(),
        oneConst, 4);
    rewriter.create<LLVM::StoreOp>(op->getLoc(), transformed.value(), alloca);
    auto bc = rewriter.create<LLVM::BitcastOp>(op->getLoc(), i8PtrTy, alloca);

    //! get value size
    // get value size in bytes, rounded to the next byte
    int size = std::ceil(drvOp.value().getType().getIntOrFloatBitWidth() / 8.0);
    auto sizeConst = rewriter.create<LLVM::ConstantOp>(
        op->getLoc(), i64Ty, rewriter.getI64IntegerAttr(size));

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

    // get state pointer from function arguments
    Value statePtr = op->getParentOfType<LLVM::LLVMFuncOp>().getArgument(0);
    // define library call arguments
    llvm::SmallVector<Value, 7> args({statePtr, transformed.signal(), bc,
                                      sizeConst, realTimeConst, deltaConst,
                                      epsConst});
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

  // partial conversion of only the inst op
  patterns.insert<InstOpConversion>(&getContext(), converter);
  // define target
  LLVMConversionTarget target(getContext());
  target.addIllegalOp<InstOp>();

  // apply partial conversion
  if (failed(
          applyPartialConversion(getOperation(), target, patterns, &converter)))
    signalPassFailure();

  // setup full conversion
  populateStdToLLVMConversionPatterns(converter, patterns);
  populateLLHDToLLVMConversionPatterns(converter, patterns);

  target.addLegalDialect<LLVM::LLVMDialect>();
  target.addLegalOp<ModuleOp, ModuleTerminatorOp>();

  // apply full conversion
  if (failed(applyFullConversion(getOperation(), target, patterns, &converter)))
    signalPassFailure();
}

/// Create an LLHD to LLVM conversion pass.
std::unique_ptr<OperationPass<ModuleOp>>
mlir::llhd::createConvertLLHDToLLVMPass() {
  return std::make_unique<LLHDToLLVMLoweringPass>();
}
