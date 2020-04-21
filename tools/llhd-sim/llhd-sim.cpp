#include "Dialect/LLHD/LLHDDialect.h"
#include "LLHDToLLVM/LLHDToLLVM.h"
#include "Simulator/Engine.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Target/LLVMIR.h"

#include "llvm/Support/CommandLine.h"

using namespace llvm;
using namespace mlir;

static cl::opt<std::string>
    inputFilename(cl::Positional, cl::desc("<input-file>"), cl::init("-"));

static cl::opt<int> nSteps("n", cl::desc("Set the maximum number of steps"),
                           cl::value_desc("max-steps"));

static cl::opt<bool>
    dumpLLVMDialect("dump-llvm-dialect",
                    cl::desc("Dump the LLVM IR dialect module"));

static cl::opt<bool> dumpLLVM("dump-llvm-ir",
                              cl::desc("Dump the LLVM IR module"));

static cl::opt<bool> dumpMLIR("dump-mlir",
                              cl::desc("Dump the original MLIR module"));

int parseMLIR(MLIRContext &context, OwningModuleRef &module) {
  module = parseSourceFile(inputFilename, &context);
  if (!module)
    return 1;
  return 0;
}

int dumpLLVMIR(mlir::ModuleOp module) {

  // Translate the module, that contains the LLVM dialect, to LLVM IR.
  auto llvmModule = mlir::translateModuleToLLVMIR(module);
  if (!llvmModule) {
    llvm::errs() << "Failed to emit LLVM IR\n";
    return -1;
  }

  llvm::errs() << *llvmModule << "\n";
  return 0;
}

int main(int argc, char **argv) {
  registerDialect<llhd::LLHDDialect>();
  registerDialect<LLVM::LLVMDialect>();

  llhd::initLLHDToLLVMPass();

  cl::ParseCommandLineOptions(argc, argv, "LLHD simulator\n");

  // Set up the input file.
  std::string errorMessage;
  auto file = openInputFile(inputFilename, &errorMessage);
  if (!file) {
    llvm::errs() << errorMessage << "\n";
    return 1;
  }

  // parse input file
  MLIRContext context;
  OwningModuleRef module;

  if (parseMLIR(context, module))
    return 1;

  if (dumpMLIR) {
    module->dump();
    llvm::errs() << "\n";
    return 0;
  }

  mlir::PassManager pm(&context);
  pm.addPass(llhd::createConvertLLHDToLLVMPass());
  pm.run(*module);

  if (dumpLLVMDialect) {
    module->dump();
    llvm::errs() << "\n";
    return 0;
  };

  if (dumpLLVM) {
    dumpLLVMIR(*module);
    return 0;
  }

  llhd::sim::Engine engine(*module);
  engine.simulate(nSteps);

  return 0;
}
