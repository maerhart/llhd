#include "Target/VerilogPrinter.h"
#include "mlir/Translation.h"

using namespace mlir;

namespace mlir {
namespace llhd {

void registerToVerilogTranslation() {
  TranslateFromMLIRRegistration registration(
      "llhd-to-verilog", [](ModuleOp module, raw_ostream &output) {
        formatted_raw_ostream out(output);
        llhd::VerilogPrinter printer(out);
        printer.printModule(module);
        return success();
      });
}

} // namespace llhd
} // namespace mlir
