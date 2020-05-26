#include "Simulator/Engine.h"
#include "LLHDToLLVM/LLHDToLLVM.h"
#include "Simulator/State.h"

#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/IR/Module.h"
#include "mlir/Pass/PassManager.h"

#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>

using namespace mlir;
using namespace llhd::sim;

Engine::Engine(llvm::raw_ostream &out, OwningModuleRef &module,
               MLIRContext &context, std::string root)
    : out(out), root(root) {
  state = std::make_unique<State>();

  buildLayout(*module);

  auto rootEntity = module->lookupSymbol<EntityOp>(root);

  // insert explicit instantiation of root
  OpBuilder insertInst =
      OpBuilder::atBlockTerminator(&rootEntity.getBody().getBlocks().front());
  insertInst.create<InstOp>(rootEntity.getBlocks().front().back().getLoc(),
                            llvm::None, root, root, ArrayRef<Value>(),
                            ArrayRef<Value>());

  // add 0-time event
  state->queue.push(Slot(Time()));

  mlir::PassManager pm(&context);
  pm.addPass(llhd::createConvertLLHDToLLVMPass());
  pm.run(*module);

  this->module = *module;

  // init jit
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  auto maybeEngine = mlir::ExecutionEngine::create(this->module);
  assert(maybeEngine && "failed to create JIT");
  engine = std::move(*maybeEngine);
}

int Engine::simulate(int n) {
  assert(engine && "engine not found");
  assert(state && "state not found");

  // initialize simulation state
  auto invocationResult = engine->invoke("llhd_init", state);
  if (invocationResult) {
    llvm::errs() << "Failed invocation of llhd_init: " << invocationResult;
    return -1;
  }

  int i = 0;

  // keep track of instances that need to wakeup
  std::vector<std::string> wakeupQueue;
  // start with all entities in queue
  for (auto k : state->instances.keys())
    wakeupQueue.push_back(k.str());

  while (!state->queue.empty()) {
    if (n > 0 && i >= n) {
      break;
    }
    // pop queue
    auto pop = state->popQueue();

    // update time
    assert(state->time < pop.time || pop.time.time == 0);
    state->time = pop.time;

    // dump changes, only if actually changed
    unsigned actual = 0;
    for (auto change : pop.changes) {
      assert(state->signals[change.first].size == change.second.size() &&
             "size mismatch");
      bool equal = true;
      for (int i = 0; i < change.second.size(); i++) {
        equal &= (state->signals[change.first].value[i] == change.second[i]);
      }
      if (equal)
        continue;

      // owner is always triggered
      wakeupQueue.push_back(state->signals[change.first].owner);
      // add sensitive instances
      for (auto inst : state->signals[change.first].triggers) {
        wakeupQueue.push_back(inst);
      }
      state->updateSignal(change.first, change.second);
      state->dumpSignal(out, change.first);
      actual++;
    }
    // continue if no updates at non-zero time
    if (actual == 0 && !state->time.isZero()) {
      continue;
    }

    // run entities in wakeupqueue
    for (auto inst : wakeupQueue) {
      auto name = state->instances[inst].unit;
      auto sigTable = state->instances[inst].signalTable.data();
      auto sensitivityList = state->instances[inst].sensitivityList;
      auto outputList = state->instances[inst].outputs;
      // combine inputs and outputs in one argument table
      sensitivityList.insert(sensitivityList.end(), outputList.begin(),
                             outputList.end());
      auto argTable = sensitivityList.data();

      // run the unit
      auto invocationResult = engine->invoke(name, state, sigTable, argTable);
      if (invocationResult) {
        llvm::errs() << "Failed invocation of " << root << ": "
                     << invocationResult;
        return -1;
      }
    }

    // dump signals initial values
    if (state->time.isZero()) {
      for (int i = 0; i < state->signals.size(); i++) {
        state->dumpSignal(out, i);
      }
    }

    // clear wakeup queue
    wakeupQueue.clear();
    i++;
  }
  llvm::errs() << "Finished after " << i << " steps.\n";
  return 0;
}

void Engine::buildLayout(ModuleOp module) {
  // start from root
  auto rootEntity = module.lookupSymbol<EntityOp>(root);
  assert(rootEntity && "root entity not found!");

  // build root instance, parent and name are the same for the root.
  Instance rootInst(root, root);
  rootInst.unit = root;

  // recursively walk entities starting at root.
  walkEntity(rootEntity, rootInst);

  // store root instance
  state->instances[rootInst.name] = rootInst;

  // add triggers and outputs to signals
  for (auto &inst : state->instances) {
    for (auto trigger : inst.getValue().sensitivityList) {
      state->signals[trigger].triggers.push_back(inst.getKey().str());
    }
    for (auto out : inst.getValue().outputs) {
      state->signals[out].outOf.push_back(inst.getKey().str());
    }
  }
}

void Engine::walkEntity(EntityOp entity, Instance &child) {
  auto res = entity.walk([&](Operation *op) -> WalkResult {
    assert(op);

    //! add signal to signal table
    if (auto sig = dyn_cast<SigOp>(op)) {
      int index = state->addSignal(sig.name().str(), child.name);
      child.signalTable.push_back(index);
    }

    //! build (recursive) instance layout
    if (auto inst = dyn_cast<InstOp>(op)) {
      // skip self-recursion
      if (inst.callee() == child.name)
        return WalkResult::advance();
      if (auto e = op->getParentOfType<ModuleOp>().lookupSymbol<EntityOp>(
              inst.callee())) {
        Instance newChild(inst.name().str(), child.name);
        newChild.unit = inst.callee().str();

        // gather sensitivity list
        for (auto arg : inst.inputs()) {
          // check if the argument comes from a parent's argument
          if (auto a = arg.dyn_cast<BlockArgument>()) {
            int argInd = a.getArgNumber();
            // the argument comes either from one of the inputs or one of the
            // outputs
            if (argInd < newChild.sensitivityList.size())
              newChild.sensitivityList.push_back(child.sensitivityList[argInd]);
            else
              newChild.sensitivityList.push_back(
                  child.outputs[argInd - newChild.sensitivityList.size()]);
          } else if (auto sig = dyn_cast<SigOp>(arg.getDefiningOp())) {
            // otherwise has to come from a sigop,
            // search through the intantce's signal table
            for (auto s : child.signalTable) {
              if (state->signals[s].name == sig.name() &&
                  state->signals[s].owner == child.name) {
                // found, exit loop
                newChild.sensitivityList.push_back(s);
                break;
              }
            }
          }
        }

        // gather outputs list
        for (auto out : inst.outputs()) {
          // check if comes from arg
          if (auto a = out.dyn_cast<BlockArgument>()) {
            int argInd = a.getArgNumber();
            // the argument comes either from one of the inputs or one of the
            // outputs
            if (argInd < newChild.sensitivityList.size())
              newChild.outputs.push_back(child.sensitivityList[argInd]);
            else
              newChild.outputs.push_back(
                  child.outputs[argInd - newChild.sensitivityList.size()]);
          } else if (auto sig = dyn_cast<SigOp>(out.getDefiningOp())) {
            // search through the signal table
            for (auto s : child.signalTable) {
              if (state->signals[s].name == sig.name() &&
                  state->signals[s].owner == child.name) {
                // found, exit loop
                newChild.outputs.push_back(s);
                break;
              }
            }
          }
        }

        // recursively walk new entity
        walkEntity(e, newChild);

        // store the created instance
        state->instances[newChild.name] = newChild;
      }
    }
    return WalkResult::advance();
  });
}
