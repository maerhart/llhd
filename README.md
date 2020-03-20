# LLHD

Development repository for the LLHD Dialect. The repository depends on a build of llvm including mlir. Before building mlir register your custom dialects in include/mlir/IR/DialectSymbolRegistry.def and change the main cmake file to install the td and def files. Once the llvm and mlir are built setup configure the project using the following commands.

```
mkdir build && cd build
cmake -G Ninja .. -DCMAKE_LINKER=<path_to_lld> -DLLVM_DIR=<install_root>/lib/cmake/llvm/ -DLLVM_EXTERNAL_LIT=<build_root>/bin/llvm-lit
cmake --build . --target llhdc-opt
cmake --build . --target check-llhdc
```

In case an error occurs stating that `llvm_expand_pseudo_components` (or some other llvm related cmake command) is not found, make sure that cmake uses the `LLVMConfig.cmake` file built and installed previously (and not the one of another installation, e.g. in /usr/...)

# mlir main repo patches

In DialectSymbolRegistry.def:

```
 DEFINE_SYM_KIND_RANGE(SPIRV) // SPIR-V dialect
 DEFINE_SYM_KIND_RANGE(XLA_HLO) // XLA HLO dialect
 
+DEFINE_SYM_KIND_RANGE(LLHD)
```

In the main CMakeLists.txt:
``` 
     FILES_MATCHING
     PATTERN "*.h"
     PATTERN "*.inc"
+    PATTERN "*.td"
+    PATTERN "*.def"
     PATTERN "LICENSE.TXT"
     )
```

# llvm build instructions

Cmake configuration for llvm

```
cmake -G Ninja ../llvm -DLLVM_BUILD_EXAMPLES=OFF -DLLVM_TARGETS_TO_BUILD="host" -DCMAKE_INSTALL_PREFIX=<install_root> -DLLVM_ENABLE_PROJECTS='mlir' -DLLVM_OPTIMIZED_TABLEGEN=ON -DLLVM_ENABLE_OCAMLDOC=OFF -DLLVM_ENABLE_BINDINGS=OFF -DLLVM_INSTALL_UTILS=ON -DCMAKE_LINKER=<path_to_lld> -DLLVM_PARALLEL_LINK_JOBS=2
```

Build llvm with

```
cmake --build . --target install
```

Do not forget to apply possible patches to llvm before compiling (patches located in stencil-dialect/patches).
