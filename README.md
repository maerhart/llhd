# LLHD

![](https://github.com/maerhart/llhd/workflows/Build%20and%20Test/badge.svg?event=push)

- [LLHD Documentation](https://rodonisi.github.io/llhd-docs/)

Development repository for the LLHD Dialect. The repository depends on a build of llvm including mlir. Once the llvm and mlir are built setup configure the project using the following commands.

```
mkdir build && cd build
cmake -G Ninja .. -DCMAKE_LINKER=<path_to_lld> -DLLVM_DIR=<install_root>/lib/cmake/llvm/ -DLLVM_EXTERNAL_LIT=<build_root>/bin/llvm-lit
cmake --build . --target llhdc
cmake --build . --target check-llhdc
```

In case an error occurs stating that `llvm_expand_pseudo_components` (or some other llvm related cmake command) is not found, make sure that cmake uses the `LLVMConfig.cmake` file built and installed previously (not the one of another installation, e.g. `/usr/...`)

# llvm build instructions

Cmake configuration for llvm

```
cmake -G Ninja ../llvm -DLLVM_BUILD_EXAMPLES=OFF -DLLVM_TARGETS_TO_BUILD="host" -DCMAKE_INSTALL_PREFIX=<install_root> -DLLVM_ENABLE_PROJECTS='mlir' -DLLVM_OPTIMIZED_TABLEGEN=ON -DLLVM_ENABLE_OCAMLDOC=OFF -DLLVM_ENABLE_BINDINGS=OFF -DLLVM_INSTALL_UTILS=ON -DCMAKE_LINKER=<path_to_lld> -DLLVM_PARALLEL_LINK_JOBS=2
```

Build llvm with

```
cmake --build . --target install
```
