cmake_minimum_required(VERSION 3.12.4)

project(llhdc LANGUAGES CXX)

find_package(LLVM REQUIRED CONFIG)

if(LLVM_FOUND)
  message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
  message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
endif()

if(LLVM_ENABLE_ZLIB)
  find_package(ZLIB)
endif()

# MLIR project.
set(MLIR_MAIN_SRC_DIR ${LLVM_INCLUDE_DIR}) # --src-root
set(MLIR_INCLUDE_DIR ${LLVM_INCLUDE_DIR}) # --includedir
set(MLIR_TABLEGEN_EXE mlir-tblgen)

# setup llvm lit
set(LLVM_LIT_ARGS "-sv" CACHE STRING "lit default options")

# set the output directories
set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin )
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib )

# import llvm functionality
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)

include_directories(${LLVM_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS})

function(mlir_tablegen ofn)
  tablegen(MLIR ${ARGV} "-I${MLIR_MAIN_SRC_DIR}" "-I${MLIR_INCLUDE_DIR}" "-I${PROJECT_SOURCE_DIR}/include")
  set(TABLEGEN_OUTPUT ${TABLEGEN_OUTPUT} ${CMAKE_CURRENT_BINARY_DIR}/${ofn}
      PARENT_SCOPE)
endfunction()

function(add_mlir_dialect dialect dialect_doc_filename)
  set(LLVM_TARGET_DEFINITIONS ${dialect}.td)
  mlir_tablegen(${dialect}.h.inc -gen-op-decls)
  mlir_tablegen(${dialect}.cpp.inc -gen-op-defs)
  add_public_tablegen_target(MLIR${dialect}IncGen)
endfunction()

function(add_mlir_doc doc_filename command output_file output_directory)
  set(LLVM_TARGET_DEFINITIONS ${doc_filename}.td)
  tablegen(MLIR ${output_file}.md ${command} "-I${MLIR_MAIN_SRC_DIR}" "-I${MLIR_INCLUDE_DIR}" "-I${PROJECT_SOURCE_DIR}/include")
  set(GEN_DOC_FILE ${PROJECT_SOURCE_DIR}/docs/${output_directory}${output_file}.md)
  add_custom_command(
          OUTPUT ${GEN_DOC_FILE}
          COMMAND ${CMAKE_COMMAND} -E copy
                  ${CMAKE_CURRENT_BINARY_DIR}/${output_file}.md
                  ${GEN_DOC_FILE}
          DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${output_file}.md)
  add_custom_target(${output_file}DocGen DEPENDS ${GEN_DOC_FILE})
  add_dependencies(llhd-doc ${output_file}DocGen)
endfunction()

#function(whole_archive_link target)
#  # currently not working for our passes (instead we added them to target link libraries)
#  #add_dependencies(${target} ${ARGN})
#  set(link_flags "-L${LLVM_LIBRARY_DIR} -L${CMAKE_BINARY_DIR}/lib -Wl,--whole-archive,")
#  FOREACH(LIB ${ARGN})
#    string(CONCAT link_flags ${link_flags} "-l${LIB},")
#  ENDFOREACH(LIB)
#  string(CONCAT link_flags ${link_flags} "--no-whole-archive")
#  set_target_properties(${target} PROPERTIES LINK_FLAGS ${link_flags})
#endfunction(whole_archive_link)

# add target for documentation generation
add_custom_target(llhd-doc)

add_subdirectory(include/)
add_subdirectory(lib/)
add_subdirectory(test/)
add_subdirectory(llhdc/)
