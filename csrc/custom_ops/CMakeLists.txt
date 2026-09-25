# Copyright 2026 The Tessiq Authors.
# SPDX-License-Identifier: Apache-2.0

# Every directory here is one custom operation: a .cpp that defines it and, for one with a kernel of
# its own, the .cu file that holds the kernel. Adding a directory is all it takes to add one; the
# globs below are re-checked at every build, so a new directory needs no change here.
#
# The .cpp files are compiled into the extension. The .cu files are not compiled here. Their text
# is carried in the extension and compiled with NVRTC for the device a kernel runs on, the first
# time it runs there, the same way the kernels the compiler writes are. That keeps one extension
# working on every GPU generation, and a machine without a GPU still builds and runs the reference.

set(embedded_kernels "")
file(GLOB entries LIST_DIRECTORIES true CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*")
foreach(entry IN LISTS entries)
  if(NOT IS_DIRECTORY "${entry}")
    continue()
  endif()
  get_filename_component(op "${entry}" NAME)

  file(GLOB definitions CONFIGURE_DEPENDS "${entry}/*.cpp")
  if(NOT definitions)
    message(WARNING "csrc/custom_ops/${op} has no .cpp defining an operation, so it is left out")
    continue()
  endif()
  target_sources(_C PRIVATE ${definitions})

  file(GLOB kernels CONFIGURE_DEPENDS "${entry}/*.cu")
  foreach(kernel IN LISTS kernels)
    get_filename_component(file "${kernel}" NAME)
    set(embedded "${CMAKE_CURRENT_BINARY_DIR}/${op}/${file}.cpp")
    add_custom_command(
      OUTPUT "${embedded}"
      COMMAND "${CMAKE_COMMAND}" "-DKERNEL=${kernel}" "-DKEY=${op}/${file}" "-DOUTPUT=${embedded}"
              -P "${CMAKE_CURRENT_SOURCE_DIR}/embed.cmake"
      DEPENDS "${kernel}" "${CMAKE_CURRENT_SOURCE_DIR}/embed.cmake"
      COMMENT "Embedding csrc/custom_ops/${op}/${file}"
      VERBATIM)
    # A kernel longer than the few kilobytes ISO C++ promises a string literal can hold is fine for
    # every compiler this builds with, so -Wpedantic's warning about it is not one to act on.
    set_source_files_properties("${embedded}" PROPERTIES COMPILE_OPTIONS "-Wno-overlength-strings")
    target_sources(_C PRIVATE "${embedded}")
    list(APPEND embedded_kernels "${embedded}")
  endforeach()
endforeach()

# A custom command only runs for targets in the directory that declares it, and the extension is
# declared in bindings/. This target runs the embedding here, and the extension waits for it.
add_custom_target(tessiq_custom_op_kernels DEPENDS ${embedded_kernels})
add_dependencies(_C tessiq_custom_op_kernels)
