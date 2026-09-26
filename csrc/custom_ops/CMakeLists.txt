# Copyright 2026 The Tessiq Authors.
# SPDX-License-Identifier: Apache-2.0

# Every directory here is one custom operation: a .cpp that defines it and, for one with a kernel of
# its own, the .cu file that holds the kernel. A .tsr file holds the pattern the operation's rewrite
# looks for. Adding a directory is all it takes to add one; the globs below are re-checked at every
# build, so a new directory needs no change here.
#
# The .cpp files are compiled into the extension. The .cu and .tsr files are not compiled here: their
# text is carried in the extension. A pattern is parsed when tessiq is imported. A kernel is
# compiled with NVRTC for the device it runs on, the first time it runs there, the same way the
# kernels the compiler writes are. That keeps one extension working on every GPU generation, and a
# machine without a GPU still builds and runs the reference.
#
# A .nvcc.cu file is different: nvcc compiles it here, for CMAKE_CUDA_ARCHITECTURES, because it holds
# host code as well as kernels. Without nvcc it is left out and its operation runs the reference.

set(embedded_files "")
set(precompiled_files "")
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

  file(GLOB precompiled CONFIGURE_DEPENDS "${entry}/*.nvcc.cu")
  if(precompiled AND TESSIQ_NVCC)
    list(APPEND precompiled_files ${precompiled})
  elseif(precompiled)
    message(STATUS "tessiq: no nvcc, so tessiq::${op} runs its reference")
  endif()

  file(GLOB carried CONFIGURE_DEPENDS "${entry}/*.cu" "${entry}/*.tsr")
  if(precompiled)
    list(REMOVE_ITEM carried ${precompiled})
  endif()
  foreach(source IN LISTS carried)
    get_filename_component(file "${source}" NAME)
    set(embedded "${CMAKE_CURRENT_BINARY_DIR}/${op}/${file}.cpp")
    add_custom_command(
      OUTPUT "${embedded}"
      COMMAND "${CMAKE_COMMAND}" "-DSOURCE=${source}" "-DKEY=${op}/${file}" "-DOUTPUT=${embedded}"
              -P "${CMAKE_CURRENT_SOURCE_DIR}/embed.cmake"
      DEPENDS "${source}" "${CMAKE_CURRENT_SOURCE_DIR}/embed.cmake"
      COMMENT "Embedding csrc/custom_ops/${op}/${file}"
      VERBATIM)
    # A file longer than the few kilobytes ISO C++ promises a string literal can hold is fine for
    # every compiler this builds with, so -Wpedantic's warning about it is not one to act on.
    set_source_files_properties("${embedded}" PROPERTIES COMPILE_OPTIONS "-Wno-overlength-strings")
    target_sources(_C PRIVATE "${embedded}")
    list(APPEND embedded_files "${embedded}")
  endforeach()
endforeach()

# A custom command only runs for targets in the directory that declares it, and the extension is
# declared in bindings/. This target runs the embedding here, and the extension waits for it.
add_custom_target(tessiq_custom_op_files DEPENDS ${embedded_files})
add_dependencies(_C tessiq_custom_op_files)

if(precompiled_files)
  # Objects rather than an archive, so the registrations in them are linked whether or not anything
  # names them.
  add_library(tessiq_precompiled_ops OBJECT ${precompiled_files})
  target_include_directories(tessiq_precompiled_ops PRIVATE "${PROJECT_SOURCE_DIR}/csrc")
  if(TESSIQ_WARNINGS_AS_ERRORS)
    target_compile_options(tessiq_precompiled_ops PRIVATE
      "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--Werror all-warnings>")
  endif()
  target_link_libraries(_C PRIVATE tessiq_precompiled_ops CUDA::cudart)
endif()
