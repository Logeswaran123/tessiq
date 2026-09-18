# Copyright 2026 The Tessera Authors.
# SPDX-License-Identifier: Apache-2.0

# --- tessera_support ------------------------------------------------------------------------
#
# The foundation layer: assertions, errors, logging. Every other target depends on it, and it
# depends on nothing but the standard library. Keeping it free of torch and CUDA is what allows
# the IR and the passes to be tested without either (PLAN.md section 8).

# version.h is generated so that the version lives in exactly one place (CMakeLists.txt) and
# cannot drift between the native module and the Python package.
configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/tessera/version.h.in"
  "${CMAKE_CURRENT_BINARY_DIR}/generated/tessera/version.h"
  @ONLY)

add_library(tessera_support STATIC
  tessera/support/support.cpp)

target_include_directories(tessera_support PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/generated>)

target_compile_features(tessera_support PUBLIC cxx_std_17)

tessera_set_warnings(tessera_support)
tessera_enable_sanitizers(tessera_support)

add_library(tessera::support ALIAS tessera_support)

# --- tessera_runtime ------------------------------------------------------------------------
#
# Device memory and transfers. Depends on support and the standard library, nothing else: the
# CPU backend has to build and run on a machine with no GPU and no PyTorch.

add_library(tessera_runtime STATIC
  tessera/runtime/allocator.cpp
  tessera/runtime/cpu_device.cpp)

target_link_libraries(tessera_runtime PUBLIC tessera::support)

tessera_set_warnings(tessera_runtime)
tessera_enable_sanitizers(tessera_runtime)

add_library(tessera::runtime ALIAS tessera_runtime)

# --- Python extension module ----------------------------------------------------------------

if(TESSERA_BUILD_PYTHON)
  add_subdirectory(bindings)
endif()
