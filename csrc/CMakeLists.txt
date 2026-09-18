# Copyright 2026 The Tessiq Authors.
# SPDX-License-Identifier: Apache-2.0

# --- tessiq_support ------------------------------------------------------------------------
#
# The foundation layer: assertions, errors, logging. Every other target depends on it, and it
# depends on nothing but the standard library. Keeping it free of torch and CUDA is what allows
# the IR and the passes to be tested without either (PLAN.md section 8).

# version.h is generated so that the version lives in exactly one place (CMakeLists.txt) and
# cannot drift between the native module and the Python package.
configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/tessiq/version.h.in"
  "${CMAKE_CURRENT_BINARY_DIR}/generated/tessiq/version.h"
  @ONLY)

add_library(tessiq_support STATIC
  tessiq/support/support.cpp)

target_include_directories(tessiq_support PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/generated>)

target_compile_features(tessiq_support PUBLIC cxx_std_17)

tessiq_set_warnings(tessiq_support)
tessiq_enable_sanitizers(tessiq_support)

add_library(tessiq::support ALIAS tessiq_support)

# --- tessiq_runtime ------------------------------------------------------------------------
#
# Device memory and transfers. Depends on support and the standard library, nothing else: the
# CPU backend has to build and run on a machine with no GPU and no PyTorch.

add_library(tessiq_runtime STATIC
  tessiq/runtime/allocator.cpp
  tessiq/runtime/cpu_device.cpp)

target_link_libraries(tessiq_runtime PUBLIC tessiq::support)

tessiq_set_warnings(tessiq_runtime)
tessiq_enable_sanitizers(tessiq_runtime)

add_library(tessiq::runtime ALIAS tessiq_runtime)

# --- Python extension module ----------------------------------------------------------------

if(TESSIQ_BUILD_PYTHON)
  add_subdirectory(bindings)
endif()
