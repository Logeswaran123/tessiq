# Copyright 2026 The Tessiq Authors.
# SPDX-License-Identifier: Apache-2.0

# --- tessiq_support ------------------------------------------------------------------------
#
# The foundation layer: assertions, errors, logging. Every other target depends on it, and it
# depends on nothing but the standard library. Keeping it free of torch and CUDA is what allows
# the IR and the passes to be tested without either.

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
  tessiq/runtime/cpu_device.cpp
  tessiq/runtime/cuda_device.cpp
  tessiq/runtime/registry.cpp)

# The CUDA backend talks to the driver through the runtime API only; kernels arrive in a later
# phase and will need the CUDA language enabled. Until then this is ordinary C++ linking cudart,
# which keeps the build working on machines without a GPU.
if(TESSIQ_ENABLE_CUDA)
  find_package(CUDAToolkit REQUIRED)
  target_link_libraries(tessiq_runtime PRIVATE CUDA::cudart)
  target_compile_definitions(tessiq_runtime PRIVATE TESSIQ_WITH_CUDA)
  message(STATUS "tessiq: cuda backend enabled, cudart ${CUDAToolkit_VERSION}")
endif()

target_link_libraries(tessiq_runtime PUBLIC tessiq::support)

tessiq_set_warnings(tessiq_runtime)
tessiq_enable_sanitizers(tessiq_runtime)

add_library(tessiq::runtime ALIAS tessiq_runtime)

# --- tessiq_ir ------------------------------------------------------------------------------
#
# The IR, its textual form, and the pass infrastructure. Depends on support and nothing else: the
# compiler has to be testable without torch, without CUDA and without a GPU, which is what keeps
# `tessiq-opt` a plain binary and pass tests a matter of text in, text out.

add_library(tessiq_ir STATIC
  tessiq/ir/graph.cpp
  tessiq/ir/text.cpp
  tessiq/ir/types.cpp
  tessiq/passes/const_fold.cpp
  tessiq/passes/cse.cpp
  tessiq/passes/dce.cpp
  tessiq/passes/mark_supported.cpp
  tessiq/passes/op_support.cpp
  tessiq/passes/pass.cpp)

target_link_libraries(tessiq_ir PUBLIC tessiq::support)

tessiq_set_warnings(tessiq_ir)
tessiq_enable_sanitizers(tessiq_ir)

add_library(tessiq::ir ALIAS tessiq_ir)

# --- Python extension module ----------------------------------------------------------------

if(TESSIQ_BUILD_PYTHON)
  add_subdirectory(bindings)
endif()
