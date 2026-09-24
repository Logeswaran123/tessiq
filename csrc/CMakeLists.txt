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
  tessiq/ir/liveness.cpp
  tessiq/ir/memory_plan.cpp
  tessiq/ir/recipe.cpp
  tessiq/ir/text.cpp
  tessiq/ir/types.cpp
  tessiq/passes/algebraic.cpp
  tessiq/passes/cluster.cpp
  tessiq/passes/const_fold.cpp
  tessiq/passes/cse.cpp
  tessiq/passes/dce.cpp
  tessiq/passes/fuse.cpp
  tessiq/passes/layout_assign.cpp
  tessiq/passes/mark_supported.cpp
  tessiq/passes/op_registry.cpp
  tessiq/passes/op_support.cpp
  tessiq/passes/op_table.cpp
  tessiq/passes/op_traits.cpp
  tessiq/passes/pass.cpp
  tessiq/passes/shape_infer.cpp
  tessiq/passes/shape_rules.cpp)

target_link_libraries(tessiq_ir PUBLIC tessiq::support)

tessiq_set_warnings(tessiq_ir)
tessiq_enable_sanitizers(tessiq_ir)

add_library(tessiq::ir ALIAS tessiq_ir)

# --- tessiq_codegen ---------------------------------------------------------------------------
#
# Turning a fused group into source text. Its own target rather than part of tessiq_ir, because
# compiling that source needs the CUDA toolkit and the IR and the passes have to keep building
# without one. Emitting the text does not, which is why this half is testable anywhere.

add_library(tessiq_codegen STATIC
  tessiq/codegen/blas.cpp
  tessiq/codegen/conv.cpp
  tessiq/codegen/cubin_cache.cpp
  tessiq/codegen/cudnn.cpp
  tessiq/codegen/cuda_emitter.cpp
  tessiq/codegen/gemm.cpp
  tessiq/codegen/kernel.cpp
  tessiq/codegen/nvrtc.cpp
  tessiq/codegen/pack.cpp
  tessiq/codegen/program.cpp
  tessiq/codegen/reduction.cpp
  tessiq/codegen/timing.cpp)

target_link_libraries(tessiq_codegen PUBLIC tessiq::ir)

# NVRTC compiles generated source at run time. Guarded the same way the CUDA backend is: without
# the toolkit the compiler reports that this build has none, and everything above it carries on.
if(TESSIQ_ENABLE_CUDA)
  find_package(CUDAToolkit REQUIRED)
  # cudart sets the device and creates the primary context; the driver API loads modules and
  # launches. Both, because the runtime alone cannot load a cubin.
  target_link_libraries(tessiq_codegen PRIVATE CUDA::nvrtc CUDA::cudart CUDA::cuda_driver CUDA::cublas)
  target_compile_definitions(tessiq_codegen PRIVATE TESSIQ_WITH_CUDA)

  # cuDNN does not ship with the CUDA toolkit. It comes inside the nvidia-cudnn pip package that
  # torch depends on, so the place to look is wherever that package is installed. Asked of the
  # interpreter rather than guessed, the way the torch and pybind11 CMake configs are found.
  # Optional: without it convolution stays on the host, which is correct and slower.
  #
  # -I because pip builds in isolation: it installs a sitecustomize that removes the environment's
  # own site-packages from the path, so a plain `import nvidia.cudnn` fails under `pip install`
  # and succeeds everywhere else. Isolated mode skips sitecustomize and sees the real packages.
  find_package(Python COMPONENTS Interpreter QUIET)
  if(Python_Interpreter_FOUND)
    execute_process(
      COMMAND "${Python_EXECUTABLE}" -I -c "import nvidia.cudnn as m; print(m.__path__[0])"
      OUTPUT_VARIABLE tessiq_cudnn_hint
      OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_QUIET)
  endif()
  find_path(TESSIQ_CUDNN_INCLUDE cudnn.h HINTS "${tessiq_cudnn_hint}/include")
  # The pip package ships only the versioned name, with no plain libcudnn.so to link against.
  find_library(TESSIQ_CUDNN_LIBRARY NAMES cudnn libcudnn.so.9 HINTS "${tessiq_cudnn_hint}/lib")
  if(TESSIQ_CUDNN_INCLUDE AND TESSIQ_CUDNN_LIBRARY)
    message(STATUS "tessiq: cuDNN at ${TESSIQ_CUDNN_LIBRARY}")
    target_include_directories(tessiq_codegen SYSTEM PRIVATE "${TESSIQ_CUDNN_INCLUDE}")
    target_link_libraries(tessiq_codegen PRIVATE "${TESSIQ_CUDNN_LIBRARY}")
    target_compile_definitions(tessiq_codegen PRIVATE TESSIQ_WITH_CUDNN)
  else()
    message(STATUS "tessiq: no cuDNN found, convolution runs on the host")
  endif()
endif()

tessiq_set_warnings(tessiq_codegen)
tessiq_enable_sanitizers(tessiq_codegen)

add_library(tessiq::codegen ALIAS tessiq_codegen)

# --- Python extension module ----------------------------------------------------------------

if(TESSIQ_BUILD_PYTHON)
  add_subdirectory(bindings)
endif()
