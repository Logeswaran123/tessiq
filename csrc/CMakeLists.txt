# Copyright 2026 The Tessiq Authors.
# SPDX-License-Identifier: Apache-2.0

# --- tessiq_support ------------------------------------------------------------------------
#
# Assertions, errors, logging. Every other target depends on it, and it depends only on the
# standard library, so the IR and the passes can be tested without torch or CUDA.

# version.h is generated from the version in the top-level CMakeLists.txt.
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
# Device memory and transfers. Depends only on support, so the CPU backend builds and runs on a
# machine with no GPU and no PyTorch.

add_library(tessiq_runtime STATIC
  tessiq/runtime/allocator.cpp
  tessiq/runtime/cpu_device.cpp
  tessiq/runtime/cuda_device.cpp
  tessiq/runtime/registry.cpp)

# The CUDA backend uses the runtime API only, so it is ordinary C++ linking cudart. Kernels are
# compiled at run time with NVRTC (see tessiq_codegen), so the CUDA language is never enabled.
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
# The IR, its textual form, and the passes. Depends only on support, so `tessiq-opt` is a plain
# binary and pass tests need no torch, CUDA or GPU.

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
  tessiq/passes/dead_node_elimination.cpp
  tessiq/passes/fuse.cpp
  tessiq/passes/layout_assign.cpp
  tessiq/passes/mark_supported.cpp
  tessiq/passes/op_registry.cpp
  tessiq/passes/op_support.cpp
  tessiq/passes/op_table.cpp
  tessiq/passes/op_traits.cpp
  tessiq/passes/pass.cpp
  tessiq/passes/replace_op.cpp
  tessiq/passes/rewrite.cpp
  tessiq/passes/shape_infer.cpp
  tessiq/passes/shape_rules.cpp)

target_link_libraries(tessiq_ir PUBLIC tessiq::support)

tessiq_set_warnings(tessiq_ir)
tessiq_enable_sanitizers(tessiq_ir)

add_library(tessiq::ir ALIAS tessiq_ir)

# --- tessiq_codegen ---------------------------------------------------------------------------
#
# Kernel descriptions, source emission, and compiling and launching through NVRTC, cuBLAS and
# cuDNN. Separate from tessiq_ir so the IR and passes build without the toolkit. Emitting source
# needs no toolkit, so that part is testable anywhere.

add_library(tessiq_codegen STATIC
  tessiq/codegen/blas.cpp
  tessiq/codegen/call.cpp
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

# Without the toolkit the codegen reports that this build has none, and everything else carries
# on.
if(TESSIQ_ENABLE_CUDA)
  find_package(CUDAToolkit REQUIRED)
  # cudart sets the device and creates the primary context; the driver API loads modules and
  # launches. Both, because the runtime alone cannot load a cubin.
  target_link_libraries(tessiq_codegen PRIVATE CUDA::nvrtc CUDA::cudart CUDA::cuda_driver CUDA::cublas)
  target_compile_definitions(tessiq_codegen PRIVATE TESSIQ_WITH_CUDA)

  # cuDNN is not part of the CUDA toolkit; it comes in the nvidia-cudnn pip package torch depends
  # on, so the interpreter is asked where that is. Optional: without it convolution stays on the
  # host.
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
  # After bindings, because it adds the operations it finds to the extension that defines.
  add_subdirectory(custom_ops)
endif()
