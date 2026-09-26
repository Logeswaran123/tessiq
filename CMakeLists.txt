# Copyright 2026 The Tessiq Authors.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.26)

project(
  tessiq
  VERSION 0.0.1
  DESCRIPTION "A PyTorch backend for NVIDIA GPUs: device, graph compiler, and kernels"
  LANGUAGES CXX)

# --- Options ------------------------------------------------------------------------------------

option(TESSIQ_BUILD_TESTS "Build the C++ test suite" ON)
option(TESSIQ_BUILD_PYTHON "Build the Python extension module" ON)
option(TESSIQ_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" ON)
# On when a CUDA toolkit is found, so the same command works with and without a GPU. Force it
# with -DTESSIQ_ENABLE_CUDA=ON/OFF.
find_package(CUDAToolkit QUIET)
option(TESSIQ_ENABLE_CUDA "Build the CUDA backend" ${CUDAToolkit_FOUND})

# nvcc compiles the precompiled custom operations, csrc/custom_ops/*/*.nvcc.cu. Without it those
# files are left out and the operations run their reference.
set(TESSIQ_NVCC OFF)
if(TESSIQ_ENABLE_CUDA)
  if(NOT CMAKE_CUDA_COMPILER AND CUDAToolkit_NVCC_EXECUTABLE)
    set(CMAKE_CUDA_COMPILER "${CUDAToolkit_NVCC_EXECUTABLE}")
  endif()
  # Read before check_language, which fills it in with nvcc's own default.
  set(tessiq_cuda_architectures "${CMAKE_CUDA_ARCHITECTURES}")
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    # The GPUs in this machine unless told otherwise, since a precompiled kernel only runs on an
    # architecture it was built for. A machine with nvcc and no GPU has to say which.
    if(tessiq_cuda_architectures)
      set(CMAKE_CUDA_ARCHITECTURES "${tessiq_cuda_architectures}")
    else()
      set(CMAKE_CUDA_ARCHITECTURES native)
    endif()
    # The same shared runtime the rest of tessiq links, so the process has one.
    set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    enable_language(CUDA)
    set(TESSIQ_NVCC ON)
  endif()
endif()

# Reserved for the MLIR export target; not implemented yet.
option(TESSIQ_ENABLE_MLIR "Build the MLIR export target (PHASES.md, phase 15)" OFF)

set(TESSIQ_SANITIZER "" CACHE STRING
    "Comma-separated sanitizers to enable, e.g. 'address,undefined'. Empty disables them.")

# --- Global settings ----------------------------------------------------------------------------

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)  # -std=c++17, never -std=gnu++17: portability over convenience

# Position-independent code everywhere: the static support library is linked into a shared Python
# extension module, which requires it.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Export compile_commands.json for clang-tidy, clangd, and editor integration.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Build type" FORCE)
  message(STATUS "tessiq: CMAKE_BUILD_TYPE not set, defaulting to RelWithDebInfo")
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(CompilerWarnings)
include(Sanitizers)

if(TESSIQ_ENABLE_MLIR)
  message(FATAL_ERROR
          "TESSIQ_ENABLE_MLIR is reserved for phase 15 (see PHASES.md) and is not "
          "implemented yet. Configure without it.")
endif()

# --- Subdirectories -----------------------------------------------------------------------------

# Which interpreter to ask, decided once for every find_package(Python) below. More than one place
# asks: the extension module needs its headers, and codegen asks it where the nvidia-cudnn package
# is installed. Two answers would mean building against one environment and linking another's cuDNN.
#
# FindPython defaults to the highest version it can locate, not the active environment. In a conda
# layout that means the base install rather than the env on PATH. LOCATION takes the first on PATH,
# and an explicit -DPython_EXECUTABLE still wins over both.
# STANDARD, not the default FIRST: FIRST trusts CONDA_PREFIX, which points at the base environment
# whenever a conda env is on PATH without having been activated in this shell. With STANDARD the
# search follows PATH, so an activated venv still wins by being first.
set(Python_FIND_VIRTUALENV STANDARD)
set(Python_FIND_STRATEGY LOCATION)

add_subdirectory(csrc)

add_subdirectory(tools)

if(TESSIQ_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests/cpp)
endif()

# --- Configuration summary ----------------------------------------------------------------------

message(STATUS "")
message(STATUS "tessiq ${PROJECT_VERSION} configuration:")
message(STATUS "  build type ......... ${CMAKE_BUILD_TYPE}")
message(STATUS "  compiler ........... ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "  tests .............. ${TESSIQ_BUILD_TESTS}")
message(STATUS "  python module ...... ${TESSIQ_BUILD_PYTHON}")
message(STATUS "  warnings as errors . ${TESSIQ_WARNINGS_AS_ERRORS}")
message(STATUS "  cuda backend ....... ${TESSIQ_ENABLE_CUDA}")
if(TESSIQ_NVCC)
  message(STATUS "  precompiled ops .... ${CMAKE_CUDA_ARCHITECTURES}")
else()
  message(STATUS "  precompiled ops .... OFF (no nvcc)")
endif()
message(STATUS "  sanitizers ......... ${TESSIQ_SANITIZER}")
message(STATUS "")
