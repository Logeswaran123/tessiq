# Copyright 2026 The Tessiq Authors.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.26)

project(
  tessiq
  VERSION 0.0.1
  DESCRIPTION "A from-scratch PyTorch accelerator backend: device, graph compiler, and kernels"
  LANGUAGES CXX)

# --- Options ------------------------------------------------------------------------------------

option(TESSIQ_BUILD_TESTS "Build the C++ test suite" ON)
option(TESSIQ_BUILD_PYTHON "Build the Python extension module" ON)
option(TESSIQ_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" ON)
option(TESSIQ_ENABLE_CUDA "Build the CUDA backend" OFF)

# Reserved for the MLIR export target. Declared now so the option name is fixed and documented
# before anything depends on it; enabling it is a later phase.
option(TESSIQ_ENABLE_MLIR "Build the MLIR export target (PHASES.md, phase 10)" OFF)

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
          "TESSIQ_ENABLE_MLIR is reserved for phase 10 (see PHASES.md) and is not "
          "implemented yet. Configure without it.")
endif()

# --- Subdirectories -----------------------------------------------------------------------------

add_subdirectory(csrc)

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
message(STATUS "  sanitizers ......... ${TESSIQ_SANITIZER}")
message(STATUS "")
