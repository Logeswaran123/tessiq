# Copyright 2026 The Tessera Authors.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.26)

project(
  tessera
  VERSION 0.0.1
  DESCRIPTION "A from-scratch PyTorch accelerator backend: device, graph compiler, and kernels"
  LANGUAGES CXX)

# --- Options ------------------------------------------------------------------------------------

option(TESSERA_BUILD_TESTS "Build the C++ test suite" ON)
option(TESSERA_BUILD_PYTHON "Build the Python extension module" ON)
option(TESSERA_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" ON)
option(TESSERA_ENABLE_CUDA "Build the CUDA backend" OFF)

# Reserved for PLAN.md section 9.3. Declared now so the option name is fixed and documented
# before anything depends on it; enabling it is a later phase.
option(TESSERA_ENABLE_MLIR "Build the MLIR export target (PLAN.md phase 9)" OFF)

set(TESSERA_SANITIZER "" CACHE STRING
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
  message(STATUS "tessera: CMAKE_BUILD_TYPE not set, defaulting to RelWithDebInfo")
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(CompilerWarnings)
include(Sanitizers)

if(TESSERA_ENABLE_MLIR)
  message(FATAL_ERROR
          "TESSERA_ENABLE_MLIR is reserved for phase 9 (see PLAN.md section 9.3) and is not "
          "implemented yet. Configure without it.")
endif()

# --- Subdirectories -----------------------------------------------------------------------------

add_subdirectory(csrc)

if(TESSERA_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests/cpp)
endif()

# --- Configuration summary ----------------------------------------------------------------------

message(STATUS "")
message(STATUS "tessera ${PROJECT_VERSION} configuration:")
message(STATUS "  build type ......... ${CMAKE_BUILD_TYPE}")
message(STATUS "  compiler ........... ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "  tests .............. ${TESSERA_BUILD_TESTS}")
message(STATUS "  python module ...... ${TESSERA_BUILD_PYTHON}")
message(STATUS "  warnings as errors . ${TESSERA_WARNINGS_AS_ERRORS}")
message(STATUS "  cuda backend ....... ${TESSERA_ENABLE_CUDA}")
message(STATUS "  sanitizers ......... ${TESSERA_SANITIZER}")
message(STATUS "")
