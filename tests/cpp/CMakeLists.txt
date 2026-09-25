# Copyright 2026 The Tessiq Authors.
# SPDX-License-Identifier: Apache-2.0

include(FetchContent)

# Pinned to an exact tag, so an upstream change cannot break CI.
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.15.2
  GIT_SHALLOW TRUE)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

add_executable(tessiq_tests
  test_allocator.cpp
  test_codegen.cpp
  test_conv.cpp
  test_device.cpp
  test_error.cpp
  test_ir.cpp
  test_logging.cpp
  test_liveness.cpp
  test_memory_plan.cpp
  test_op_registry.cpp
  test_op_support.cpp
  test_nvrtc.cpp
  test_op_traits.cpp
  test_pack.cpp
  test_passes.cpp
  test_precompiled.cpp
  test_program.cpp
  test_recipe.cpp
  test_replace_op.cpp
  test_rewrite.cpp
  test_reduction.cpp
  test_result.cpp
  test_shape_rules.cpp)

# WHOLE_ARCHIVE for the same reason tessiq-opt needs it: passes register themselves at static
# initialisation, and a static library only contributes objects something already references.
target_link_libraries(tessiq_tests PRIVATE
  "$<LINK_LIBRARY:WHOLE_ARCHIVE,tessiq_ir>"
  tessiq::codegen
  tessiq::runtime
  tessiq::support
  GTest::gtest_main)

target_compile_definitions(tessiq_tests PRIVATE
  TESSIQ_GOLDEN_DIR="${CMAKE_CURRENT_SOURCE_DIR}/golden")

# Host code of the tests' own, compiled the way csrc/custom_ops/*/*.nvcc.cu is.
if(TESSIQ_NVCC)
  target_sources(tessiq_tests PRIVATE test_scale.nvcc.cu)
  target_include_directories(tessiq_tests PRIVATE "${PROJECT_SOURCE_DIR}/csrc")
  target_link_libraries(tessiq_tests PRIVATE CUDA::cudart)
  target_compile_definitions(tessiq_tests PRIVATE TESSIQ_TEST_PRECOMPILED)
endif()

tessiq_set_warnings(tessiq_tests)
tessiq_enable_sanitizers(tessiq_tests)

include(GoogleTest)
# PRE_TEST rather than the default POST_BUILD, because golden cases are found by walking a
# directory when the binary starts rather than by anything the compiler sees. Discovering at build
# time meant the list ctest holds only changed when something relinked, so adding a golden file
# left it out of the run and deleting one left an entry behind that passed: a filter matching no
# test is not an error. PRE_TEST asks the binary for its tests each time ctest runs, so both cases
# are right.
gtest_discover_tests(tessiq_tests DISCOVERY_MODE PRE_TEST)
