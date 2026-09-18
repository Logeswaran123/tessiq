# Copyright 2026 The Tessera Authors.
# SPDX-License-Identifier: Apache-2.0

include(FetchContent)

# Pinned to an exact tag: a test framework that moves under you turns an unrelated CI failure into
# an afternoon of debugging.
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.15.2
  GIT_SHALLOW TRUE)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

add_executable(tessera_tests
  test_allocator.cpp
  test_device.cpp
  test_error.cpp
  test_logging.cpp
  test_result.cpp)

target_link_libraries(tessera_tests PRIVATE
  tessera::runtime
  tessera::support
  GTest::gtest_main)

tessera_set_warnings(tessera_tests)
tessera_enable_sanitizers(tessera_tests)

include(GoogleTest)
gtest_discover_tests(tessera_tests)
