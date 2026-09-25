# Copyright 2026 The Tessiq Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Warning configuration.
#
# Warnings fail the build in CI.

# Applies the project's warning set to `target`.
#
# PRIVATE, so the policy does not leak into targets that link against tessiq.
function(tessiq_set_warnings target)
  set(gcc_clang_warnings
      -Wall
      -Wextra
      -Wpedantic
      -Wshadow                  # a shadowed variable is almost always a bug in the making
      -Wnon-virtual-dtor        # deleting through a base pointer without a virtual destructor
      -Wcast-align
      -Wunused
      -Woverloaded-virtual      # an overload that silently fails to override
      -Wconversion              # implicit narrowing; noisy at first, and worth it
      -Wsign-conversion
      -Wnull-dereference
      -Wdouble-promotion        # accidental float -> double, which costs real time on device code
      -Wformat=2
      -Wimplicit-fallthrough)

  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    list(APPEND gcc_clang_warnings
         -Wmisleading-indentation
         -Wduplicated-cond
         -Wduplicated-branches
         -Wlogical-op
         -Wuseless-cast)
  endif()

  target_compile_options(${target} PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:${gcc_clang_warnings}>)

  if(TESSIQ_WARNINGS_AS_ERRORS)
    target_compile_options(${target} PRIVATE
      $<$<COMPILE_LANGUAGE:CXX>:-Werror>)
  endif()
endfunction()
