# Copyright 2026 The Tessiq Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Sanitizer configuration (PLAN.md section 7.7).
#
# ASan and UBSan run on the CPU backend in CI. They are not enabled by default because they change
# the ABI and slow builds; they are a distinct build configuration, selected with
# -DTESSIQ_SANITIZER=address,undefined.

# Applies the sanitizers named in TESSIQ_SANITIZER to `target`.
#
# Both compile and link options are required: a sanitizer applied only at compile time links
# against missing runtime symbols, which produces a confusing failure far from its cause.
function(tessiq_enable_sanitizers target)
  if(NOT TESSIQ_SANITIZER)
    return()
  endif()

  if(NOT CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    message(WARNING "TESSIQ_SANITIZER is set but ${CMAKE_CXX_COMPILER_ID} is unsupported; ignoring")
    return()
  endif()

  string(REPLACE "," ";" sanitizer_list "${TESSIQ_SANITIZER}")
  string(REPLACE ";" "," sanitizer_flag "${sanitizer_list}")

  message(STATUS "tessiq: sanitizers enabled for ${target}: ${sanitizer_flag}")

  target_compile_options(${target} PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-fsanitize=${sanitizer_flag}>
    # Without frame pointers the stack traces sanitizers produce are close to useless.
    $<$<COMPILE_LANGUAGE:CXX>:-fno-omit-frame-pointer>)
  target_link_options(${target} PRIVATE -fsanitize=${sanitizer_flag})
endfunction()
