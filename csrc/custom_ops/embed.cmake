# Copyright 2026 The Tessiq Authors.
# SPDX-License-Identifier: Apache-2.0

# Writes the text of kernel file KERNEL into OUTPUT, a C++ source that registers it as KEY when the
# extension loads. Run by the build, once per .cu file, whenever that file changes.

file(READ "${KERNEL}" text)
if(text MATCHES "\\)tessiq_cu\"")
  message(FATAL_ERROR "${KERNEL} contains )tessiq_cu\", which ends the string it is embedded in")
endif()
file(WRITE "${OUTPUT}" "// Generated from csrc/custom_ops/${KEY} by embed.cmake. Edit that file, not this one.

#include \"tessiq/torch/custom_op.h\"

namespace {

[[maybe_unused]] const tessiq::custom_ops::KernelFile kFile(\"${KEY}\", R\"tessiq_cu(${text})tessiq_cu\");

} // namespace
")
