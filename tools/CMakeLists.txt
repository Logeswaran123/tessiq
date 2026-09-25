# Copyright 2026 The Tessiq Authors.
# SPDX-License-Identifier: Apache-2.0

add_executable(tessiq-opt tessiq_opt.cpp)

# WHOLE_ARCHIVE because passes register themselves at static initialisation. A static library
# only contributes the objects something already references, so without this the linker drops
# every pass and the registry comes up empty, with no error.
target_link_libraries(tessiq-opt PRIVATE "$<LINK_LIBRARY:WHOLE_ARCHIVE,tessiq_ir>"
  tessiq::codegen)

tessiq_set_warnings(tessiq-opt)
tessiq_enable_sanitizers(tessiq-opt)
