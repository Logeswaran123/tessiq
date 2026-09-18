# Copyright 2026 The Tessera Authors.
# SPDX-License-Identifier: Apache-2.0

# Without Python_EXECUTABLE, FindPython picks whatever interpreter is first on PATH, which in a
# conda layout is the base environment rather than the active one. Building against 3.13 headers
# and importing into 3.12 produces a module that fails to load with no useful message.
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
message(STATUS "tessera: building the extension for ${Python_EXECUTABLE} (${Python_VERSION})")
find_package(pybind11 CONFIG REQUIRED)
find_package(Torch REQUIRED)

pybind11_add_module(_C MODULE
  module.cpp
  ../tessera/torch/backend.cpp
  ../tessera/torch/ops.cpp)

target_link_libraries(_C PRIVATE
  tessera::runtime
  tessera::support
  ${TORCH_LIBRARIES})

# torch's Python bindings live in libtorch_python, which TORCH_LIBRARIES does not include.
find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib" REQUIRED)
target_link_libraries(_C PRIVATE "${TORCH_PYTHON_LIBRARY}")

# Must match the ABI torch itself was built with, or every std::string crossing the boundary is
# silently misinterpreted. Read from torch rather than assumed.
target_compile_definitions(_C PRIVATE
  _GLIBCXX_USE_CXX11_ABI=$<IF:$<BOOL:${TORCH_CXX_FLAGS}>,1,1>)

# The extension links libtorch, which lives in the torch package next to ours in site-packages.
# Without an RPATH, importing tessera before torch fails with a bare "libtorch.so: cannot open
# shared object file" that says nothing about the cause.
set_target_properties(_C PROPERTIES
  INSTALL_RPATH "$ORIGIN;$ORIGIN/../torch/lib;${TORCH_INSTALL_PREFIX}/lib"
  BUILD_WITH_INSTALL_RPATH ON)

tessera_set_warnings(_C)
tessera_enable_sanitizers(_C)

install(TARGETS _C DESTINATION tessera)
