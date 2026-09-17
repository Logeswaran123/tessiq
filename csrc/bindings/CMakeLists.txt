# Copyright 2026 The Tessera Authors.
# SPDX-License-Identifier: Apache-2.0

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_C MODULE module.cpp)

target_link_libraries(_C PRIVATE tessera::support)

tessera_set_warnings(_C)
tessera_enable_sanitizers(_C)

# scikit-build-core installs into the wheel from here; the path must match the Python package
# layout so that `from tessera import _C` resolves.
install(TARGETS _C DESTINATION tessera)
