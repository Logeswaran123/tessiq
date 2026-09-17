# Copyright 2026 The Tessera Authors.
# SPDX-License-Identifier: Apache-2.0
"""Type stubs for the native extension module.

Hand-written because pybind11 does not emit stubs. Keep in step with csrc/bindings/module.cpp;
``mypy`` checks the Python package against this file, so a drift here is a silent loss of type
coverage rather than an error.
"""

__version__: str

def version() -> str: ...
def set_log_level(level: str) -> None: ...
def get_log_level() -> str: ...
def _raise_test_error(code: str) -> None: ...
