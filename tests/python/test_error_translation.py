# Copyright 2026 The Tessiq Authors.
# SPDX-License-Identifier: Apache-2.0
"""Every ErrorCode maps to the Python exception type a caller would expect.

The mapping lives in csrc/bindings/module.cpp and is the single exception boundary of the project
Python code distinguishes these with ``except`` clauses, so a regression
here silently forces callers to match on message text instead.
"""

from __future__ import annotations

import pytest

from tessiq import _C


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        ("invalid_argument", ValueError),
        ("unsupported", TypeError),
        ("not_found", KeyError),
        ("resource_exhausted", MemoryError),
        ("internal", RuntimeError),
    ],
)
def test_error_code_maps_to_exception_type(code: str, expected: type[BaseException]) -> None:
    with pytest.raises(expected):
        _C._raise_test_error(code)


def test_message_retains_the_code_name() -> None:
    with pytest.raises(ValueError, match="invalid_argument"):
        _C._raise_test_error("invalid_argument")
    with pytest.raises(MemoryError, match="resource_exhausted"):
        _C._raise_test_error("resource_exhausted")
