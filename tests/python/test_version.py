# Copyright 2026 The Tessiq Authors.
# SPDX-License-Identifier: Apache-2.0
"""The package imports, and its version agrees with the native module's."""

from __future__ import annotations

import re

import tessiq
from tessiq import _C


def test_package_imports() -> None:
    assert tessiq.__version__


def test_version_is_semver() -> None:
    assert re.fullmatch(r"\d+\.\d+\.\d+", tessiq.__version__), (
        f"version {tessiq.__version__!r} is not major.minor.patch"
    )


def test_python_and_native_versions_agree() -> None:
    # The version is defined once, in CMakeLists.txt, and flows into both. A mismatch means the
    # extension in this environment was built from different sources than the package.
    assert tessiq.__version__ == _C.version()
