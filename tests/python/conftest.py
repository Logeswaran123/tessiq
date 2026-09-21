# Copyright 2026 The Tessiq Authors.
# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for the Python tests.

Mostly one thing: reading ops.toml. Several test modules check the backend against the table
rather than against a list they carry themselves, so the table is the only place an operation is
written down and a test cannot fall behind it.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# tomllib is standard from 3.11. The project supports 3.10, where tomli is the same parser under
# its original name.
if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[2]
OPS_TOML = ROOT / "ops.toml"


def load_operations() -> list[dict[str, Any]]:
    """Every row of ops.toml, in the order it is written."""
    with OPS_TOML.open("rb") as handle:
        return list(tomllib.load(handle)["op"])
