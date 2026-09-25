#!/usr/bin/env python3
# Copyright 2026 The Tessiq Authors.
# SPDX-License-Identifier: Apache-2.0
"""How much of a real model the backend runs itself.

`docs/op_coverage.md` says what the operation table holds, which does not say how much of a model
runs on the device. This imports the blocks from the test suite and counts.

Run with: python tools/model_coverage.py
"""

from __future__ import annotations

import collections
import dataclasses
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests" / "python"))

import torch  # noqa: E402

import tessiq  # noqa: E402
from test_models import EncoderBlock, ResidualBlock, capture  # noqa: E402


@dataclasses.dataclass(frozen=True)
class Coverage:
    """What the backend could run, against what it actually will.

    `claimable` counts operations the backend has an implementation for. `placed` counts the ones
    partitioning puts on the device, which can be fewer: a cluster of nothing but movement
    operations is left on the host.
    """

    claimable: int
    placed: int
    total: int
    on_host: collections.Counter[str]


def coverage(model: torch.nn.Module, example: torch.Tensor) -> Coverage:
    text, _ = tessiq.partition(capture(model, example))

    claimable = 0
    placed = 0
    on_host: collections.Counter[str] = collections.Counter()
    total = 0
    for line in text.splitlines():
        if "tessiq.supported" not in line:
            continue
        total += 1
        operation = line.split("=", 1)[1].strip().split("(")[0]
        if "tessiq.supported = true" in line:
            claimable += 1
        if "tessiq.cluster = " in line:
            placed += 1
        else:
            on_host[operation] += 1
    return Coverage(claimable, placed, total, on_host)


def main() -> None:
    blocks = [
        ("residual block", ResidualBlock().eval(), torch.randn(1, 16, 8, 8)),
        ("encoder block", EncoderBlock().eval(), torch.randn(2, 6, 32)),
    ]

    for name, model, example in blocks:
        with torch.no_grad():
            found = coverage(model, example)
        share = 100.0 * found.placed / found.total if found.total else 0.0
        print(
            f"\n{name}: {found.placed} of {found.total} operations on the device ({share:.0f}%),"
            f" {found.claimable} claimable"
        )
        if not found.on_host:
            continue
        print(f"  {'left on the host':<48}{'count':>6}")
        for operation, count in found.on_host.most_common():
            print(f"  {operation:<48}{count:>6}")


if __name__ == "__main__":
    main()
