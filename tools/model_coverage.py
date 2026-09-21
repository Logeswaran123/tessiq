#!/usr/bin/env python3
# Copyright 2026 The Tessiq Authors.
# SPDX-License-Identifier: Apache-2.0
"""How much of a real model the backend runs itself.

`docs/op_coverage.md` says what the operation table holds. That is the wrong number to quote on
its own: a table can list forty operations and still leave most of a model on the host, because
what matters is how often the operations a model actually uses appear. This imports the blocks
from the test suite and counts.

Run with: python tools/model_coverage.py
"""

from __future__ import annotations

import collections
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests" / "python"))

import torch  # noqa: E402

import tessiq  # noqa: E402
from test_models import EncoderBlock, ResidualBlock, capture  # noqa: E402


def coverage(model: torch.nn.Module, example: torch.Tensor) -> tuple[int, int, collections.Counter]:
    """Operations claimed, operations total, and what was declined."""
    text, _ = tessiq.partition(capture(model, example))

    claimed = 0
    declined: collections.Counter[str] = collections.Counter()
    for line in text.splitlines():
        if "tessiq.supported" not in line:
            continue
        if "tessiq.supported = true" in line:
            claimed += 1
        else:
            declined[line.split("=", 1)[1].strip().split("(")[0]] += 1
    return claimed, claimed + sum(declined.values()), declined


def main() -> None:
    blocks = [
        ("residual block", ResidualBlock().eval(), torch.randn(1, 16, 8, 8)),
        ("encoder block", EncoderBlock().eval(), torch.randn(2, 6, 32)),
    ]

    for name, model, example in blocks:
        with torch.no_grad():
            claimed, total, declined = coverage(model, example)
        share = 100.0 * claimed / total if total else 0.0
        print(f"\n{name}: {claimed} of {total} operations on the device ({share:.0f}%)")
        if not declined:
            continue
        print(f"  {'left on the host':<48}{'count':>6}")
        for operation, count in declined.most_common():
            print(f"  {operation:<48}{count:>6}")


if __name__ == "__main__":
    main()
