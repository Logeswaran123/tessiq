#!/usr/bin/env python3
# Copyright 2026 The Tessiq Authors.
# SPDX-License-Identifier: Apache-2.0
"""What a compiled call actually spends its time on.

`against_torch.py` says how far behind PyTorch a whole block is. This says why: how much of a call
is kernels, how much is moving data across PCIe, and how much is steps that have no kernel and go
one operation at a time through PyTorch's dispatcher.

The clock is the host's and every region is drained before it stops, so the total here is longer
than the call it describes. The proportions are the point.

Run with: python bench/where_the_time_goes.py [trace.json]
"""

from __future__ import annotations

import collections
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests" / "python"))

import torch  # noqa: E402

import tessiq  # noqa: E402
from test_models import EncoderBlock  # noqa: E402


def main() -> None:
    if tessiq.gpu_count() == 0:
        print("no GPU, nothing to measure")
        return

    width, heads, hidden, batch, steps = 512, 8, 2048, 8, 128
    torch.manual_seed(0)
    block = EncoderBlock(width, heads, hidden).eval()
    example = torch.randn(batch, steps, width)

    torch._dynamo.reset()
    compiled = torch.compile(
        block, backend="tessiq", dynamic=False, options={"device": str(tessiq.gpu)}
    )

    with torch.no_grad():
        # Warm up past compilation and let the weights settle on the device, so what is measured is
        # a steady-state call rather than the first one.
        for _ in range(8):
            compiled(example)
        with tessiq.profile() as collected:
            compiled(example)

    print(f"encoder block, {width} wide over {steps} steps, batch {batch}")
    print(f"  input {example.numel() * 4 / 2**20:.0f} MB\n")
    print(collected.summary())

    print("\nslowest steps")
    for event in collected.slowest(8):
        moved = f"   {event.nbytes / 2**20:.1f} MB" if event.nbytes else ""
        print(f"  {event.microseconds:8.1f} us  {event.category:9} {event.name[:40]}{moved}")

    interpreted: collections.Counter[str] = collections.Counter()
    for event in collected.events:
        if event.category == "interpreted":
            interpreted[event.name] += int(event.microseconds)
    if interpreted:
        print("\ninterpreted time by operation")
        for name, micros in interpreted.most_common(6):
            print(f"  {micros:8d} us  {name}")

    if len(sys.argv) > 1:
        collected.write_chrome_trace(sys.argv[1])
        print(f"\nwrote {sys.argv[1]}; open it in chrome://tracing or ui.perfetto.dev")


if __name__ == "__main__":
    main()
