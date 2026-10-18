#!/usr/bin/env python3
# Copyright 2026 The Tessiq Authors.
# SPDX-License-Identifier: Apache-2.0
"""Why a compiled call is slower than eager CUDA, split into parts that can be acted on.

`where_the_time_goes.py` reports the host's view of one call, in which a launch returns before
the work does. This adds the device's view: each kernel is launched many times between one pair of
stream marks, which gives the device's time for the kernels back to back, including the gap it
leaves between one kernel and the next. Next to the call's duration, that bounds how much of the
call is the host issuing work.

Run with: python bench/where_the_gap_is.py
"""

from __future__ import annotations

import collections
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests" / "python"))

import torch  # noqa: E402

import tessiq  # noqa: E402
from tessiq.costs import kernel_costs  # noqa: E402
from test_models import EncoderBlock  # noqa: E402

BATCH, STEPS, WIDTH = 8, 128, 32


def median_ms(call, *, reps: int = 30, sync=None) -> float:
    """Median wall time of one call, warmed up first."""
    with torch.no_grad():
        for _ in range(10):
            call()
        values = []
        for _ in range(reps):
            start = time.perf_counter()
            call()
            if sync is not None:
                sync()
            values.append((time.perf_counter() - start) * 1e3)
    return statistics.median(values)


def main() -> None:
    if tessiq.gpu_count() == 0:
        print("no GPU, nothing to measure")
        return

    torch.manual_seed(0)
    block = EncoderBlock(WIDTH, 4, 64).eval()
    example = torch.randn(BATCH, STEPS, WIDTH)

    on_cuda = EncoderBlock(WIDTH, 4, 64).eval().cuda()
    on_cuda.load_state_dict({k: v.cuda() for k, v in block.state_dict().items()})
    cuda_example = example.cuda()

    torch._dynamo.reset()
    ours = torch.compile(block, backend="tessiq", options={"device": str(tessiq.gpu)})

    inductor = torch.compile(on_cuda)

    print(f"encoder block, batch {BATCH}, {STEPS} steps, {WIDTH} wide\n")
    print("one call, wall clock")
    for name, run, sync in [
        ("eager cuda", lambda: on_cuda(cuda_example), torch.cuda.synchronize),
        ("inductor cuda", lambda: inductor(cuda_example), torch.cuda.synchronize),
        ("tessiq", lambda: ours(example), None),
    ]:
        print(f"  {name:<16}{median_ms(run, sync=sync):6.2f} ms")

    # The kernels back to back on an idle device, compiled from the same graph as the call above.
    recipe = tessiq.build_recipe(tessiq.capture(block, example), tessiq.gpu, use_cache=False)
    costs = kernel_costs(recipe)
    device_total = sum(cost.microseconds for cost in costs)

    call = median_ms(lambda: ours(example)) * 1000

    # Warmed up again, since a cold call profiles as a slow one.
    with torch.no_grad():
        for _ in range(10):
            ours(example)
        start = time.perf_counter()
        with tessiq.profile() as collected:
            ours(example)
        profiled = (time.perf_counter() - start) * 1e6
    host = collections.Counter()
    counts = collections.Counter()
    for event in collected.events:
        host[event.category] += event.microseconds
        counts[event.category] += 1
    measured = sum(host.values())

    # Back-to-back time is the least the device would take if the host issued everything at once,
    # so the rest of the call is the most a faster host could save.
    print(f"\nthe call takes          {call:7.0f} us on the host")
    print(f"its kernels back to back {device_total:7.0f} us, across {len(costs)} kernels")
    print(f"so the host accounts for up to {100 * (1 - device_total / call):.0f}% of the call")

    # Shares rather than microseconds, because profiling inflates the call.
    print(f"\nwhat the host spends it on, from a profiled call of {profiled:.0f} us")
    for category, micros in host.most_common():
        print(
            f"  {100 * micros / measured:5.0f}%   {category:<12}{counts[category]:>3} regions,"
            f" {micros / counts[category]:5.1f} us each while profiled"
        )

    print("\nslowest kernels")
    for cost in costs[:6]:
        print(f"  {cost.microseconds:7.2f} us   {cost.kind:<10}{cost.name}")


if __name__ == "__main__":
    main()
