# Copyright 2026 The Tessiq Authors.
# SPDX-License-Identifier: Apache-2.0
"""Eager elementwise calls on a GPU tensor.

Each operation runs three ways on the same data:

  tessiq       the eager kernel, called on a tessiq GPU tensor
  fallback     what the call cost before it had one: the tensors copied to the host, PyTorch's
               CPU kernel, and the result copied back, as the CPU fallback does it
  torch cuda   PyTorch's own CUDA kernel, on a CUDA tensor

The first table is wall time per call with the device drained, which is what a caller waits for.
The second is the kernels' device time from CUPTI, which compares the generated kernel with
PyTorch's.

Run with: python bench/eager.py [--elements N ...]
"""

from __future__ import annotations

import argparse
import statistics
import time
from collections.abc import Callable

import torch
from kernel_timing import cache_and_dram_bytes, gigabytes_per_second, kernel_microseconds

import tessiq

RUNS = 50
WARMUP = 10

#: Each call reads one float32 operand and writes one, so this many bytes cross memory per element.
BYTES_PER_ELEMENT = 2 * 4

#: The row added in `x + row`, small enough that its reads stay in the cache.
ROW = 1024

OPERATIONS: dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = {
    "x * 0.5": lambda x, _: x * 0.5,
    "x + row": lambda x, row: x + row,
    "exp(x)": lambda x, _: torch.exp(x),
    "gelu(x)": lambda x, _: torch.nn.functional.gelu(x),
}


def wall_milliseconds(work: Callable[[], object], synchronise: Callable[[], None]) -> float:
    """Median wall time of one call, with the device drained before the clock stops."""
    for _ in range(WARMUP):
        work()
        synchronise()
    samples = []
    for _ in range(RUNS):
        started = time.perf_counter()
        work()
        synchronise()
        samples.append((time.perf_counter() - started) * 1e3)
    return statistics.median(samples)


def measure(count: int) -> None:
    torch.manual_seed(0)
    host = torch.randn(count // ROW, ROW)
    row = torch.randn(ROW)
    ours = (host.to(tessiq.gpu), row.to(tessiq.gpu))
    theirs = (host.cuda(), row.cuda())
    moved = count * BYTES_PER_ELEMENT

    def drain() -> None:
        tessiq.synchronize(tessiq.gpu)

    print(f"\n{count:,} elements, {moved / 2**20:,.0f} MiB moved per call")
    # Everything on the device before the fallback. While its copies run the GPU idles and clocks
    # down, and CUPTI sometimes drops a record from the session after them.
    device = {
        name: (
            kernel_microseconds(lambda fn=fn: fn(*ours), RUNS, WARMUP),
            kernel_microseconds(lambda fn=fn: fn(*theirs), RUNS, WARMUP),
        )
        for name, fn in OPERATIONS.items()
    }
    wall = {
        name: (
            wall_milliseconds(lambda fn=fn: fn(*ours), drain),
            wall_milliseconds(lambda fn=fn: fn(*theirs), torch.cuda.synchronize),
        )
        for name, fn in OPERATIONS.items()
    }
    x, y = ours
    host_path = {
        name: wall_milliseconds(lambda fn=fn: fn(x.cpu(), y.cpu()).to(tessiq.gpu), drain)
        for name, fn in OPERATIONS.items()
    }

    print("                  wall time per call                     device time")
    print("  operation      tessiq   fallback  torch cuda    tessiq  torch cuda       GB/s")
    for name in OPERATIONS:
        eager, cuda = wall[name]
        mine, reference = device[name]
        print(
            f"  {name:10s} {eager:8.3f} ms {host_path[name]:7.2f} ms {cuda:8.3f} ms"
            f"  {mine:8.1f} us {reference:8.1f} us  {gigabytes_per_second(moved, mine):9.0f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--elements", type=int, nargs="+", help="sizes; default: half the L2 and 16 times it"
    )
    args = parser.parse_args()
    if tessiq.gpu_count() == 0:
        print("no GPU, nothing to measure")
        return

    sizes = args.elements or [size // BYTES_PER_ELEMENT for size in cache_and_dram_bytes()]
    print(f"torch {torch.__version__}, medians of {RUNS} calls after {WARMUP} warm-up calls")
    for count in sizes:
        measure(count // ROW * ROW)


if __name__ == "__main__":
    main()
