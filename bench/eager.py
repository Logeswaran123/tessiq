# Copyright 2026 The Tessiq Authors.
# SPDX-License-Identifier: Apache-2.0
"""Eager calls on a GPU tensor.

Each elementwise operation runs three ways on the same data:

  tessiq       the eager kernel, called on a tessiq GPU tensor
  fallback     what the call cost before it had one: the tensors copied to the host, PyTorch's
               CPU kernel, and the result copied back, as the CPU fallback does it
  torch cuda   PyTorch's own CUDA kernel, on a CUDA tensor

Wall time is per call with the device drained, which is what a caller waits for. Device time is
the kernels' own, from CUPTI, which compares the generated kernel with PyTorch's.

Then whole model blocks run eagerly, one operation at a time, as a model outside `torch.compile`
does: on a tessiq GPU, on a CUDA GPU, and on the CPU. torch lets cuDNN use TF32 for convolutions
by default and tessiq asks it for fp32, so the CUDA column of the residual block is the faster
arithmetic.

Run with: python bench/eager.py [--elements N ...]
"""

from __future__ import annotations

import argparse
import copy
import statistics
import sys
import time
from collections.abc import Callable
from pathlib import Path

import torch
from kernel_timing import cache_and_dram_bytes, gigabytes_per_second, kernel_microseconds

import tessiq
from tessiq import _C

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


def blocks() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests" / "python"))
    from test_models import EncoderBlock, ResidualBlock

    torch.manual_seed(0)
    cases = [
        ("encoder, 512 wide, 128 steps, batch 8", EncoderBlock(512, 8, 2048), (8, 128, 512)),
        ("residual, 64 channels, 56x56, batch 8", ResidualBlock(64), (8, 64, 56, 56)),
    ]

    def drain() -> None:
        tessiq.synchronize(tessiq.gpu)

    print("\nmodel blocks run eagerly, input already on the device, wall time per call")
    print("  block                                    tessiq  torch cuda   torch cpu   host calls")
    for name, model, shape in cases:
        model.eval()
        example = torch.randn(shape)
        ours = copy.deepcopy(model).to(tessiq.gpu)
        theirs = copy.deepcopy(model).cuda()
        ours_in = example.to(tessiq.gpu)
        theirs_in = example.cuda()
        with torch.no_grad():
            mine = wall_milliseconds(lambda ours=ours, x=ours_in: ours(x), drain)
            cuda = wall_milliseconds(
                lambda theirs=theirs, x=theirs_in: theirs(x), torch.cuda.synchronize
            )
            cpu = wall_milliseconds(lambda model=model, x=example: model(x), lambda: None)
            before = sum(_C.cpu_fallback_counts().values())
            ours(ours_in)
            host_calls = sum(_C.cpu_fallback_counts().values()) - before
        print(f"  {name:38s} {mine:7.2f} ms {cuda:8.2f} ms {cpu:8.2f} ms {host_calls:9d}")


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
    blocks()


if __name__ == "__main__":
    main()
