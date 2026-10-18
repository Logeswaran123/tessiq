# Copyright 2026 The Tessiq Authors.
# SPDX-License-Identifier: Apache-2.0
"""What the generated reduction kernel costs, against PyTorch's own.

A reduction is memory bound: it reads the input once and writes far less. Both sides run with the
data already on the device and are timed from CUPTI's records of their kernels; the transfers
around a call are `against_torch.py`'s subject.

Reducing the last axis walks contiguous memory, and reducing the first walks with a stride, so both
are measured, at one size held in the GPU's L2 cache and one read from DRAM.

Run with: python bench/reduction.py [--elements N ...]
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from math import isqrt
from typing import Any

import torch
from kernel_timing import cache_and_dram_bytes, gigabytes_per_second, kernel_microseconds
from torch._decomp import core_aten_decompositions
from torch._dynamo.backends.common import aot_autograd

import tessiq
from tessiq.recipe import build_recipe

RUNS = 300
WARMUP = 100

#: The shape for `n` elements, the axis to reduce, and why each case is here.
CASES: list[tuple[Callable[[int], tuple[int, int]], int, str]] = [
    (lambda n: (isqrt(n), n // isqrt(n)), 1, "last axis, contiguous"),
    (lambda n: (isqrt(n), n // isqrt(n)), 0, "first axis, strided"),
    (lambda n: (512, n // 512), 1, "long axis, few outputs"),
    (lambda n: (n // 64, 64), 1, "short axis, many outputs"),
]


def capture(fn: Callable[..., Any], *inputs: torch.Tensor) -> torch.fx.GraphModule:
    graphs: list[torch.fx.GraphModule] = []

    def backend(graph_module: torch.fx.GraphModule, _inputs: object) -> Callable[..., object]:
        graphs.append(graph_module)
        return graph_module.forward

    torch._dynamo.reset()
    torch.compile(
        fn,
        backend=aot_autograd(fw_compiler=backend, decompositions=core_aten_decompositions()),
        dynamic=False,
    )(*inputs)
    return graphs[0]


def compare(shape: tuple[int, int], axis: int, why: str) -> None:
    host = torch.randn(*shape)

    def model(x: torch.Tensor) -> torch.Tensor:
        return torch.sum(x, dim=axis)

    recipe = build_recipe(capture(model, host), tessiq.gpu, use_cache=False)
    kernels = list(recipe.kernels.values())
    if not kernels:
        print(f"{shape!s:>16}  {axis:>4}  no kernel ({why})")
        return

    source = host.to(tessiq.gpu)
    target = torch.empty(model(host).shape, dtype=host.dtype, device=tessiq.gpu)
    ours = kernel_microseconds(lambda: kernels[0].launch([source], target), RUNS, WARMUP)
    cuda_input = host.cuda()
    theirs = kernel_microseconds(lambda: torch.sum(cuda_input, dim=axis), RUNS, WARMUP)

    read = host.numel() * host.element_size()
    print(
        f"{shape!s:>16}  {axis:>4}  {ours:>9.1f}  {theirs:>9.1f}  "
        f"{gigabytes_per_second(read, ours):>9.0f}  {gigabytes_per_second(read, theirs):>9.0f}"
        f"   {why}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--elements", type=int, nargs="+", help="input sizes; default: half the L2 and 16 times it"
    )
    args = parser.parse_args()
    if tessiq.gpu_count() == 0:
        print("no GPU, nothing to measure")
        return

    sizes = args.elements or [size // 4 for size in cache_and_dram_bytes()]
    print(f"kernel execution, medians of {RUNS} calls; GB/s counts the input read once")
    print(f"{'shape':>16}  {'axis':>4}  {'tessiq':>9}  {'torch':>9}  {'tessiq':>9}  {'torch':>9}")
    print(f"{'':>16}  {'':>4}  {'us':>9}  {'us':>9}  {'GB/s':>9}  {'GB/s':>9}")
    for count in sizes:
        for shape_of, axis, why in CASES:
            compare(shape_of(count), axis, why)


if __name__ == "__main__":
    main()
