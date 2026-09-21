# Copyright 2026 The Tessiq Authors.
# SPDX-License-Identifier: Apache-2.0
"""What the generated reduction kernel costs, against PyTorch's own.

A reduction is memory bound in the same way an elementwise kernel is: it reads the input once and
writes far less, so the input's size is the work. Both sides are measured with the data already on
the device, because the transfer around a call dominates it otherwise and that is a different
measurement, made in `against_torch.py`.

The shape matters more here than it does for an elementwise kernel. Reducing the last axis walks
contiguous memory; reducing the first walks with a stride, and every block reads a column. Both are
measured for that reason.

Run with: python bench/reduction.py
"""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable
from typing import Any

import torch
from torch._decomp import core_aten_decompositions
from torch._dynamo.backends.common import aot_autograd

import tessiq
from tessiq.recipe import build_recipe

RUNS = 300
WARMUP = 100

#: Shapes and the axis to reduce, with why each one is here.
CASES = [
    ((4096, 4096), 1, "last axis, contiguous"),
    ((4096, 4096), 0, "first axis, strided"),
    ((1024, 16384), 1, "long axis, few outputs"),
    ((262144, 64), 1, "short axis, many outputs"),
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


def measure(work: Callable[[], object], synchronise: Callable[[], None]) -> float:
    """Median wall time in milliseconds, with the device drained before the clock stops."""
    for _ in range(WARMUP):
        work()
        synchronise()

    samples = []
    for _ in range(RUNS):
        started = time.perf_counter()
        work()
        synchronise()
        samples.append((time.perf_counter() - started) * 1e3)
    samples.sort()
    return statistics.median(samples)


def bandwidth(elements: int, milliseconds: float) -> float:
    """Gigabytes per second, counting the input read once. The output is negligible."""
    return elements * 4 / (milliseconds * 1e-3) / 2**30


def main() -> None:
    if tessiq.gpu_count() == 0:
        print("no GPU, nothing to measure")
        return

    print(f"{'shape':>16}  {'axis':>4}  {'tessiq':>9}  {'torch':>9}  {'tessiq':>9}  {'torch':>9}")
    print(f"{'':>16}  {'':>4}  {'ms':>9}  {'ms':>9}  {'GB/s':>9}  {'GB/s':>9}")

    for shape, axis, why in CASES:
        host = torch.randn(*shape)

        def model(x: torch.Tensor, axis: int = axis) -> torch.Tensor:
            return torch.sum(x, dim=axis)

        recipe = build_recipe(capture(model, host), tessiq.gpu, use_cache=False)
        kernels = list(recipe.kernels.values())
        if not kernels:
            print(f"{shape!s:>16}  {axis:>4}  no kernel ({why})")
            continue
        kernel = kernels[0]

        device = torch.empty(0, device=tessiq.gpu).device
        source = host.to(device)
        target = torch.empty(model(host).shape, dtype=host.dtype, device=device)
        ours = measure(
            lambda k=kernel, s=source, t=target: k.launch([s], t), torch.cuda.synchronize
        )

        cuda_input = host.cuda()
        theirs = measure(
            lambda x=cuda_input, a=axis: torch.sum(x, dim=a),
            torch.cuda.synchronize,
        )

        count = host.numel()
        print(
            f"{shape!s:>16}  {axis:>4}  {ours:>9.4f}  {theirs:>9.4f}  "
            f"{bandwidth(count, ours):>9.0f}  {bandwidth(count, theirs):>9.0f}   {why}"
        )


if __name__ == "__main__":
    main()
