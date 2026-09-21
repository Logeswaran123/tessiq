# Copyright 2026 The Tessiq Authors.
# SPDX-License-Identifier: Apache-2.0
"""How tessiq compares to PyTorch on the same hardware.

`fusion.py` asks what fusion buys inside tessiq. This asks the question a user would: is any of
this faster than just using PyTorch. The answer has two halves and they point opposite ways, so
both are measured here rather than whichever one flatters.

  kernel       The generated kernel against Inductor's, with the data already on the device.
               This measures the code generator.
  end to end   A whole `torch.compile` call against PyTorch's, from host tensors. This measures
               the path around the kernel, which is currently what dominates.

Run with: python bench/against_torch.py
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

RUNS = 100
WARMUP = 20

#: Reads and writes one element of each input and the output, so this many bytes cross memory per
#: element. Three inputs and one output, four bytes each.
BYTES_PER_ELEMENT = 4 * 4


def chain(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """The chain from the phase 6 milestone."""
    return torch.sigmoid(torch.relu(a * b + c))


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
    """Median wall time in milliseconds, with the device drained before the clock stops.

    A launch returns as soon as it is queued, so timing one without synchronising measures how
    long it takes to ask rather than how long it takes.
    """
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


def nothing() -> None:
    """No synchronise needed: the tessiq path already drains the device before it returns."""


def bandwidth(elements: int, milliseconds: float) -> float:
    return elements * BYTES_PER_ELEMENT / (milliseconds * 1e-3) / 2**30


def main() -> None:
    if tessiq.gpu_count() == 0:
        print("no GPU, nothing to measure")
        return

    count = 1 << 22
    host = (torch.randn(count), torch.randn(count), torch.randn(count))
    resident = tuple(x.cuda() for x in host)

    print(f"torch {torch.__version__}, {count:,} elements, {RUNS} runs after {WARMUP} warmup\n")

    # --- the kernel, with nothing to transfer -----------------------------------------------
    recipe = build_recipe(capture(chain, *host), tessiq.gpu, use_cache=False)
    kernel = next(iter(recipe.kernels.values()))
    ours = [x.to(tessiq.gpu) for x in host]
    out = torch.empty(count, dtype=torch.float32, device=tessiq.gpu)

    torch._dynamo.reset()
    inductor = torch.compile(chain)
    inductor(*resident)

    mine = measure(lambda: kernel.launch(ours, out), torch.cuda.synchronize)
    theirs = measure(lambda: inductor(*resident), torch.cuda.synchronize)

    print("kernel, data already on the device")
    print(f"  tessiq generated       {mine:8.4f} ms   {bandwidth(count, mine):6.0f} GB/s")
    print(f"  torch.compile inductor {theirs:8.4f} ms   {bandwidth(count, theirs):6.0f} GB/s")
    print(f"  ratio                  {mine / theirs:8.2f}x\n")

    # --- the whole call, from host tensors --------------------------------------------------
    torch._dynamo.reset()
    compiled = torch.compile(
        chain, backend="tessiq", dynamic=False, options={"device": str(tessiq.gpu)}
    )
    compiled(*host)

    def torch_with_transfers() -> torch.Tensor:
        moved = [x.cuda() for x in host]
        return inductor(*moved).cpu()

    end_to_end = measure(lambda: compiled(*host), nothing)
    torch_resident = measure(lambda: inductor(*resident), torch.cuda.synchronize)
    torch_moving = measure(torch_with_transfers, torch.cuda.synchronize)

    print("whole call, inputs on the host")
    print(f"  tessiq                 {end_to_end:8.4f} ms")
    print(
        f"  torch, already on gpu  {torch_resident:8.4f} ms   {end_to_end / torch_resident:5.0f}x"
    )
    print(f"  torch, moving too      {torch_moving:8.4f} ms   {end_to_end / torch_moving:5.2f}x")
    print(
        f"\n  the kernel is {mine / end_to_end * 100:.0f}% of the tessiq call; "
        "the rest is moving data across PCIe"
    )


if __name__ == "__main__":
    main()
