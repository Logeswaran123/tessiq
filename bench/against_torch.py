# Copyright 2026 The Tessiq Authors.
# SPDX-License-Identifier: Apache-2.0
"""How tessiq compares to PyTorch on the same hardware.

`fusion.py` asks what fusion buys inside tessiq. This asks whether tessiq is faster than PyTorch,
in two halves that can point opposite ways:

  kernel       The generated kernel against Inductor's, with the data already on the device,
               timed from CUPTI's records of the kernels. This measures the code generator.
  end to end   A whole `torch.compile` call against PyTorch's, from host tensors. This also
               measures everything around the kernel.

The kernels run at two sizes: one held in the GPU's L2 cache and one read from DRAM.

Run with: python bench/against_torch.py [--elements N ...]
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from kernel_timing import cache_and_dram_bytes, gigabytes_per_second, kernel_microseconds
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
    """The fused chain the README quotes."""
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

    A launch returns as soon as it is queued, so timing without synchronising measures only the
    launch.
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
    """No synchronise needed: the result is copied back to the host, which waits for the device."""


def kernels(count: int) -> float:
    """Prints the two kernels' execution time at `count` elements and returns tessiq's."""
    host = (torch.randn(count), torch.randn(count), torch.randn(count))
    recipe = build_recipe(capture(chain, *host), tessiq.gpu, use_cache=False)
    kernel = next(iter(recipe.kernels.values()))
    ours = [x.to(tessiq.gpu) for x in host]
    out = torch.empty(count, dtype=torch.float32, device=tessiq.gpu)
    resident = tuple(x.cuda() for x in host)
    torch._dynamo.reset()
    inductor = torch.compile(chain)

    mine = kernel_microseconds(lambda: kernel.launch(ours, out), RUNS, WARMUP)
    theirs = kernel_microseconds(lambda: inductor(*resident), RUNS, WARMUP)
    moved = count * BYTES_PER_ELEMENT
    print(f"  {count:,} elements, {moved / 2**20:,.0f} MiB moved")
    print(
        f"    tessiq generated       {mine:9.1f} us   {gigabytes_per_second(moved, mine):6.0f} GB/s"
    )
    print(
        f"    torch.compile inductor {theirs:9.1f} us   {gigabytes_per_second(moved, theirs):6.0f} GB/s"
    )
    return mine


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--elements", type=int, nargs="+", help="chain sizes; default: half the L2 and 16 times it"
    )
    args = parser.parse_args()
    if tessiq.gpu_count() == 0:
        print("no GPU, nothing to measure")
        return

    sizes = args.elements or [size // BYTES_PER_ELEMENT for size in cache_and_dram_bytes()]
    print(f"torch {torch.__version__}, medians of {RUNS} calls after {WARMUP} warm-up calls\n")
    print("kernel execution, data already on the device")
    mine = {count: kernels(count) for count in sizes}

    # --- the whole call, from host tensors --------------------------------------------------
    count = sizes[0]
    host = (torch.randn(count), torch.randn(count), torch.randn(count))
    resident = tuple(x.cuda() for x in host)
    torch._dynamo.reset()
    inductor = torch.compile(chain)
    inductor(*resident)
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

    print(f"\nwhole call, inputs on the host, {count:,} elements, wall clock")
    print(f"  tessiq                 {end_to_end:8.4f} ms")
    print(
        f"  torch, already on gpu  {torch_resident:8.4f} ms   {end_to_end / torch_resident:5.0f}x"
    )
    print(f"  torch, moving too      {torch_moving:8.4f} ms   {end_to_end / torch_moving:5.2f}x")
    print(
        f"\n  the kernel is {mine[count] / 1e3 / end_to_end * 100:.0f}% of the tessiq call; "
        "the rest is moving data across PCIe"
    )

    block()


def block() -> None:
    """The same question on a whole transformer encoder block rather than one chain.

    One chain is a single kernel, so it compares code generation. A block is many launches and the
    transfers between them, which is what a user runs.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests" / "python"))
    from test_models import EncoderBlock

    width, heads, hidden, batch, steps = 512, 8, 2048, 8, 128
    torch.manual_seed(0)
    host = EncoderBlock(width, heads, hidden).eval()
    example = torch.randn(batch, steps, width)
    weights = sum(p.numel() for p in host.parameters()) * 4 / 2**20

    resident = EncoderBlock(width, heads, hidden).eval()
    resident.load_state_dict(host.state_dict())
    resident = resident.cuda()
    on_device = example.cuda()

    with torch.no_grad():
        torch._dynamo.reset()
        ours = torch.compile(
            host, backend="tessiq", dynamic=False, options={"device": str(tessiq.gpu)}
        )
        ours(example)

        torch._dynamo.reset()
        inductor = torch.compile(resident)
        inductor(on_device)

        mine = measure(lambda: ours(example), nothing)
        eager_moving = measure(lambda: resident(example.cuda()).cpu(), torch.cuda.synchronize)
        eager_resident = measure(lambda: resident(on_device), torch.cuda.synchronize)
        compiled = measure(lambda: inductor(on_device), torch.cuda.synchronize)
        on_cpu = measure(lambda: host(example), nothing)

    print(f"\nencoder block, {width} wide over {steps} steps, batch {batch}")
    print(f"  weights {weights:.0f} MB, input {example.numel() * 4 / 2**20:.0f} MB")
    print(f"  tessiq                 {mine:8.4f} ms")
    print(f"  torch cuda, moving too {eager_moving:8.4f} ms   {mine / eager_moving:5.1f}x")
    print(f"  torch cuda, resident   {eager_resident:8.4f} ms   {mine / eager_resident:5.1f}x")
    print(f"  torch inductor         {compiled:8.4f} ms   {mine / compiled:5.1f}x")
    print(f"  torch cpu              {on_cpu:8.4f} ms   {mine / on_cpu:5.1f}x")
    print(
        "\n  the weights cross once rather than every call: AOTAutograd lifts them to graph\n"
        "  inputs, and the executor keeps the device copy of an input that has not changed."
    )


if __name__ == "__main__":
    main()
