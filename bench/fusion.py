# Copyright 2026 The Tessiq Authors.
# SPDX-License-Identifier: Apache-2.0
"""What fusion is worth.

Compares one generated kernel against the same recipe with the kernels removed, so the only
difference is one kernel against separate operations.

Without the recipe's kernel, each operation runs as an eager kernel of its own, one launch per
operation with every intermediate written to memory and read back. Both sides also copy the inputs
to the device and the result back, which is most of a call at these sizes.

Run with: python bench/fusion.py
"""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable
from dataclasses import replace
from typing import Any

import torch
from torch._decomp import core_aten_decompositions
from torch._dynamo.backends.common import aot_autograd

import tessiq
from tessiq.recipe import build_recipe

RUNS = 50
WARMUP = 10


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


def measure(work: Callable[[], object]) -> tuple[float, float, float]:
    """Median, tenth and ninetieth percentile, in milliseconds."""
    for _ in range(WARMUP):
        work()
    samples = []
    for _ in range(RUNS):
        started = time.perf_counter()
        work()
        samples.append((time.perf_counter() - started) * 1e3)
    samples.sort()
    return (
        statistics.median(samples),
        samples[len(samples) // 10],
        samples[(len(samples) * 9) // 10],
    )


def chain(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """The fused chain the README quotes."""
    return torch.sigmoid(torch.relu(a * b + c))


def main() -> None:
    if tessiq.gpu_count() == 0:
        print("no GPU, nothing to measure")
        return

    print(f"torch {torch.__version__}, {RUNS} runs after {WARMUP} warmup")
    print(f"{'elements':>10} {'fused':>22} {'unfused':>22} {'ratio':>7}")

    for exponent in (16, 18, 20, 22):
        count = 1 << exponent
        inputs = (torch.randn(count), torch.randn(count), torch.randn(count))

        fused = build_recipe(capture(chain, *inputs), tessiq.gpu, use_cache=False)
        unfused = replace(fused, kernels={}, fused_away=frozenset())

        run_fused = tessiq.execute(fused)
        run_unfused = tessiq.execute(unfused)

        # Both have to give the same numbers before either is timed.
        assert torch.allclose(run_fused(*inputs)[0], run_unfused(*inputs)[0], atol=1e-6)

        # Bound as defaults, since a closure over loop variables would see only the last size.
        f_median, f_low, f_high = measure(lambda r=run_fused, x=inputs: r(*x))
        u_median, u_low, u_high = measure(lambda r=run_unfused, x=inputs: r(*x))
        print(
            f"{count:>10} "
            f"{f_median:>8.3f} ms ({f_low:.2f}-{f_high:.2f}) "
            f"{u_median:>8.3f} ms ({u_low:.2f}-{u_high:.2f}) "
            f"{u_median / f_median:>6.1f}x"
        )

    recipe = build_recipe(
        capture(chain, torch.randn(1024), torch.randn(1024), torch.randn(1024)),
        tessiq.gpu,
        use_cache=False,
    )
    print(
        f"\n{len(recipe.devices)} operations, "
        f"{len(recipe.kernels)} launch, "
        f"{len(recipe.fused_away)} values never written to memory"
    )


if __name__ == "__main__":
    main()
