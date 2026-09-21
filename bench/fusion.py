# Copyright 2026 The Tessiq Authors.
# SPDX-License-Identifier: Apache-2.0
"""What fusion is worth.

Compares one generated kernel against the same graph with the kernels removed. Same recipe, same
device, same everything else, so the difference is whether the chain ran as one kernel or as
separate operations.

Be clear about what the other side is. Without a kernel, each operation on a tessiq tensor reaches
PyTorch's CPU implementation through the fallback, which copies to the host and back. So this
measures a generated kernel against the fallback, which is what the backend does otherwise, rather
than against a hand-written CUDA kernel per operation.

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
    """The chain from the phase 6 milestone."""
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

        # Same numbers either way, checked here rather than assumed, because a benchmark that
        # measures the wrong answer quickly is worse than no benchmark.
        assert torch.allclose(run_fused(*inputs)[0], run_unfused(*inputs)[0], atol=1e-6)

        # Bound as defaults: a bare closure over the loop variables reads whatever they hold when
        # it runs, which for a benchmark timing several sizes is the last one every time.
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
