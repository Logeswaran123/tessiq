# Copyright 2026 The Tessiq Authors.
# SPDX-License-Identifier: Apache-2.0
"""What the recipe cache and the move analysis save.

Two separate questions, both about overhead rather than arithmetic, because no kernels exist yet
and every operation still reaches PyTorch's CPU implementation through the fallback.

  compile   How long building a recipe takes against answering from the cache.
  call      How long a call takes when only the edges of a cluster move, against moving at every
            step, which is what the executor did before recipes.

Run with: python bench/compile_overhead.py
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
from tessiq.cache import signature
from tessiq.recipe import Recipe

#: Timed runs kept per measurement. Enough for a stable median without the script taking minutes.
RUNS = 200

#: Runs discarded first. The first few calls pay for lazily built caches inside torch and for the
#: allocator taking its first blocks from the device.
WARMUP = 20


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
    """Median, tenth and ninetieth percentile, in microseconds."""
    for _ in range(WARMUP):
        work()

    samples = []
    for _ in range(RUNS):
        started = time.perf_counter()
        work()
        samples.append((time.perf_counter() - started) * 1e6)

    samples.sort()
    return (
        statistics.median(samples),
        samples[len(samples) // 10],
        samples[(len(samples) * 9) // 10],
    )


def report(label: str, timing: tuple[float, float, float]) -> None:
    median, low, high = timing
    print(f"  {label:<28} {median:8.1f} us   (p10 {low:.1f}, p90 {high:.1f})")


def moves_everywhere(recipe: Recipe) -> Recipe:
    """The same recipe with the move analysis switched off.

    Every node claims it has to move its arguments, which is what the executor assumed before it
    had a recipe to consult. Nothing else differs, so the gap is the analysis and nothing else.
    """
    return Recipe(
        graph_module=recipe.graph_module,
        devices=recipe.devices,
        needs_move=frozenset(recipe.devices),
        output_spec=recipe.output_spec,
        arena_bytes=recipe.arena_bytes,
        naive_bytes=recipe.naive_bytes,
    )


def model(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Twenty elementwise layers, which is long enough for the overheads to separate.

    A five-node graph is not: compiling one takes about as long as computing its cache key, and
    the difference between moving once and moving five times is lost in the noise. Both effects
    are about the number of nodes, so the graph has to have some.
    """
    x = a + b
    for _ in range(20):
        x = torch.tanh(torch.sigmoid(torch.relu(x)))
    return x


def main() -> None:
    inputs = (torch.randn(1024), torch.randn(1024))
    graph = capture(model, *inputs)
    recipe = tessiq.build_recipe(graph)

    print(f"torch {torch.__version__}, {RUNS} runs after {WARMUP} warmup")
    print(
        f"{len(recipe.devices)} steps, {len(recipe.needs_move)} of them moving arguments, "
        f"arena {recipe.arena_bytes} bytes against {recipe.naive_bytes} with no reuse\n"
    )

    print("compile")
    tessiq.clear_cache()
    report("build a recipe", measure(lambda: tessiq.build_recipe(graph, use_cache=False)))
    tessiq.build_recipe(graph)
    report("answer from the cache", measure(lambda: tessiq.build_recipe(graph)))
    report("of which, the cache key", measure(lambda: signature(graph, tessiq.cpu)))

    print("\ncall")
    lean = tessiq.execute(recipe)
    heavy = tessiq.execute(moves_everywhere(recipe))
    report("moving only at the edges", measure(lambda: lean(*inputs)))
    report("moving at every step", measure(lambda: heavy(*inputs)))


if __name__ == "__main__":
    main()
