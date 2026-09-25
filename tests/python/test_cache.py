# Copyright 2026 The Tessiq Authors.
# SPDX-License-Identifier: Apache-2.0
"""The recipe cache.

A graph that would compile to the same recipe must hit. One that would compile to a different
recipe must miss: a recipe built for another shape has buffers of the wrong size.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

import pytest
import torch
from torch._dynamo.backends.common import aot_autograd

import tessiq
from tessiq.cache import MAX_ENTRIES, signature


@pytest.fixture(autouse=True)
def _empty_cache() -> None:
    tessiq.clear_cache()


def captured(fn: Callable[..., Any], *inputs: torch.Tensor) -> torch.fx.GraphModule:
    graphs: list[torch.fx.GraphModule] = []

    def backend(graph_module: torch.fx.GraphModule, _inputs: object) -> Callable[..., object]:
        graphs.append(graph_module)
        return graph_module.forward

    torch._dynamo.reset()
    torch.compile(fn, backend=aot_autograd(fw_compiler=backend), dynamic=False)(*inputs)
    return graphs[0]


def test_the_same_graph_hits() -> None:
    graph = captured(lambda a, b: torch.relu(a + b), torch.randn(8), torch.randn(8))
    first = tessiq.build_recipe(graph)
    second = tessiq.build_recipe(graph)

    assert first is second
    assert tessiq.cache_stats().hits == 1
    assert tessiq.cache_stats().misses == 1


def test_a_different_shape_misses() -> None:
    # The memory plan is built from shapes, so the same structure at another size is a different
    # recipe. Handing back the first one would give every buffer the wrong size.
    small = captured(lambda a, b: torch.relu(a + b), torch.randn(8), torch.randn(8))
    large = captured(lambda a, b: torch.relu(a + b), torch.randn(4096), torch.randn(4096))

    first = tessiq.build_recipe(small)
    second = tessiq.build_recipe(large)

    assert first is not second
    assert first.arena_bytes != second.arena_bytes
    assert tessiq.cache_stats().misses == 2


def test_a_different_dtype_misses() -> None:
    single = captured(lambda a: torch.relu(a), torch.randn(8))
    double = captured(lambda a: torch.relu(a), torch.randn(8, dtype=torch.float64))
    assert tessiq.build_recipe(single) is not tessiq.build_recipe(double)


def test_a_different_structure_misses() -> None:
    relu = captured(lambda a, b: torch.relu(a + b), torch.randn(8), torch.randn(8))
    tanh = captured(lambda a, b: torch.tanh(a + b), torch.randn(8), torch.randn(8))
    assert tessiq.build_recipe(relu) is not tessiq.build_recipe(tanh)


def test_a_different_device_misses() -> None:
    graph = captured(lambda a, b: torch.relu(a + b), torch.randn(8), torch.randn(8))
    host = tessiq.build_recipe(graph, torch.device("cpu"))
    device = tessiq.build_recipe(graph, tessiq.cpu)
    assert host is not device


def test_the_cache_survives_a_dynamo_reset() -> None:
    # Dynamo caches compiled code against the code object and forgets it on reset. This cache is
    # keyed on the graph, so it survives.
    def model(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.tanh(torch.relu(a + b) * b)

    a, b = torch.randn(256), torch.randn(256)
    for _ in range(3):
        torch._dynamo.reset()
        torch.compile(model, backend="tessiq", dynamic=False)(a, b)

    stats = tessiq.cache_stats()
    assert stats.misses == 1, f"compiled more than once: {stats}"
    assert stats.hits == 2, f"expected two hits: {stats}"


def test_a_cached_recipe_still_gives_the_right_answer() -> None:
    def model(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.tanh(torch.relu(a + b) * b)

    a, b = torch.randn(8), torch.randn(8)
    graph = captured(model, a, b)

    first = tessiq.execute(tessiq.build_recipe(graph))(a, b)
    second = tessiq.execute(tessiq.build_recipe(graph))(a, b)

    assert tessiq.cache_stats().hits == 1
    assert torch.allclose(first[0], model(a, b), atol=1e-5)
    assert torch.equal(first[0], second[0])


def test_use_cache_false_compiles_every_time() -> None:
    graph = captured(lambda a: torch.relu(a), torch.randn(8))
    first = tessiq.build_recipe(graph, use_cache=False)
    second = tessiq.build_recipe(graph, use_cache=False)

    assert first is not second
    assert tessiq.cache_stats() == tessiq.CacheStats()


def test_the_cache_is_bounded() -> None:
    for size in range(1, MAX_ENTRIES + 5):
        tessiq.build_recipe(captured(lambda a: torch.relu(a), torch.randn(size)))

    stats = tessiq.cache_stats()
    assert stats.entries == MAX_ENTRIES
    assert stats.evictions == 4


def test_clearing_resets_the_counters() -> None:
    tessiq.build_recipe(captured(lambda a: torch.relu(a), torch.randn(8)))
    assert tessiq.cache_stats().entries == 1
    tessiq.clear_cache()
    assert tessiq.cache_stats() == tessiq.CacheStats()


def test_signature_is_stable_across_calls() -> None:
    graph = captured(lambda a, b: torch.relu(a + b), torch.randn(8), torch.randn(8))
    assert signature(graph, tessiq.cpu) == signature(graph, tessiq.cpu)


def test_compiling_from_several_threads_is_safe() -> None:
    graph = captured(lambda a, b: torch.relu(a + b), torch.randn(8), torch.randn(8))
    seen: list[tessiq.Recipe] = []
    lock = threading.Lock()

    def compile_once() -> None:
        recipe = tessiq.build_recipe(graph)
        with lock:
            seen.append(recipe)

    threads = [threading.Thread(target=compile_once) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(seen) == 8
    stats = tessiq.cache_stats()
    # Several threads may miss before the first one stores, which is a duplicated compile rather
    # than a wrong answer. What must hold is that the counters add up and nothing was lost.
    assert stats.hits + stats.misses == 8
    assert stats.entries == 1
