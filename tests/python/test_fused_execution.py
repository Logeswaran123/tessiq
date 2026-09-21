# Copyright 2026 The Tessiq Authors.
# SPDX-License-Identifier: Apache-2.0
"""Running a model with generated kernels.

The point of these is not that a kernel runs. It is that running one gives the same numbers as not
running one, which is the only claim that matters and the one a code generator is most able to
break quietly.
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import replace
from typing import Any

import pytest
import torch
from torch._decomp import core_aten_decompositions
from torch._dynamo.backends.common import aot_autograd

import tessiq
from tessiq.recipe import build_recipe

pytestmark = pytest.mark.cuda


@pytest.fixture(autouse=True)
def _needs_gpu() -> None:
    if tessiq.gpu_count() == 0:
        pytest.skip("needs a gpu")
    torch._dynamo.reset()


def captured(fn: Callable[..., Any], *inputs: torch.Tensor) -> torch.fx.GraphModule:
    graphs: list[torch.fx.GraphModule] = []

    def backend(graph_module: torch.fx.GraphModule, _inputs: object) -> Callable[..., object]:
        graphs.append(graph_module)
        return graph_module.forward

    torch.compile(
        fn,
        backend=aot_autograd(fw_compiler=backend, decompositions=core_aten_decompositions()),
        dynamic=False,
    )(*inputs)
    return graphs[0]


def chain(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(torch.relu(a * b + c))


def test_a_chain_becomes_one_launch() -> None:
    inputs = (torch.randn(1024), torch.randn(1024), torch.randn(1024))
    recipe = build_recipe(captured(chain, *inputs), tessiq.gpu, use_cache=False)

    assert len(recipe.kernels) == 1, recipe.kernels
    assert recipe.fused_away == {"mul", "add", "relu"}, recipe.fused_away
    # Four operations in the graph, one of which is now a kernel and three of which it subsumes.
    assert len(recipe.devices) == 4


def test_the_kernel_agrees_with_eager() -> None:
    inputs = (torch.randn(4096), torch.randn(4096), torch.randn(4096))
    compiled = torch.compile(
        chain, backend="tessiq", dynamic=False, options={"device": str(tessiq.gpu)}
    )
    assert torch.allclose(compiled(*inputs), chain(*inputs), atol=1e-5)


def test_the_kernel_agrees_with_the_same_graph_unfused() -> None:
    # The sharper comparison. Same recipe, same device, same everything, with the kernels removed,
    # so the only difference is whether the chain ran as one kernel or four operations.
    inputs = (torch.randn(4096), torch.randn(4096), torch.randn(4096))
    fused = build_recipe(captured(chain, *inputs), tessiq.gpu, use_cache=False)
    unfused = replace(fused, kernels={}, fused_away=frozenset())

    with_kernel = tessiq.execute(fused)(*inputs)[0]
    without = tessiq.execute(unfused)(*inputs)[0]
    assert torch.allclose(with_kernel, without, atol=1e-6)


def test_the_kernel_agrees_with_the_host_reference() -> None:
    # The CPU backend is written to be obviously correct rather than fast, which is what makes it
    # worth comparing against. This is the check the phase milestone asks for.
    inputs = (torch.randn(8192), torch.randn(8192), torch.randn(8192))
    graph = captured(chain, *inputs)

    on_gpu = tessiq.execute(build_recipe(graph, tessiq.gpu, use_cache=False))(*inputs)[0]
    on_host = tessiq.execute(build_recipe(graph, tessiq.cpu, use_cache=False))(*inputs)[0]
    assert torch.allclose(on_gpu, on_host, atol=1e-6)


@pytest.mark.parametrize("size", [1, 255, 256, 257, 4095, 4096])
def test_sizes_around_the_block_boundary(size: int) -> None:
    # The block is 256 threads. A missing bounds check shows up only where the count does not
    # divide it, so the sizes that do are here as controls rather than as the interesting case.
    inputs = (torch.randn(size), torch.randn(size), torch.randn(size))
    recipe = build_recipe(captured(chain, *inputs), tessiq.gpu, use_cache=False)
    assert torch.allclose(tessiq.execute(recipe)(*inputs)[0], chain(*inputs), atol=1e-5)


def test_a_matmul_splits_the_chain_and_both_halves_agree() -> None:
    def model(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(torch.tanh(torch.relu(a + b) @ b))

    inputs = (torch.randn(64, 64), torch.randn(64, 64))
    recipe = build_recipe(captured(model, *inputs), tessiq.gpu, use_cache=False)

    assert len(recipe.kernels) == 2, recipe.kernels
    assert torch.allclose(tessiq.execute(recipe)(*inputs)[0], model(*inputs), atol=1e-4)


def test_repeated_calls_give_the_same_answer() -> None:
    # A launch is asynchronous and the transfers that read it are on another stream. If the
    # ordering between them were wrong, this is where it would show up.
    inputs = (torch.randn(65536), torch.randn(65536), torch.randn(65536))
    run = tessiq.execute(build_recipe(captured(chain, *inputs), tessiq.gpu, use_cache=False))

    first = run(*inputs)[0]
    for _ in range(50):
        assert torch.equal(run(*inputs)[0], first)


def test_training_through_a_kernel_matches_eager() -> None:
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(16, 32), torch.nn.Tanh(), torch.nn.Sigmoid())
    reference = copy.deepcopy(model)
    inputs = torch.randn(8, 16)

    torch.compile(model, backend="tessiq", dynamic=False, options={"device": str(tessiq.gpu)})(
        inputs
    ).sum().backward()
    reference(inputs).sum().backward()

    for got, want in zip(model.parameters(), reference.parameters(), strict=True):
        assert got.grad is not None
        assert want.grad is not None
        assert torch.allclose(got.grad, want.grad, atol=1e-4)


def test_the_host_reference_compiles_no_kernels() -> None:
    inputs = (torch.randn(256), torch.randn(256), torch.randn(256))
    recipe = build_recipe(captured(chain, *inputs), tessiq.cpu, use_cache=False)
    assert recipe.kernels == {}
    assert recipe.fused_away == frozenset()
