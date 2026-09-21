# Copyright 2026 The Tessiq Authors.
# SPDX-License-Identifier: Apache-2.0
"""Compiling a graph into a recipe.

A recipe decides everything that does not depend on the tensors. These tests check that what it
decided is right, and in particular that the output spec describes the graph it came from, since
the autograd engine matches results to inputs by position and a spec that is off by one puts a
gradient on the wrong parameter.
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
from tessiq import build_recipe, execute
from tessiq.recipe import HOST


def forward_graph(fn: Callable[..., Any], *inputs: torch.Tensor) -> torch.fx.GraphModule:
    captured: list[torch.fx.GraphModule] = []

    def backend(graph_module: torch.fx.GraphModule, _inputs: object) -> Callable[..., object]:
        captured.append(graph_module)
        return graph_module.forward

    torch.compile(
        fn,
        backend=aot_autograd(fw_compiler=backend, decompositions=core_aten_decompositions()),
        dynamic=False,
    )(*inputs)
    return captured[0]


def backward_graph(model: torch.nn.Module, inputs: torch.Tensor) -> torch.fx.GraphModule:
    captured: list[torch.fx.GraphModule] = []

    def forward(graph_module: torch.fx.GraphModule, _inputs: object) -> Callable[..., object]:
        return graph_module.forward

    def backward(graph_module: torch.fx.GraphModule, _inputs: object) -> Callable[..., object]:
        captured.append(graph_module)
        return graph_module.forward

    torch.compile(
        model, backend=aot_autograd(fw_compiler=forward, bw_compiler=backward), dynamic=False
    )(inputs).sum().backward()
    assert captured, "no backward graph was compiled"
    return captured[0]


def test_a_recipe_reports_what_the_arena_would_cost() -> None:
    graph = forward_graph(
        lambda a, b: torch.tanh(torch.relu(a + b) * b), torch.randn(1024), torch.randn(1024)
    )
    recipe = build_recipe(graph)
    assert recipe.arena_bytes > 0
    # Reuse is the point, so the arena has to be smaller than holding everything at once.
    assert recipe.arena_bytes < recipe.naive_bytes


def test_only_the_edges_of_a_cluster_need_a_move() -> None:
    # Four elementwise operations in a row all land in one cluster. The first reads the graph's
    # inputs, which arrive on the host, so it moves. The rest find their operands where the
    # previous step left them.
    graph = forward_graph(
        lambda a, b: torch.tanh(torch.sigmoid(torch.relu(a + b))), torch.randn(8), torch.randn(8)
    )
    recipe = build_recipe(graph)

    clustered = [name for name, device in recipe.devices.items() if device != torch.device("cpu")]
    assert len(clustered) > 1, "expected the chain to cluster"
    assert len(recipe.needs_move) < len(clustered), (
        f"every clustered node moved: {sorted(recipe.needs_move)}"
    )


def test_the_output_spec_matches_the_graph() -> None:
    graph = forward_graph(lambda a: (torch.relu(a), torch.tanh(a)), torch.randn(8))
    recipe = build_recipe(graph)
    assert len(recipe.output_spec) == 2
    assert all(name is not None for name in recipe.output_spec)


def test_the_output_spec_records_where_a_backward_graph_returns_nothing() -> None:
    # A backward graph returns one value per forward input: a gradient, or None where that input
    # takes no gradient. The IR cannot record those positions, because a position that returns
    # nothing computes nothing, so the spec is read off FX instead.
    model = torch.nn.Linear(4, 3)
    model.bias.requires_grad_(False)
    recipe = build_recipe(backward_graph(model, torch.randn(2, 4)))

    assert None in recipe.output_spec, f"expected a None position, got {recipe.output_spec}"
    assert any(name is not None for name in recipe.output_spec)


def test_a_recipe_runs_and_matches_eager() -> None:
    def model(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.tanh(torch.relu(a + b) @ b)

    a, b = torch.randn(4, 4), torch.randn(4, 4)
    produced = execute(build_recipe(forward_graph(model, a, b)))(a, b)
    assert torch.allclose(produced[0], model(a, b), atol=1e-5)


def test_a_recipe_is_reusable() -> None:
    a, b = torch.randn(8), torch.randn(8)
    recipe = build_recipe(forward_graph(lambda x, y: torch.relu(x + y), a, b))
    run = execute(recipe)
    assert torch.equal(run(a, b)[0], run(a, b)[0])


def test_training_through_a_recipe_matches_eager() -> None:
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.Tanh())
    reference = copy.deepcopy(model)
    inputs = torch.randn(3, 4)

    torch._dynamo.reset()
    torch.compile(model, backend="tessiq", dynamic=False)(inputs).sum().backward()
    reference(inputs).sum().backward()

    for got, want in zip(model.parameters(), reference.parameters(), strict=True):
        assert got.grad is not None
        assert want.grad is not None
        assert torch.allclose(got.grad, want.grad, atol=1e-5)


def test_a_spec_of_the_wrong_length_is_caught() -> None:
    # The spec is rebuilt into the returned tuple on every call, so a spec that no longer
    # describes the graph fails here rather than handing the caller shifted positions.
    graph = forward_graph(lambda a: torch.relu(a), torch.randn(8))
    good = build_recipe(graph)
    # replace rather than a fresh Recipe, so a new field does not break this test instead of the
    # thing it is testing.
    broken = replace(good, output_spec=(*good.output_spec, None))
    with pytest.raises(ValueError, match=r"argument 2 is shorter|zip"):
        execute(broken)(torch.randn(8))


def test_an_unclustered_graph_still_runs() -> None:
    a, b = torch.randn(4, 4), torch.randn(4, 4)
    recipe = build_recipe(forward_graph(lambda x, y: x @ y, a, b))
    assert all(device == torch.device("cpu") for device in recipe.devices.values())
    assert torch.allclose(execute(recipe)(a, b)[0], a @ b, atol=1e-5)


def test_host_is_the_sentinel_the_native_side_uses() -> None:
    assert HOST == -1


@pytest.mark.cuda
def test_a_recipe_can_target_the_gpu() -> None:
    if tessiq.gpu_count() == 0:
        pytest.skip("needs a gpu")

    a, b = torch.randn(16), torch.randn(16)
    recipe = build_recipe(forward_graph(lambda x, y: torch.relu(x + y), a, b), tessiq.gpu)

    before = tessiq.memory_stats(tessiq.gpu).allocation_count
    produced = execute(recipe)(a, b)
    assert tessiq.memory_stats(tessiq.gpu).allocation_count > before
    assert torch.allclose(produced[0], torch.relu(a + b), atol=1e-5)
