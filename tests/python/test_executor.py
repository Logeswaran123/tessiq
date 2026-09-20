# Copyright 2026 The Tessiq Authors.
# SPDX-License-Identifier: Apache-2.0
"""Running a recipe.

Every test checks the result against eager. That is the only property that matters here: the
compiler may route an operation wherever it likes, but the number it produces has to be the number
PyTorch would have produced.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
import torch
from torch._dynamo.backends.common import aot_autograd

import tessiq
from tessiq import build_recipe, execute


def captured_graph(fn: Callable[..., Any], *inputs: torch.Tensor) -> torch.fx.GraphModule:
    """The graph a backend receives for `fn`, after AOTAutograd."""
    captured: list[torch.fx.GraphModule] = []

    def backend(graph_module: torch.fx.GraphModule, _inputs: object) -> Callable[..., object]:
        captured.append(graph_module)
        return graph_module.forward

    torch.compile(fn, backend=aot_autograd(fw_compiler=backend), dynamic=False)(*inputs)
    return captured[0]


def compiled(fn: Callable[..., Any], *inputs: torch.Tensor) -> Callable[..., Any]:
    """Capture, compile to a recipe, and return something that runs it."""
    return execute(build_recipe(captured_graph(fn, *inputs)))


def assert_matches_eager(fn: Callable[..., Any], *inputs: torch.Tensor) -> None:
    run = compiled(fn, *inputs)
    produced = run(*inputs)
    expected = fn(*inputs)

    if isinstance(expected, torch.Tensor):
        expected = (expected,)
    assert len(produced) == len(expected), f"{len(produced)} results, expected {len(expected)}"
    for got, want in zip(produced, expected, strict=True):
        assert torch.allclose(got, want, atol=1e-5), f"\ngot      {got}\nexpected {want}"


def test_an_all_supported_model() -> None:
    assert_matches_eager(lambda a, b: torch.relu(a + b), torch.randn(8), torch.randn(8))


def test_a_model_split_by_an_unsupported_op() -> None:
    def model(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.tanh(torch.relu(a + b) @ b)

    assert_matches_eager(model, torch.randn(4, 4), torch.randn(4, 4))


def test_a_model_with_no_supported_ops() -> None:
    assert_matches_eager(lambda a, b: a @ b, torch.randn(4, 4), torch.randn(4, 4))


def test_a_scalar_argument_survives_the_move() -> None:
    # 2.0 is not a tensor, so moving arguments to a device has to leave it alone.
    assert_matches_eager(lambda a: torch.relu(a * 2.0), torch.randn(8))


def test_several_results() -> None:
    assert_matches_eager(lambda a: (torch.relu(a), torch.tanh(a)), torch.randn(8))


def test_a_value_used_by_both_sides_of_a_split() -> None:
    def model(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        shared = torch.relu(a + b)
        return torch.tanh(shared) + (shared @ b)

    assert_matches_eager(model, torch.randn(4, 4), torch.randn(4, 4))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_dtypes_are_preserved(dtype: torch.dtype) -> None:
    inputs = (torch.randn(8, dtype=dtype), torch.randn(8, dtype=dtype))
    run = compiled(lambda a, b: torch.relu(a + b), *inputs)
    assert run(*inputs)[0].dtype == dtype


def test_results_come_back_to_the_host() -> None:
    # Whatever device the last node ran on, the caller gets what eager would have given them.
    run = compiled(lambda a, b: torch.relu(a + b), torch.randn(4), torch.randn(4))
    produced = run(torch.randn(4), torch.randn(4))
    assert produced[0].device == torch.device("cpu")


def test_clustered_nodes_really_run_on_the_tessiq_device() -> None:
    # Without this the tests above would pass just as well if the executor ignored the partition
    # and ran everything on the host.
    inputs = (torch.randn(64), torch.randn(64))
    run = compiled(lambda a, b: torch.relu(a + b), *inputs)

    before = tessiq.memory_stats(tessiq.cpu).allocation_count
    run(*inputs)
    assert tessiq.memory_stats(tessiq.cpu).allocation_count > before


def test_running_twice_gives_the_same_answer() -> None:
    inputs = (torch.randn(8), torch.randn(8))
    run = compiled(lambda a, b: torch.relu(a + b), *inputs)
    assert torch.equal(run(*inputs)[0], run(*inputs)[0])


@pytest.mark.cuda
def test_clusters_can_run_on_the_gpu() -> None:
    if tessiq.gpu_count() == 0:
        pytest.skip("needs a gpu")

    inputs = (torch.randn(16), torch.randn(16))
    graph = captured_graph(lambda a, b: torch.relu(a + b), *inputs)

    before = tessiq.memory_stats(tessiq.gpu).allocation_count
    produced = execute(build_recipe(graph, tessiq.gpu))(*inputs)
    assert tessiq.memory_stats(tessiq.gpu).allocation_count > before
    assert torch.allclose(produced[0], torch.relu(inputs[0] + inputs[1]), atol=1e-5)
