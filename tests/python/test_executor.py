# Copyright 2026 The Tessiq Authors.
# SPDX-License-Identifier: Apache-2.0
"""Running a recipe.

Every test checks the result against eager: wherever an operation runs, it has to produce the
number PyTorch would.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
import torch
from torch._dynamo.backends.common import aot_autograd

import tessiq
from tessiq import build_recipe, execute
from tessiq.executor import ResidentInputs


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
    # The tests above would also pass if the executor ignored the partition and ran everything on
    # the host.
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


# --- inputs that stay on the device --------------------------------------------------------------


def test_an_unchanged_input_is_moved_once() -> None:
    # AOTAutograd lifts a module's parameters to graph inputs, and they are the same tensors on
    # every call.
    table = ResidentInputs()
    host = torch.randn(64)

    first = table.move(host, tessiq.cpu)
    second = table.move(host, tessiq.cpu)

    assert first is second
    assert table.stats() == (1, 1), "one reuse and one copy"


def test_writing_to_an_input_copies_it_again() -> None:
    # A parameter an optimizer step writes to has to cross again; the version counter notices.
    table = ResidentInputs()
    host = torch.zeros(64)

    before = table.move(host, tessiq.cpu)
    host.add_(1.0)
    after = table.move(host, tessiq.cpu)

    assert after is not before
    assert after.sum().item() == pytest.approx(64.0)


def test_a_different_tensor_at_the_same_address_is_not_reused() -> None:
    # A freed tensor's id can be reused, so the weak reference makes a dead entry miss rather than
    # match the wrong tensor.
    table = ResidentInputs()
    for _ in range(64):
        table.move(torch.randn(16), tessiq.cpu)

    # Every one of those is dead, so nothing accumulated.
    hits, misses = table.stats()
    assert hits == 0, "no transient tensor should have been reused"
    assert misses == 64


def test_the_table_does_not_grow_with_the_calls() -> None:
    table = ResidentInputs()
    weights = torch.randn(64)
    for _ in range(50):
        table.move(weights, tessiq.cpu)
        table.move(torch.randn(64), tessiq.cpu)

    hits, _ = table.stats()
    assert hits == 49, "the weight is reused every call after the first"
    assert len(table._entries) <= 2, f"one live weight and one live input: {len(table._entries)}"
