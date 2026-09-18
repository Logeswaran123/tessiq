# Copyright 2026 The Tessiq Authors.
# SPDX-License-Identifier: Apache-2.0
"""The ``torch.compile`` backend.

The entry point is the contract: `backend="tessiq"` has to resolve without the caller importing
tessiq, because that is the only thing that makes the string usable.
"""

from __future__ import annotations

import copy
from importlib.metadata import entry_points

import pytest
import torch

import tessiq
from tessiq.backend import compile_graph


@pytest.fixture(autouse=True)
def _fresh_dynamo() -> None:
    # Dynamo caches compiled code against the code object it traced. Without this a test reuses
    # what an earlier test compiled and never exercises the backend.
    torch._dynamo.reset()


def traced() -> torch.fx.GraphModule:
    return torch.fx.symbolic_trace(torch.nn.Tanh())


def test_the_entry_point_is_declared() -> None:
    declared = {e.name: e.value for e in entry_points(group="torch_dynamo_backends")}
    assert declared.get("tessiq") == "tessiq.backend:compile_graph"


def test_the_name_resolves() -> None:
    assert torch._dynamo.lookup_backend("tessiq") is compile_graph


def test_a_compiled_function_matches_eager() -> None:
    def model(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.relu(a + b) * b

    a, b = torch.randn(8), torch.randn(8)
    compiled = torch.compile(model, backend="tessiq", dynamic=False)
    assert torch.allclose(compiled(a, b), model(a, b), atol=1e-5)


def test_a_compiled_module_matches_eager() -> None:
    model = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.Tanh(), torch.nn.Linear(8, 2))
    inputs = torch.randn(3, 4)
    compiled = torch.compile(model, backend="tessiq", dynamic=False)
    assert torch.allclose(compiled(inputs), model(inputs), atol=1e-5)


def test_results_come_back_on_the_host() -> None:
    compiled = torch.compile(lambda a: torch.relu(a), backend="tessiq", dynamic=False)
    assert compiled(torch.randn(8)).device == torch.device("cpu")


def test_clusters_run_on_the_tessiq_device() -> None:
    compiled = torch.compile(lambda a, b: torch.relu(a + b) * b, backend="tessiq", dynamic=False)
    before = tessiq.memory_stats(tessiq.cpu).allocation_count
    compiled(torch.randn(8), torch.randn(8))
    assert tessiq.memory_stats(tessiq.cpu).allocation_count > before


def test_an_op_outside_the_support_table_is_decomposed_into_ones_inside_it() -> None:
    # silu is not in the support table. AOTAutograd rewrites it as x * sigmoid(x), and both of
    # those are, so the model clusters anyway. Without the decompositions it would run on the host.
    compiled = torch.compile(torch.nn.functional.silu, backend="tessiq", dynamic=False)
    inputs = torch.randn(8)

    before = tessiq.memory_stats(tessiq.cpu).allocation_count
    produced = compiled(inputs)
    assert tessiq.memory_stats(tessiq.cpu).allocation_count > before
    assert torch.allclose(produced, torch.nn.functional.silu(inputs), atol=1e-5)


def test_backward_matches_eager() -> None:
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.Tanh())
    reference = copy.deepcopy(model)
    inputs = torch.randn(3, 4)

    torch.compile(model, backend="tessiq", dynamic=False)(inputs).sum().backward()
    reference(inputs).sum().backward()

    for got, want in zip(model.parameters(), reference.parameters(), strict=True):
        assert got.grad is not None
        assert want.grad is not None
        assert torch.allclose(got.grad, want.grad, atol=1e-5)


def test_recompiling_for_a_new_shape_still_matches_eager() -> None:
    def model(a: torch.Tensor) -> torch.Tensor:
        return torch.relu(a) + 1.0

    compiled = torch.compile(model, backend="tessiq", dynamic=False)
    for size in (4, 16, 64):
        inputs = torch.randn(size)
        assert torch.allclose(compiled(inputs), model(inputs), atol=1e-5)


def test_a_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="no compilation modes"):
        compile_graph(traced(), [], mode="max-autotune")


def test_an_unknown_option_is_rejected_by_name() -> None:
    with pytest.raises(ValueError, match=r"stream.*supported: device"):
        compile_graph(traced(), [], options={"stream": 1})


def test_an_option_naming_another_backend_s_device_is_rejected() -> None:
    with pytest.raises(ValueError, match="expected a tessiq device"):
        compile_graph(traced(), [], options={"device": "cpu"})


@pytest.mark.cuda
def test_the_device_option_selects_a_gpu() -> None:
    if tessiq.gpu_count() == 0:
        pytest.skip("needs a gpu")

    compiled = torch.compile(
        lambda a, b: torch.relu(a + b),
        backend="tessiq",
        dynamic=False,
        options={"device": str(tessiq.gpu)},
    )
    a, b = torch.randn(16), torch.randn(16)

    before = tessiq.memory_stats(tessiq.gpu).allocation_count
    produced = compiled(a, b)
    assert tessiq.memory_stats(tessiq.gpu).allocation_count > before
    assert torch.allclose(produced, torch.relu(a + b), atol=1e-5)
