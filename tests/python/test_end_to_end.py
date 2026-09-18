# Copyright 2026 The Tessiq Authors.
# SPDX-License-Identifier: Apache-2.0
"""The whole path, on models rather than expressions.

The tests elsewhere take one seam each. These take a model, compile it, run it, and compare
against eager, which is the only claim a user of the backend actually cares about.
"""

from __future__ import annotations

import copy

import pytest
import torch

import tessiq


def mlp() -> torch.nn.Sequential:
    torch.manual_seed(0)
    return torch.nn.Sequential(
        torch.nn.Linear(16, 32),
        torch.nn.Tanh(),
        torch.nn.Linear(32, 32),
        torch.nn.Sigmoid(),
        torch.nn.Linear(32, 4),
    )


@pytest.fixture(autouse=True)
def _fresh_dynamo() -> None:
    torch._dynamo.reset()


def test_an_mlp_matches_eager() -> None:
    model = mlp()
    inputs = torch.randn(8, 16)
    compiled = torch.compile(model, backend="tessiq", dynamic=False)
    assert torch.allclose(compiled(inputs), model(inputs), atol=1e-5)


def test_an_mlp_runs_its_clusters_on_the_device() -> None:
    # The agreement above would hold just as well if the backend quietly ran everything on the
    # host, so the counters are what make the rest of this file mean something.
    compiled = torch.compile(mlp(), backend="tessiq", dynamic=False)
    before = tessiq.memory_stats(tessiq.cpu).allocation_count
    compiled(torch.randn(8, 16))
    assert tessiq.memory_stats(tessiq.cpu).allocation_count > before


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_dtypes_survive_the_round_trip(dtype: torch.dtype) -> None:
    model = mlp().to(dtype)
    inputs = torch.randn(8, 16, dtype=dtype)
    compiled = torch.compile(model, backend="tessiq", dynamic=False)

    produced = compiled(inputs)
    assert produced.dtype == dtype
    assert torch.allclose(produced, model(inputs), atol=1e-5)


def test_training_converges_to_the_same_weights() -> None:
    model = mlp()
    reference = copy.deepcopy(model)
    compiled = torch.compile(model, backend="tessiq", dynamic=False)

    optimiser = torch.optim.SGD(model.parameters(), lr=0.1)
    reference_optimiser = torch.optim.SGD(reference.parameters(), lr=0.1)

    torch.manual_seed(1)
    batches = [(torch.randn(8, 16), torch.randn(8, 4)) for _ in range(5)]

    # torch.autograd.backward rather than loss.backward(): Tensor.backward carries no annotation,
    # and mypy runs strict here.
    for inputs, target in batches:
        optimiser.zero_grad()
        torch.autograd.backward(torch.nn.functional.mse_loss(compiled(inputs), target))
        optimiser.step()

        reference_optimiser.zero_grad()
        torch.autograd.backward(torch.nn.functional.mse_loss(reference(inputs), target))
        reference_optimiser.step()

    for got, want in zip(model.parameters(), reference.parameters(), strict=True):
        assert torch.allclose(got, want, atol=1e-5)


def test_a_graph_break_still_matches_eager() -> None:
    # A data-dependent branch splits the function into two graphs. Each reaches the backend on its
    # own, and the pieces have to compose back into the same answer.
    def model(a: torch.Tensor) -> torch.Tensor:
        scaled = torch.tanh(a) * 2.0
        if scaled.sum() > 0:
            return torch.relu(scaled)
        return torch.sigmoid(scaled)

    compiled = torch.compile(model, backend="tessiq", dynamic=False)
    for _ in range(4):
        inputs = torch.randn(8)
        assert torch.allclose(compiled(inputs), model(inputs), atol=1e-5)


def test_a_model_returning_several_tensors_matches_eager() -> None:
    def model(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.relu(a + b), torch.tanh(a * b)

    a, b = torch.randn(8), torch.randn(8)
    compiled = torch.compile(model, backend="tessiq", dynamic=False)

    produced = compiled(a, b)
    expected = model(a, b)
    assert len(produced) == len(expected)
    for got, want in zip(produced, expected, strict=True):
        assert torch.allclose(got, want, atol=1e-5)


def test_compiling_twice_gives_one_answer() -> None:
    model = mlp()
    inputs = torch.randn(8, 16)
    first = torch.compile(model, backend="tessiq", dynamic=False)(inputs)
    torch._dynamo.reset()
    second = torch.compile(model, backend="tessiq", dynamic=False)(inputs)
    assert torch.equal(first, second)


@pytest.mark.cuda
def test_an_mlp_on_the_gpu_matches_eager() -> None:
    if tessiq.gpu_count() == 0:
        pytest.skip("needs a gpu")

    model = mlp()
    inputs = torch.randn(8, 16)
    compiled = torch.compile(
        model, backend="tessiq", dynamic=False, options={"device": str(tessiq.gpu)}
    )

    before = tessiq.memory_stats(tessiq.gpu).allocation_count
    produced = compiled(inputs)
    assert tessiq.memory_stats(tessiq.gpu).allocation_count > before
    assert torch.allclose(produced, model(inputs), atol=1e-5)
