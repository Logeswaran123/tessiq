# Copyright 2026 The Tessiq Authors.
# SPDX-License-Identifier: Apache-2.0
"""Training through the backend.

`torch.compile` on a model that requires gradients produces a forward and a backward graph, and
AOTAutograd hands both to the same compiler. These check that a model trained through tessiq
reaches the same parameters as the same model trained eagerly.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
import torch
from torch import nn
from torch._dynamo.backends.common import aot_autograd

import tessiq
from tessiq.decompositions import decompositions
from tessiq.recipe import build_recipe


def model() -> nn.Module:
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(16, 32), nn.Tanh(), nn.Linear(32, 1))


def dataset() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(1)
    inputs = torch.randn(64, 16)
    return inputs, inputs[:, :1] * 2.0 + 0.5


def train(
    compile_with: Callable[[nn.Module], Any] | None, steps: int = 200
) -> tuple[float, float, nn.Module]:
    """Train for `steps` and return the first loss, the last loss, and the model."""
    net = model()
    inputs, targets = dataset()
    trained = net if compile_with is None else compile_with(net)

    optimiser = torch.optim.SGD(net.parameters(), lr=0.05)
    first = last = 0.0
    for step in range(steps):
        optimiser.zero_grad()
        loss = torch.nn.functional.mse_loss(trained(inputs), targets)
        loss.backward()  # type: ignore[no-untyped-call]
        optimiser.step()
        if step == 0:
            first = loss.item()
        last = loss.item()
    return first, last, net


@pytest.fixture(autouse=True)
def _fresh_dynamo() -> None:
    torch._dynamo.reset()


def on(device: torch.device) -> Callable[[nn.Module], Any]:
    def compile_it(net: nn.Module) -> Any:
        return torch.compile(net, backend="tessiq", dynamic=False, options={"device": str(device)})

    return compile_it


def test_a_model_trains_to_convergence() -> None:
    first, last, _ = train(on(tessiq.cpu))
    assert last < first / 100, f"loss went {first:.4f} to {last:.4f}"


@pytest.mark.cuda
def test_a_model_trains_to_convergence_on_a_gpu() -> None:
    if tessiq.gpu_count() == 0:
        pytest.skip("needs a gpu")
    first, last, _ = train(on(tessiq.gpu))
    assert last < first / 100, f"loss went {first:.4f} to {last:.4f}"


@pytest.mark.cuda
def test_training_reaches_the_same_parameters_as_eager() -> None:
    # Two hundred steps, and every weight ends where eager put it. A slightly wrong backward pass
    # still converges, to somewhere else.
    if tessiq.gpu_count() == 0:
        pytest.skip("needs a gpu")

    _, eager_loss, eager_net = train(None)
    torch._dynamo.reset()
    _, ours_loss, our_net = train(on(tessiq.gpu))

    assert ours_loss == pytest.approx(eager_loss, rel=1e-5)
    for theirs, ours in zip(eager_net.parameters(), our_net.parameters(), strict=True):
        torch.testing.assert_close(ours, theirs, rtol=1e-4, atol=1e-5)


class WithItsLoss(nn.Module):
    """The model and its loss in one module, so the compiled backward starts from a scalar."""

    def __init__(self) -> None:
        super().__init__()
        self.net = model()

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Written out rather than as mse_loss, whose backward is one operation that never reads the
        # broadcast gradient through a kernel. A mean written out divides it, and that does.
        loss: torch.Tensor = ((self.net(inputs) - targets) ** 2).mean()
        return loss


@pytest.mark.cuda
def test_a_module_that_computes_its_own_loss_trains_the_same_as_eager() -> None:
    # The backward graph begins by broadcasting the scalar gradient to every element, a value with
    # all-zero strides, which layout-assign records a copy of. The kernel has to name the original
    # value, since the copy's name is not a C identifier and the executor does not hold it.
    if tessiq.gpu_count() == 0:
        pytest.skip("needs a gpu")
    inputs, targets = dataset()

    def train(compiled: bool) -> nn.Module:
        net = WithItsLoss()
        step = (
            torch.compile(net, backend="tessiq", dynamic=False, options={"device": str(tessiq.gpu)})
            if compiled
            else net
        )
        optimiser = torch.optim.SGD(net.parameters(), lr=0.05)
        for _ in range(50):
            optimiser.zero_grad()
            step(inputs, targets).backward()
            optimiser.step()
        return net

    eager = train(compiled=False)
    torch._dynamo.reset()
    ours = train(compiled=True)
    for theirs, mine in zip(eager.parameters(), ours.parameters(), strict=True):
        torch.testing.assert_close(mine, theirs, rtol=1e-4, atol=1e-5)


@pytest.mark.cuda
def test_the_backward_graph_is_compiled_too() -> None:
    # The backward graph is the larger one here, so a backend that compiled only forwards would
    # leave most of training on the host without failing any correctness test.
    if tessiq.gpu_count() == 0:
        pytest.skip("needs a gpu")

    graphs: list[torch.fx.GraphModule] = []

    def collect(graph_module: torch.fx.GraphModule, _inputs: object) -> object:
        graphs.append(graph_module)
        return graph_module.forward

    net = torch.compile(
        model(),
        backend=aot_autograd(
            fw_compiler=collect, bw_compiler=collect, decompositions=decompositions()
        ),
        dynamic=False,
    )
    inputs, targets = dataset()
    torch.nn.functional.mse_loss(net(inputs), targets).backward()  # type: ignore[no-untyped-call]

    assert len(graphs) == 2, "a forward graph and a backward graph"
    forward, backward = graphs
    assert len(list(backward.graph.nodes)) > len(list(forward.graph.nodes)), "backward is larger"

    for graph in graphs:
        recipe = build_recipe(graph, tessiq.gpu, use_cache=False)
        assert recipe.kernels, f"nothing compiled for a graph of {len(recipe.devices)} steps"
