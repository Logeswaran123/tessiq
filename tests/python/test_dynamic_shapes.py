# Copyright 2026 The Tessiq Authors.
# SPDX-License-Identifier: Apache-2.0
"""Graphs whose shapes are not known at compile time.

Dynamo captures concrete shapes the first time it sees a function and recompiles with symbolic
ones the second time a dimension changes. That is the default rather than something a caller opts
into, so a model called with two sequence lengths reaches the backend with a symbolic graph
without anyone asking for one.

Nothing here compiles such a graph. What these check is that it is declined and runs, rather than
raising, and that declining it has not made the ordinary case decline too.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch
from torch import nn
from torch._dynamo.backends.common import aot_autograd

import tessiq
from tessiq.decompositions import decompositions
from tessiq.shapes import is_static


def model() -> nn.Module:
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(32, 32), nn.Tanh()).eval()


def captured(
    fn: Callable[..., object], example: torch.Tensor, *, dynamic: bool
) -> torch.fx.GraphModule:
    """The graph the backend receives, which is after AOTAutograd rather than before it.

    Symbolic shapes show up here. Dynamo's own graph still carries the concrete example, so
    checking that one would say every graph is static.
    """
    graphs: list[torch.fx.GraphModule] = []

    def collect(graph_module: torch.fx.GraphModule, _inputs: object) -> object:
        graphs.append(graph_module)
        return graph_module.forward

    torch._dynamo.reset()
    with torch.no_grad():
        torch.compile(
            fn,
            backend=aot_autograd(fw_compiler=collect, decompositions=decompositions()),
            dynamic=dynamic,
        )(example)
    return graphs[0]


@pytest.fixture(autouse=True)
def _fresh_dynamo() -> None:
    torch._dynamo.reset()


def test_a_concrete_graph_is_static() -> None:
    assert is_static(captured(model(), torch.randn(2, 6, 32), dynamic=False))


def test_a_symbolic_graph_is_not() -> None:
    # Its shapes are expressions over dimensions that arrive as their own graph inputs.
    assert not is_static(captured(model(), torch.randn(2, 6, 32), dynamic=True))


def test_a_changing_shape_does_not_raise() -> None:
    # The whole point. Before this, the second distinct shape crashed the backend under torch's
    # own default settings, with no `dynamic` argument anywhere.
    net = model()
    compiled = torch.compile(net, backend="tessiq", options={"device": str(tessiq.cpu)})

    with torch.no_grad():
        for steps in (6, 7, 8, 13, 21):
            example = torch.randn(2, steps, 32)
            torch.testing.assert_close(compiled(example), net(example), rtol=1e-5, atol=1e-5)


def test_asking_for_dynamic_shapes_works_too() -> None:
    net = model()
    compiled = torch.compile(
        net, backend="tessiq", dynamic=True, options={"device": str(tessiq.cpu)}
    )
    with torch.no_grad():
        example = torch.randn(2, 6, 32)
        torch.testing.assert_close(compiled(example), net(example), rtol=1e-5, atol=1e-5)


def test_a_static_graph_is_still_compiled() -> None:
    # Declining symbolic graphs would be a poor trade if it declined the ordinary ones as well,
    # and the difference is invisible from the outside: both give the right answer.
    from tessiq.recipe import build_recipe

    graph = captured(model(), torch.randn(2, 6, 32), dynamic=False)
    recipe = build_recipe(graph, tessiq.cpu, use_cache=False)
    assert recipe.devices, "a static graph should still reach the compiler"


@pytest.mark.cuda
def test_a_changing_shape_still_uses_kernels_for_the_first_shape() -> None:
    # The first shape is concrete, so it compiles and runs on the device. Only the shapes after a
    # change are declined.
    if tessiq.gpu_count() == 0:
        pytest.skip("needs a gpu")

    from tessiq.recipe import build_recipe

    graph = captured(model(), torch.randn(2, 6, 32), dynamic=False)
    recipe = build_recipe(graph, tessiq.gpu, use_cache=False)
    assert recipe.kernels, "the concrete graph should compile to kernels"
