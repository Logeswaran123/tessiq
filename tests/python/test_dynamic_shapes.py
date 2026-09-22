# Copyright 2026 The Tessiq Authors.
# SPDX-License-Identifier: Apache-2.0
"""Graphs whose shapes are not known at compile time.

Dynamo captures concrete shapes the first time it sees a function and recompiles with symbolic
ones the second time a dimension changes. That is the default rather than something a caller opts
into, so a model called with two sequence lengths reaches the backend with a symbolic graph
without anyone asking for one.

The compiler takes such a graph. These check that it gives the same answers across shapes, that
the elementwise work in it reaches the device rather than being interpreted, and that the parts
whose extents are still baked in decline instead of guessing.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch
from torch import nn
from torch._dynamo.backends.common import aot_autograd

import tessiq
from tessiq import _C
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
    # Taking symbolic graphs would be a poor trade if it changed what happens to the ordinary
    # ones, and the difference is invisible from the outside: both give the right answer.
    from tessiq.recipe import build_recipe

    graph = captured(model(), torch.randn(2, 6, 32), dynamic=False)
    recipe = build_recipe(graph, tessiq.cpu, use_cache=False)
    assert recipe.devices, "a static graph should still reach the compiler"


def test_an_unknown_extent_reaches_the_ir() -> None:
    # `?` is how the IR spells an extent it does not know. Everything downstream reads it from
    # the type rather than from a table kept beside the graph.
    graph = captured(model(), torch.randn(2, 6, 32), dynamic=True)
    text = _C.import_fx(graph, "forward")
    assert "f32[?,?,32]" in text, text
    assert "%tanh" in text, text


def test_the_extents_a_reshape_computes_are_operands() -> None:
    # A reshape in a symbolic graph takes its sizes as values the graph computes, not as
    # constants. They are operands in the IR, and the argument is marked rather than invented.
    graph = captured(model(), torch.randn(2, 6, 32), dynamic=True)
    text = _C.import_fx(graph, "forward")
    assert 'aten.view.default(%arg4_1, %mul_6) {arg1 = "dynamic"}' in text, text
    assert "operator.mul(" in text, text


@pytest.mark.cuda
def test_a_symbolic_graph_compiles_kernels() -> None:
    if tessiq.gpu_count() == 0:
        pytest.skip("needs a gpu")

    from tessiq.recipe import build_recipe

    graph = captured(model(), torch.randn(2, 6, 32), dynamic=True)
    recipe = build_recipe(graph, tessiq.gpu, use_cache=False)
    assert recipe.kernels, "a symbolic graph should still compile to kernels"
    # Only fusion groups, and every one of them takes its count at the launch. A reduction or a
    # matmul here would mean something claimed a shape it cannot know.
    groups = [k for k in recipe.kernels.values() if isinstance(k, _C.FusedKernel)]
    assert len(groups) == len(recipe.kernels)
    assert all(group.has_dynamic_extent for group in groups)


@pytest.mark.cuda
def test_one_compile_serves_every_shape() -> None:
    # The point of the whole thing. One kernel, launched with the count read off each call's
    # output, gives the same answers as eager across shapes it was never compiled for.
    if tessiq.gpu_count() == 0:
        pytest.skip("needs a gpu")

    net = model()
    compiled = torch.compile(net, backend="tessiq", options={"device": str(tessiq.gpu)})
    with torch.no_grad():
        for batch, steps in ((2, 6), (3, 11), (5, 4), (1, 2), (2, 6)):
            example = torch.randn(batch, steps, 32)
            torch.testing.assert_close(compiled(example), net(example), rtol=1e-5, atol=1e-5)


@pytest.mark.cuda
def test_the_elementwise_work_runs_on_the_device() -> None:
    # Answers that match eager prove nothing on their own: running the whole graph uncompiled
    # matches too. What this checks is that the launches happened.
    if tessiq.gpu_count() == 0:
        pytest.skip("needs a gpu")

    from tessiq import profiler

    net = model()
    compiled = torch.compile(net, backend="tessiq", options={"device": str(tessiq.gpu)})
    with torch.no_grad():
        compiled(torch.randn(2, 6, 32))
        with profiler.profile() as collected:
            compiled(torch.randn(7, 3, 32))

    launched = [event for event in collected.events if event.category == "kernel"]
    assert launched, "the second shape should still have launched a kernel"


@pytest.mark.cuda
def test_a_zero_extent_has_nothing_to_launch() -> None:
    # A dimension that turns out to be zero. The static path never sees this, because a group of
    # no elements is not compiled at all, so the dynamic launch is the only place it can arrive.
    if tessiq.gpu_count() == 0:
        pytest.skip("needs a gpu")

    net = model()
    compiled = torch.compile(net, backend="tessiq", options={"device": str(tessiq.gpu)})
    with torch.no_grad():
        compiled(torch.randn(2, 6, 32))
        empty = torch.randn(0, 5, 32)
        torch.testing.assert_close(compiled(empty), net(empty), rtol=1e-5, atol=1e-5)


@pytest.mark.cuda
def test_a_mismatched_input_is_caught_at_the_launch() -> None:
    # The IR spells every unknown extent as the same `?`, so it cannot prove that two of them are
    # the same number. This check is what makes that safe: a kernel handed buffers of different
    # sizes refuses rather than reading past the shorter one.
    if tessiq.gpu_count() == 0:
        pytest.skip("needs a gpu")

    from tessiq.recipe import build_recipe

    graph = captured(model(), torch.randn(2, 6, 32), dynamic=True)
    kernel = next(iter(build_recipe(graph, tessiq.gpu, use_cache=False).kernels.values()))

    out = torch.empty(12, 32, device=str(tessiq.gpu))
    short = torch.empty(11, 32, device=str(tessiq.gpu))
    with pytest.raises(ValueError, match="elements for a result of 384"):
        kernel.launch([short] * len(kernel.inputs), out)


@pytest.mark.cuda
def test_what_a_symbolic_graph_does_not_claim_yet() -> None:
    # A record of where the line is, so that moving it is a deliberate change to this number
    # rather than something noticed later. The matmuls, the reductions and the fusions with a
    # broadcast in them all need an extent written into the kernel.
    if tessiq.gpu_count() == 0:
        pytest.skip("needs a gpu")

    from tessiq.recipe import build_recipe

    example = torch.randn(2, 6, 32)
    static = build_recipe(captured(model(), example, dynamic=False), tessiq.gpu, use_cache=False)
    symbolic = build_recipe(captured(model(), example, dynamic=True), tessiq.gpu, use_cache=False)
    assert 0 < len(symbolic.kernels) < len(static.kernels)
