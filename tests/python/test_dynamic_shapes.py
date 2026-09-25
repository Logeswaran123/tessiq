# Copyright 2026 The Tessiq Authors.
# SPDX-License-Identifier: Apache-2.0
"""Graphs whose shapes are not known at compile time.

By default Dynamo captures concrete shapes the first time it sees a function and recompiles with
symbolic ones the second time a dimension changes.

These check that such a graph gives the same answers across shapes, that its work reaches the
device, and that what needs its extents at compile time declines instead of guessing.
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
    fn: Callable[..., object], *examples: torch.Tensor, dynamic: bool
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
        )(*examples)
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
    # The second distinct shape gives a symbolic graph under torch's defaults, with no `dynamic`
    # argument anywhere.
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
    # Supporting symbolic graphs must not change what happens to static ones.
    from tessiq.recipe import build_recipe

    graph = captured(model(), torch.randn(2, 6, 32), dynamic=False)
    recipe = build_recipe(graph, tessiq.cpu, use_cache=False)
    assert recipe.devices, "a static graph should still reach the compiler"


def test_an_unknown_extent_reaches_the_ir() -> None:
    # `?` is how the IR spells an unknown extent.
    graph = captured(model(), torch.randn(2, 6, 32), dynamic=True)
    text = _C.import_fx(graph, "forward")
    assert "f32[?,?,32]" in text, text
    assert "%tanh" in text, text


def test_the_extents_a_reshape_computes_are_operands() -> None:
    # A reshape in a symbolic graph takes its sizes as computed values. They become operands, and
    # the argument is marked as computed.
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
    # Every kernel takes its extents at launch, since the traced shape is only one of many.
    assert all(kernel.has_dynamic_extent for kernel in recipe.kernels.values())


@pytest.mark.cuda
def test_one_compile_serves_every_shape() -> None:
    # One kernel, launched with the count read off each call's output, matches eager across
    # shapes it was not compiled for.
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
    # Matching eager proves nothing on its own, since running uncompiled matches too. This checks
    # that the launches happened.
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
    # A dimension that turns out to be zero, which only a dynamic launch can meet.
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
    # The IR cannot prove two `?` extents equal, so a kernel handed buffers of different sizes
    # refuses rather than reading past the shorter one.
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
def test_a_symbolic_graph_claims_what_a_concrete_one_does() -> None:
    # Unknown shapes cost nothing in coverage: the same operations get kernels.
    if tessiq.gpu_count() == 0:
        pytest.skip("needs a gpu")

    from tessiq.recipe import build_recipe
    from test_models import EncoderBlock

    torch.manual_seed(0)
    net = EncoderBlock().eval()
    example = torch.randn(4, 9, 32)
    static = build_recipe(captured(net, example, dynamic=False), tessiq.gpu, use_cache=False)
    symbolic = build_recipe(captured(net, example, dynamic=True), tessiq.gpu, use_cache=False)
    assert len(symbolic.kernels) == len(static.kernels)
    assert {type(k).__name__ for k in symbolic.kernels.values()} == {
        type(k).__name__ for k in static.kernels.values()
    }


@pytest.mark.cuda
def test_a_softmax_gets_its_row_length_at_the_launch() -> None:
    # A softmax's rows are as long as the sequence, which changes, so the kernel takes the row
    # length as an argument.
    if tessiq.gpu_count() == 0:
        pytest.skip("needs a gpu")

    from test_models import EncoderBlock

    torch.manual_seed(0)
    net = EncoderBlock().eval()
    compiled = torch.compile(net, backend="tessiq", options={"device": str(tessiq.gpu)})
    with torch.no_grad():
        for steps in (6, 13, 4, 29):
            example = torch.randn(3, steps, 32)
            torch.testing.assert_close(compiled(example), net(example), rtol=1e-4, atol=1e-4)


@pytest.mark.cuda
def test_the_work_stays_on_the_device_between_kernels() -> None:
    # A symbolic graph must cross between host and device no more often than a static one.
    if tessiq.gpu_count() == 0:
        pytest.skip("needs a gpu")

    from tessiq import profiler
    from test_models import EncoderBlock

    torch.manual_seed(0)
    net = EncoderBlock().eval()
    crossings = {}
    for label, dynamic in (("symbolic", True), ("concrete", False)):
        torch._dynamo.reset()
        compiled = torch.compile(
            net, backend="tessiq", dynamic=dynamic, options={"device": str(tessiq.gpu)}
        )
        with torch.no_grad():
            compiled(torch.randn(4, 16, 32))
            with profiler.profile() as collected:
                compiled(torch.randn(4, 16, 32))
        crossings[label] = sum(1 for e in collected.events if e.category == "transfer")

    assert crossings["symbolic"] == crossings["concrete"], crossings


def add_relu(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.relu(a + b)


#: Groups whose first input broadcasts, so it does not have the result's shape. 64 and 96 rows are
#: multiples of the 32 columns, which is when a result sized from the first input got past the
#: launch's checks; 7 is not.
BROADCAST_FIRST = [
    pytest.param(lambda rows: ((rows, 1), (1, 32)), id="outer"),
    pytest.param(lambda rows: ((rows, 1), (rows, 32)), id="row"),
    pytest.param(lambda rows: ((32,), (rows, 32)), id="bias"),
]


@pytest.mark.cuda
@pytest.mark.parametrize("shapes", BROADCAST_FIRST)
def test_a_group_whose_first_input_broadcasts_is_sized_from_all_of_them(
    shapes: Callable[[int], tuple[tuple[int, ...], tuple[int, ...]]],
) -> None:
    if tessiq.gpu_count() == 0:
        pytest.skip("needs a gpu")

    from tessiq import profiler

    compiled = torch.compile(
        add_relu, backend="tessiq", dynamic=True, options={"device": str(tessiq.gpu)}
    )
    for rows in (64, 96, 7):
        first, second = shapes(rows)
        a, b = torch.randn(*first), torch.randn(*second)
        with profiler.profile() as collected:
            out = compiled(a, b)
        assert any(e.category == "kernel" for e in collected.events), "no kernel ran"
        torch.testing.assert_close(out, add_relu(a, b), rtol=1e-6, atol=1e-6)


@pytest.mark.cuda
def test_a_dynamic_kernel_refuses_an_output_of_the_wrong_shape() -> None:
    # A dynamic kernel takes its element count from the output, so the output's shape is checked
    # against the inputs'. [64,1] has an element count both inputs divide.
    if tessiq.gpu_count() == 0:
        pytest.skip("needs a gpu")

    from tessiq.recipe import build_recipe

    graph = captured(add_relu, torch.randn(64, 1), torch.randn(1, 32), dynamic=True)
    kernel = next(iter(build_recipe(graph, tessiq.gpu, use_cache=False).kernels.values()))
    assert kernel.has_dynamic_extent

    a = torch.randn(64, 1, device=tessiq.gpu)
    b = torch.randn(1, 32, device=tessiq.gpu)
    assert kernel.output_shape([a, b]) == [64, 32]
    with pytest.raises(ValueError, match=r"shape \[64,1\]; its inputs broadcast to \[64,32\]"):
        kernel.launch([a, b], torch.empty(64, 1, device=tessiq.gpu))
