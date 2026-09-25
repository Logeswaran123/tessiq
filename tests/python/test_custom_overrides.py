# Copyright 2026 The Tessiq Authors.
# SPDX-License-Identifier: Apache-2.0
"""A custom operation that replaces a torch one: `softmax` in place of `aten::_softmax`.

Nothing in a model names the custom operation. It calls `torch.softmax`, and the custom kernel runs
anyway: called on a tessiq tensor, because it is the torch operation's implementation there, and in
a compiled graph, because the `override-softmax` pass swaps the node. Whether a call ran the kernel
is read from `tessiq.library.fallbacks`: a float32 call that leaves the count alone ran the kernel,
and a float64 call that adds one shows the call came through the custom operation at all.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch
from torch import nn

import tessiq
from tessiq import _C
from tessiq.explain import pass_by_pass
from tessiq.fx import import_graph
from tessiq.recipe import build_recipe

SOFTMAXES: dict[str, Callable[[torch.Tensor, int], torch.Tensor]] = {
    "torch.softmax": lambda x, dim: torch.softmax(x, dim),
    "F.softmax": lambda x, dim: nn.functional.softmax(x, dim=dim),
    "nn.Softmax": lambda x, dim: nn.Softmax(dim=dim)(x),
}


def fallbacks() -> int:
    return tessiq.library.fallbacks("tessiq::softmax")


class Block(nn.Module):
    """A softmax between two built-in operations, written the way any model writes it."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = torch.relu(torch.softmax(x * 2.0, dim=-1))
        return out


def calls(graph: torch.fx.GraphModule) -> list[str]:
    return [str(n.target) for n in graph.graph.nodes if n.op == "call_function"]


@pytest.fixture
def needs_gpu() -> None:
    if tessiq.gpu_count() == 0:
        pytest.skip("needs a gpu")


@pytest.fixture(autouse=True)
def _fresh() -> None:
    torch._dynamo.reset()


def test_it_is_listed_with_the_operation_it_replaces() -> None:
    assert tessiq.library.overrides() == {"tessiq::softmax": "aten._softmax.default"}


def test_its_pass_runs_before_anything_decides_what_the_backend_claims() -> None:
    pipeline = tessiq.default_pipeline()
    assert pipeline.index("override-softmax") < pipeline.index("mark-supported")


@pytest.mark.cuda
@pytest.mark.usefixtures("needs_gpu")
@pytest.mark.parametrize("spelling", sorted(SOFTMAXES))
@pytest.mark.parametrize("dim", [0, 1, 2, -1])
def test_a_torch_softmax_on_a_gpu_tensor_runs_the_kernel(spelling: str, dim: int) -> None:
    x = torch.randn(6, 7, 300)
    before = fallbacks()
    got = SOFTMAXES[spelling](x.to(tessiq.gpu), dim)
    assert fallbacks() == before, "the reference ran instead of the kernel"
    torch.testing.assert_close(got.cpu(), torch.softmax(x, dim))


@pytest.mark.cuda
@pytest.mark.usefixtures("needs_gpu")
@pytest.mark.parametrize("shape", [(64, 32), (0, 5), ()])
def test_any_layout_and_shape(shape: tuple[int, ...]) -> None:
    # The 2D shape is transposed so the kernel is handed a strided tensor to pack first.
    x = torch.randn(shape)
    x = x.t() if x.dim() == 2 else x
    got = torch.softmax(x.to(tessiq.gpu), -1)
    torch.testing.assert_close(got.cpu(), torch.softmax(x, -1))


@pytest.mark.cuda
@pytest.mark.usefixtures("needs_gpu")
def test_a_type_it_was_not_written_for_runs_torchs_softmax() -> None:
    x = torch.randn(4, 50, dtype=torch.float64)
    before = fallbacks()
    got = torch.softmax(x.to(tessiq.gpu), -1)
    assert fallbacks() == before + 1
    torch.testing.assert_close(got.cpu(), torch.softmax(x, -1))


def test_on_the_cpu_reference_device_it_is_torchs_softmax() -> None:
    x = torch.randn(4, 50)
    before = fallbacks()
    got = torch.softmax(x.to(tessiq.cpu), -1)
    assert got.device == tessiq.cpu
    assert fallbacks() == before + 1
    torch.testing.assert_close(got.cpu(), torch.softmax(x, -1))


@pytest.mark.cuda
@pytest.mark.usefixtures("needs_gpu")
def test_compiled_the_softmax_stays_whole_and_becomes_the_kernel() -> None:
    graph = tessiq.capture(Block(), torch.randn(8, 128))
    assert "aten._softmax.default" in calls(graph)

    recipe = build_recipe(graph, tessiq.gpu, use_cache=False)
    custom = [k for k in recipe.kernels.values() if isinstance(k, _C.CustomKernel)]
    assert [k.op for k in custom] == ["tessiq::softmax"]
    assert custom[0].output == "_softmax"


@pytest.mark.cuda
@pytest.mark.usefixtures("needs_gpu")
def test_compiled_for_another_type_it_is_decomposed_as_before() -> None:
    graph = tessiq.capture(Block(), torch.randn(8, 128, dtype=torch.float64))
    assert "aten._softmax.default" not in calls(graph)
    assert "aten.amax.default" in calls(graph)

    recipe = build_recipe(graph, tessiq.gpu, use_cache=False)
    assert not any(isinstance(k, _C.CustomKernel) for k in recipe.kernels.values())


def test_the_pass_is_what_swaps_it() -> None:
    ir = import_graph(tessiq.capture(Block(), torch.randn(8, 128)))
    (change,) = [c for c in pass_by_pass(ir) if c.name == "override-softmax"]
    # The operation is all that changes: the value's name, its operand and its attributes stay.
    assert change.diff.splitlines() == [
        "-  %_softmax = aten._softmax.default(%mul) {arg1 = -1, arg2 = false} : f32[8,128]",
        "+  %_softmax = tessiq.softmax.default(%mul) {arg1 = -1, arg2 = false} : f32[8,128]",
    ]


@pytest.mark.cuda
@pytest.mark.usefixtures("needs_gpu")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_compiled_it_matches_eager(dtype: torch.dtype) -> None:
    x = torch.randn(16, 256, dtype=dtype)
    compiled = torch.compile(Block(), backend="tessiq", options={"device": str(tessiq.gpu)})
    with torch.no_grad():
        torch.testing.assert_close(compiled(x), Block()(x))


@pytest.mark.cuda
@pytest.mark.usefixtures("needs_gpu")
def test_compiled_on_a_shape_that_changes() -> None:
    compiled = torch.compile(
        Block(), backend="tessiq", options={"device": str(tessiq.gpu)}, dynamic=True
    )
    with torch.no_grad():
        for rows in (4, 12, 33):
            x = torch.randn(rows, 64)
            torch.testing.assert_close(compiled(x), Block()(x))


@pytest.mark.cuda
@pytest.mark.usefixtures("needs_gpu")
def test_training_through_it_reaches_the_same_parameters_as_eager() -> None:
    # The custom operation has no backward of its own. Replacing a torch operation keeps the torch
    # operation's, which reads the softmax the kernel computed.
    inputs = torch.randn(64, 16)
    targets = inputs[:, :1] * 2.0 + 0.5

    def train(compiled: bool) -> nn.Module:
        torch.manual_seed(0)
        net = nn.Sequential(nn.Linear(16, 32), nn.Softmax(dim=-1), nn.Linear(32, 1))
        run = (
            torch.compile(net, backend="tessiq", options={"device": str(tessiq.gpu)})
            if compiled
            else net
        )
        optimiser = torch.optim.SGD(net.parameters(), lr=0.5)
        for _ in range(50):
            optimiser.zero_grad()
            nn.functional.mse_loss(run(inputs), targets).backward()  # type: ignore[no-untyped-call]
            optimiser.step()
        return net

    eager = train(compiled=False)
    torch._dynamo.reset()
    ours = train(compiled=True)
    for theirs, mine in zip(eager.parameters(), ours.parameters(), strict=True):
        torch.testing.assert_close(mine, theirs, rtol=1e-4, atol=1e-5)
