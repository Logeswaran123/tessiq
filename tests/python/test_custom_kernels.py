# Copyright 2026 The Tessiq Authors.
# SPDX-License-Identifier: Apache-2.0
"""A custom operation with a kernel of its own: `rms_norm` from csrc/custom_ops.

Its kernel, rms_norm.cu, reduces each row and scales it in one launch, which no expression can say.
It runs as its own kernel in a compiled graph and is never fused. Whether a call ran the kernel or
fell back to the reference is read from `_C.custom_op_fallbacks`, since both give the same answer.
"""

from __future__ import annotations

import pytest
import torch

import tessiq
from tessiq import _C
from tessiq.recipe import build_recipe

rms_norm = torch.ops.tessiq.rms_norm


def expected(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight


def fallbacks() -> int:
    return _C.custom_op_fallbacks("tessiq::rms_norm")


class Block(torch.nn.Module):
    """RMSNorm between two built-in operations, the way it sits in a transformer."""

    def __init__(self, width: int, keyword: bool = False) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(width))
        self.keyword = keyword

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = (
            rms_norm(x * 2.0, self.weight, eps=1e-5)
            if self.keyword
            else rms_norm(x * 2.0, self.weight, 1e-5)
        )
        out: torch.Tensor = torch.relu(normed)
        return out


@pytest.fixture
def needs_gpu() -> None:
    if tessiq.gpu_count() == 0:
        pytest.skip("needs a gpu")


@pytest.fixture(autouse=True)
def _fresh() -> None:
    torch._dynamo.reset()


def test_on_the_host_it_is_the_reference() -> None:
    x, w = torch.randn(4, 64), torch.randn(64)
    torch.testing.assert_close(rms_norm(x, w, 1e-6), expected(x, w, 1e-6))


@pytest.mark.cuda
@pytest.mark.usefixtures("needs_gpu")
@pytest.mark.parametrize("shape", [(64, 512), (7, 300), (2, 3, 1000), (1, 1), (0, 16)])
def test_on_a_gpu_tensor_it_runs_the_kernel(shape: tuple[int, ...]) -> None:
    x, w = torch.randn(*shape), torch.randn(shape[-1])
    before = fallbacks()
    got = rms_norm(x.to(tessiq.gpu), w.to(tessiq.gpu), 1e-6)
    assert fallbacks() == before, "the reference ran instead of the kernel"
    torch.testing.assert_close(got.cpu(), expected(x, w, 1e-6))


@pytest.mark.cuda
@pytest.mark.usefixtures("needs_gpu")
def test_an_operand_laid_out_otherwise_is_packed_first() -> None:
    x, w = torch.randn(64, 32).t(), torch.randn(64)
    got = rms_norm(x.to(tessiq.gpu), w.to(tessiq.gpu), 1e-6)
    torch.testing.assert_close(got.cpu(), expected(x, w, 1e-6))


def test_on_the_cpu_reference_device_it_is_the_reference() -> None:
    x, w = torch.randn(4, 64), torch.randn(64)
    before = fallbacks()
    got = rms_norm(x.to(tessiq.cpu), w.to(tessiq.cpu), 1e-6)
    assert got.device == tessiq.cpu
    assert fallbacks() == before + 1
    torch.testing.assert_close(got.cpu(), expected(x, w, 1e-6))


@pytest.mark.cuda
@pytest.mark.usefixtures("needs_gpu")
def test_a_type_it_was_not_written_for_runs_the_reference() -> None:
    x, w = torch.randn(4, 64, dtype=torch.float64), torch.randn(64, dtype=torch.float64)
    before = fallbacks()
    got = rms_norm(x.to(tessiq.gpu), w.to(tessiq.gpu), 1e-6)
    assert fallbacks() == before + 1
    torch.testing.assert_close(got.cpu(), expected(x, w, 1e-6))


@pytest.mark.cuda
@pytest.mark.usefixtures("needs_gpu")
def test_a_weight_the_kernel_cannot_read_is_refused() -> None:
    # The reference would broadcast a single weight, but the kernel reads one per column, so the
    # launch function refuses it rather than let the kernel read past the end.
    x, w = torch.randn(4, 64), torch.randn(1)
    with pytest.raises(RuntimeError, match="reads one weight per column"):
        rms_norm(x.to(tessiq.gpu), w.to(tessiq.gpu), 1e-6)


@pytest.mark.cuda
@pytest.mark.usefixtures("needs_gpu")
def test_compiled_it_is_a_kernel_of_its_own() -> None:
    model = Block(128).eval()
    graph = tessiq.capture(model, torch.randn(8, 128))
    recipe = build_recipe(graph, tessiq.gpu, use_cache=False)

    custom = [k for k in recipe.kernels.values() if isinstance(k, _C.CustomKernel)]
    assert len(custom) == 1
    assert custom[0].op == "tessiq::rms_norm"
    assert custom[0].kind == "custom"
    assert custom[0].members == [custom[0].output]
    # Its neighbours are kernels of their own too: nothing fuses into a custom kernel.
    assert len(recipe.kernels) == 3


@pytest.mark.cuda
@pytest.mark.usefixtures("needs_gpu")
@pytest.mark.parametrize("keyword", [False, True])
def test_compiled_it_matches_eager(keyword: bool) -> None:
    model = Block(256, keyword=keyword).eval()
    x = torch.randn(16, 256)
    # AOTAutograd hands a keyword argument over as a positional one, so both spellings reach the
    # compiler as a call it can describe, and both become a kernel rather than an interpreted step.
    recipe = build_recipe(tessiq.capture(model, x), tessiq.gpu, use_cache=False)
    assert any(isinstance(k, _C.CustomKernel) for k in recipe.kernels.values())
    torch._dynamo.reset()

    compiled = torch.compile(model, backend="tessiq", options={"device": str(tessiq.gpu)})
    with torch.no_grad():
        torch.testing.assert_close(compiled(x), model(x))


@pytest.mark.cuda
@pytest.mark.usefixtures("needs_gpu")
def test_compiled_on_a_shape_that_changes() -> None:
    model = Block(64).eval()
    compiled = torch.compile(
        model, backend="tessiq", options={"device": str(tessiq.gpu)}, dynamic=True
    )
    with torch.no_grad():
        for rows in (4, 12, 33):
            x = torch.randn(rows, 64)
            torch.testing.assert_close(compiled(x), model(x))


@pytest.mark.cuda
@pytest.mark.usefixtures("needs_gpu")
def test_the_profiler_and_the_cost_replay_see_it_as_a_kernel() -> None:
    model = Block(128).eval()
    x = torch.randn(8, 128)
    graph = tessiq.capture(model, x)
    recipe = build_recipe(graph, tessiq.gpu, use_cache=False)
    run = tessiq.execute(recipe)
    node = next(n.name for n in graph.graph.nodes if "rms_norm" in str(n.target))

    with torch.no_grad(), tessiq.profile() as profile:
        run(x, *[p.detach() for p in model.parameters()])
    assert any(e.name == node and e.category == "kernel" for e in profile.events)

    costs = {cost.name: cost for cost in tessiq.kernel_costs(recipe, repeats=10)}
    assert costs[node].kind == "custom"
    assert costs[node].microseconds > 0
