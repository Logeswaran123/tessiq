# Copyright 2026 The Tessiq Authors.
# SPDX-License-Identifier: Apache-2.0
"""Running a model with generated kernels.

The point of these is not that a kernel runs. It is that running one gives the same numbers as not
running one, which is the only claim that matters and the one a code generator is most able to
break quietly.
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import replace
from typing import Any

import pytest
import torch
from torch._decomp import core_aten_decompositions
from torch._dynamo.backends.common import aot_autograd

import tessiq
from conftest import load_operations
from tessiq import _C
from tessiq.recipe import build_recipe

pytestmark = pytest.mark.cuda


@pytest.fixture(autouse=True)
def _needs_gpu() -> None:
    if tessiq.gpu_count() == 0:
        pytest.skip("needs a gpu")
    torch._dynamo.reset()


def captured(fn: Callable[..., Any], *inputs: torch.Tensor) -> torch.fx.GraphModule:
    graphs: list[torch.fx.GraphModule] = []

    def backend(graph_module: torch.fx.GraphModule, _inputs: object) -> Callable[..., object]:
        graphs.append(graph_module)
        return graph_module.forward

    torch.compile(
        fn,
        backend=aot_autograd(fw_compiler=backend, decompositions=core_aten_decompositions()),
        dynamic=False,
    )(*inputs)
    return graphs[0]


def chain(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(torch.relu(a * b + c))


def test_a_chain_becomes_one_launch() -> None:
    inputs = (torch.randn(1024), torch.randn(1024), torch.randn(1024))
    recipe = build_recipe(captured(chain, *inputs), tessiq.gpu, use_cache=False)

    assert len(recipe.kernels) == 1, recipe.kernels
    assert recipe.fused_away == {"mul", "add", "relu"}, recipe.fused_away
    # Four operations in the graph, one of which is now a kernel and three of which it subsumes.
    assert len(recipe.devices) == 4


def test_the_kernel_agrees_with_eager() -> None:
    inputs = (torch.randn(4096), torch.randn(4096), torch.randn(4096))
    compiled = torch.compile(
        chain, backend="tessiq", dynamic=False, options={"device": str(tessiq.gpu)}
    )
    assert torch.allclose(compiled(*inputs), chain(*inputs), atol=1e-5)


def test_the_kernel_agrees_with_the_same_graph_unfused() -> None:
    # The sharper comparison. Same recipe, same device, same everything, with the kernels removed,
    # so the only difference is whether the chain ran as one kernel or four operations.
    inputs = (torch.randn(4096), torch.randn(4096), torch.randn(4096))
    fused = build_recipe(captured(chain, *inputs), tessiq.gpu, use_cache=False)
    unfused = replace(fused, kernels={}, fused_away=frozenset())

    with_kernel = tessiq.execute(fused)(*inputs)[0]
    without = tessiq.execute(unfused)(*inputs)[0]
    assert torch.allclose(with_kernel, without, atol=1e-6)


def test_the_kernel_agrees_with_the_host_reference() -> None:
    # The CPU backend is written to be obviously correct rather than fast, which is what makes it
    # worth comparing against. This is the check the phase milestone asks for.
    inputs = (torch.randn(8192), torch.randn(8192), torch.randn(8192))
    graph = captured(chain, *inputs)

    on_gpu = tessiq.execute(build_recipe(graph, tessiq.gpu, use_cache=False))(*inputs)[0]
    on_host = tessiq.execute(build_recipe(graph, tessiq.cpu, use_cache=False))(*inputs)[0]
    assert torch.allclose(on_gpu, on_host, atol=1e-6)


@pytest.mark.parametrize("size", [1, 255, 256, 257, 4095, 4096])
def test_sizes_around_the_block_boundary(size: int) -> None:
    # The block is 256 threads. A missing bounds check shows up only where the count does not
    # divide it, so the sizes that do are here as controls rather than as the interesting case.
    inputs = (torch.randn(size), torch.randn(size), torch.randn(size))
    recipe = build_recipe(captured(chain, *inputs), tessiq.gpu, use_cache=False)
    assert torch.allclose(tessiq.execute(recipe)(*inputs)[0], chain(*inputs), atol=1e-5)


def test_a_matmul_runs_between_two_fused_chains() -> None:
    # A matmul used to split the chain and go to the host. cuBLAS runs it now, so the whole thing
    # is three kernels on the device: the chain before it, the multiply, and the chain after.
    def model(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(torch.tanh(torch.relu(a + b) @ b))

    inputs = (torch.randn(64, 64), torch.randn(64, 64))
    recipe = build_recipe(captured(model, *inputs), tessiq.gpu, use_cache=False)

    assert len(recipe.kernels) == 3, recipe.kernels
    assert torch.allclose(tessiq.execute(recipe)(*inputs)[0], model(*inputs), atol=1e-4)


def test_repeated_calls_give_the_same_answer() -> None:
    # A launch is asynchronous and the transfers that read it are on another stream. If the
    # ordering between them were wrong, this is where it would show up.
    inputs = (torch.randn(65536), torch.randn(65536), torch.randn(65536))
    run = tessiq.execute(build_recipe(captured(chain, *inputs), tessiq.gpu, use_cache=False))

    first = run(*inputs)[0]
    for _ in range(50):
        assert torch.equal(run(*inputs)[0], first)


def test_training_through_a_kernel_matches_eager() -> None:
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(16, 32), torch.nn.Tanh(), torch.nn.Sigmoid())
    reference = copy.deepcopy(model)
    inputs = torch.randn(8, 16)

    torch.compile(model, backend="tessiq", dynamic=False, options={"device": str(tessiq.gpu)})(
        inputs
    ).sum().backward()
    reference(inputs).sum().backward()

    for got, want in zip(model.parameters(), reference.parameters(), strict=True):
        assert got.grad is not None
        assert want.grad is not None
        assert torch.allclose(got.grad, want.grad, atol=1e-4)


def test_the_host_reference_compiles_no_kernels() -> None:
    inputs = (torch.randn(256), torch.randn(256), torch.randn(256))
    recipe = build_recipe(captured(chain, *inputs), tessiq.cpu, use_cache=False)
    assert recipe.kernels == {}
    assert recipe.fused_away == frozenset()


# --- the operations added for coverage -------------------------------------------------------

# Every operation needs inputs it is defined on, and there are only a few shapes of answer to that
# question. Anything not named here is fine with a plain normal sample.
POSITIVE = "positive"
ABOVE_MINUS_ONE = "above -1"
UNIT = "unit interval"
NONZERO = "nonzero"

DOMAINS = {
    "log": POSITIVE,
    "log2": POSITIVE,
    "log10": POSITIVE,
    "sqrt": POSITIVE,
    "rsqrt": POSITIVE,
    "pow": POSITIVE,
    "log1p": ABOVE_MINUS_ONE,
    "asin": UNIT,
    "acos": UNIT,
    "reciprocal": NONZERO,
}

# Applied with a trailing multiply, because a single operation is a group of one and a group of one
# is not fused. Two operations is the shortest thing that produces a kernel.
CALLS: dict[str, Callable[..., torch.Tensor]] = {
    "pow": lambda x: torch.pow(x, 1.7),
}


def sample(name: str, count: int = 4096) -> torch.Tensor:
    generator = torch.Generator().manual_seed(len(name))
    values = torch.randn(count, generator=generator)
    domain = DOMAINS.get(name)
    if domain == POSITIVE:
        return values.abs() + 0.125
    if domain == ABOVE_MINUS_ONE:
        return values.abs() - 0.5
    if domain == UNIT:
        return torch.rand(count, generator=generator) * 1.8 - 0.9
    if domain == NONZERO:
        return values + torch.sign(values) * 0.125
    return values


ARITY = {str(op["name"]): int(op["arity"]) for op in load_operations()}


def elementwise_with_kernels() -> list[str]:
    return sorted(
        str(op["name"])
        for op in load_operations()
        if op["kernel"] and op["category"] == "elementwise"
    )


def second_operand(count: int = 4096) -> torch.Tensor:
    """The right-hand side of a binary operation: positive, so it is safe as a divisor too."""
    generator = torch.Generator().manual_seed(17)
    return torch.randn(count, generator=generator).abs() + 0.125


@pytest.mark.parametrize("name", elementwise_with_kernels())
def test_an_elementwise_kernel_agrees_with_eager(name: str) -> None:
    # Driven by the table, so an operation cannot be added to ops.toml without its numbers being
    # checked here. What it checks is narrow and the only thing worth checking: the value the
    # generated kernel writes is the value PyTorch would have written.
    call: Callable[..., torch.Tensor] = CALLS.get(name, getattr(torch, name))

    def model(*args: torch.Tensor) -> torch.Tensor:
        return call(*args) * 2.0

    inputs: tuple[torch.Tensor, ...] = (sample(name),)
    if ARITY[name] == 2 and name not in CALLS:
        inputs = (*inputs, second_operand())

    recipe = build_recipe(captured(model, *inputs), tessiq.gpu, use_cache=False)
    assert len(recipe.kernels) == 1, f"{name} produced no kernel: {recipe.kernels}"

    compiled = torch.compile(
        model, backend="tessiq", dynamic=False, options={"device": str(tessiq.gpu)}
    )
    # One tolerance for every operation, and a tight one. These are the CUDA math library's own
    # functions compared against PyTorch's CPU implementations, so anything looser would be hiding
    # a wrong expression rather than accommodating float32.
    torch.testing.assert_close(compiled(*inputs), model(*inputs), rtol=1e-6, atol=1e-5)


# --- reductions ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "call"),
    [
        ("sum", lambda t: torch.sum(t, dim=1)),
        ("sum_keepdim", lambda t: torch.sum(t, dim=1, keepdim=True)),
        ("sum_all", torch.sum),
        ("mean", lambda t: torch.mean(t, dim=-1)),
        ("mean_all", torch.mean),
        ("amax", lambda t: torch.amax(t, dim=1)),
        ("amin", lambda t: torch.amin(t, dim=0)),
    ],
)
def test_a_reduction_after_a_kernel_agrees_with_eager(
    name: str, call: Callable[[torch.Tensor], torch.Tensor]
) -> None:
    # The seam: a fused kernel writes an intermediate, a reduction kernel reads it, and the
    # answer is the one eager gives. The reduction sums in a different order than a sequential
    # host loop does, so the last digits differ and the tolerance is relative.
    def model(x: torch.Tensor) -> torch.Tensor:
        return call(torch.relu(x * 2.0))

    x = torch.randn(4, 8)
    recipe = build_recipe(captured(model, x), tessiq.gpu, use_cache=False)
    # One fused kernel for the multiply and relu, one reduction kernel for the reduction itself.
    assert len(recipe.kernels) == 2, f"{name} did not get both kernels: {recipe.kernels}"

    compiled = torch.compile(
        model, backend="tessiq", dynamic=False, options={"device": str(tessiq.gpu)}
    )
    torch.testing.assert_close(compiled(x), model(x), rtol=1e-6, atol=1e-5)


def test_a_reduction_between_two_kernels_agrees_with_eager() -> None:
    def model(x: torch.Tensor) -> torch.Tensor:
        s = torch.sum(torch.relu(x * 2.0), dim=1, keepdim=True)
        return torch.tanh(torch.abs(s) + 1.0)

    x = torch.randn(4, 8)
    recipe = build_recipe(captured(model, x), tessiq.gpu, use_cache=False)
    # Two fused groups and the reduction between them.
    assert len(recipe.kernels) == 3, recipe.kernels

    compiled = torch.compile(
        model, backend="tessiq", dynamic=False, options={"device": str(tessiq.gpu)}
    )
    torch.testing.assert_close(compiled(x), model(x), rtol=1e-6, atol=1e-5)


# The shapes that decide which index arithmetic the kernel does. Reducing the last axis walks
# contiguous memory, reducing the first walks with a stride, and a middle axis does both.
REDUCTION_SHAPES = [
    ((4, 8), 1, "last axis, inner 1"),
    ((4, 8), 0, "first axis, strided"),
    ((4, 8, 3), 1, "middle axis, outer and inner both set"),
    ((4, 8, 3), -1, "last axis of a rank-3 tensor"),
    ((3, 1024), 1, "longer than one block, so the strided loop runs more than once"),
    ((1, 4096), 1, "one output element from a long axis"),
    ((512, 3), 1, "many outputs from a short axis"),
]


@pytest.mark.parametrize(("shape", "dim", "why"), REDUCTION_SHAPES)
@pytest.mark.parametrize("keepdim", [False, True])
def test_a_reduction_kernel_agrees_with_eager(
    shape: tuple[int, ...], dim: int, why: str, keepdim: bool
) -> None:
    def model(x: torch.Tensor) -> torch.Tensor:
        return torch.sum(torch.relu(x * 2.0), dim=dim, keepdim=keepdim)

    x = torch.randn(shape)
    recipe = build_recipe(captured(model, x), tessiq.gpu, use_cache=False)
    assert len(recipe.kernels) == 2, f"{why}: {recipe.kernels}"

    compiled = torch.compile(
        model, backend="tessiq", dynamic=False, options={"device": str(tessiq.gpu)}
    )
    torch.testing.assert_close(compiled(x), model(x), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(
    "call",
    [
        lambda t: torch.sum(t, dim=1),
        lambda t: torch.mean(t, dim=1),
        lambda t: torch.amax(t, dim=1),
        lambda t: torch.amin(t, dim=1),
        torch.sum,
        torch.mean,
    ],
    ids=["sum", "mean", "amax", "amin", "sum_all", "mean_all"],
)
def test_every_reduction_kernel_agrees_with_eager(
    call: Callable[[torch.Tensor], torch.Tensor],
) -> None:
    def model(x: torch.Tensor) -> torch.Tensor:
        return call(torch.relu(x * 2.0))

    x = torch.randn(4, 300)
    recipe = build_recipe(captured(model, x), tessiq.gpu, use_cache=False)
    assert len(recipe.kernels) == 2, recipe.kernels

    compiled = torch.compile(
        model, backend="tessiq", dynamic=False, options={"device": str(tessiq.gpu)}
    )
    torch.testing.assert_close(compiled(x), model(x), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("reduce_op", [torch.amax, torch.amin])
def test_max_and_min_propagate_a_nan_the_way_torch_does(
    reduce_op: Callable[..., torch.Tensor],
) -> None:
    # fmax and a plain `a > b ? a : b` both return the other operand when one is a NaN, and torch
    # returns the NaN. The kernel tests for it explicitly, and this is why.
    x = torch.randn(4, 300)
    x[2, 17] = float("nan")

    def model(t: torch.Tensor) -> torch.Tensor:
        return reduce_op(torch.abs(t) * 2.0, dim=1)

    compiled = torch.compile(
        model, backend="tessiq", dynamic=False, options={"device": str(tessiq.gpu)}
    )
    got = compiled(x)
    assert torch.isnan(got[2]), got
    torch.testing.assert_close(got, model(x), rtol=1e-5, atol=1e-5, equal_nan=True)


def test_an_axis_set_the_kernel_cannot_express_falls_back() -> None:
    # Reducing the first and last axes of a rank-3 tensor leaves the kept axis in the middle, which
    # [outer, reduce, inner] cannot describe. There is no kernel for it and the answer still has to
    # be right, which is what the fallback is for.
    def model(x: torch.Tensor) -> torch.Tensor:
        return torch.sum(torch.relu(x * 2.0), dim=(0, 2))

    x = torch.randn(4, 8, 3)
    recipe = build_recipe(captured(model, x), tessiq.gpu, use_cache=False)
    # The fused kernel in front of it, and nothing for the reduction.
    assert list(recipe.kernels) == ["relu"], (
        f"the reduction should have no kernel: {recipe.kernels}"
    )

    compiled = torch.compile(
        model, backend="tessiq", dynamic=False, options={"device": str(tessiq.gpu)}
    )
    torch.testing.assert_close(compiled(x), model(x), rtol=1e-5, atol=1e-5)


# --- matrix multiplies --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "call", "shapes"),
    [
        ("mm", lambda a, b: a @ b, ((12, 32), (32, 16))),
        ("mm transposed", lambda a, b: a @ b.t(), ((12, 32), (16, 32))),
        ("bmm", lambda a, b: torch.bmm(a, b), ((8, 6, 8), (8, 8, 6))),
        ("mm square", lambda a, b: a @ b, ((64, 64), (64, 64))),
        ("mm thin", lambda a, b: a @ b, ((1, 128), (128, 1))),
    ],
)
def test_a_matmul_agrees_with_eager(
    name: str, call: Callable[..., torch.Tensor], shapes: tuple[tuple[int, ...], ...]
) -> None:
    inputs = tuple(torch.randn(*shape) for shape in shapes)
    recipe = build_recipe(captured(call, *inputs), tessiq.gpu, use_cache=False)
    assert len(recipe.kernels) == 1, f"{name}: {recipe.kernels}"

    compiled = torch.compile(
        call, backend="tessiq", dynamic=False, options={"device": str(tessiq.gpu)}
    )
    torch.testing.assert_close(compiled(*inputs), call(*inputs), rtol=1e-4, atol=1e-4)


def test_a_linear_layer_uses_a_transposed_weight_without_copying_it() -> None:
    # nn.Linear stores its weight as [out, in] and computes with [in, out], so every one of them
    # produces a transposed operand. BLAS takes that as a flag, which is the whole reason layouts
    # are part of the type.
    layer = torch.nn.Linear(32, 16).eval()
    example = torch.randn(12, 32)

    with torch.no_grad():
        recipe = build_recipe(captured(layer, example), tessiq.gpu, use_cache=False)
        gemm = next(k for k in recipe.kernels.values() if isinstance(k, _C.GemmKernel))
        assert gemm.transposed == (False, True), gemm.transposed
        assert gemm.extents == (1, 12, 16, 32), gemm.extents

        compiled = torch.compile(
            layer, backend="tessiq", dynamic=False, options={"device": str(tessiq.gpu)}
        )
        torch.testing.assert_close(compiled(example), layer(example), rtol=1e-4, atol=1e-4)


def test_a_matmul_reads_its_operands_where_they_are() -> None:
    # Every other kernel walks one index across every buffer, so the executor packs its inputs
    # first. Packing a transposed operand would move the data and leave the transpose flag
    # describing a layout that is no longer there, which is a wrong answer rather than a slow one.
    layer = torch.nn.Linear(8, 4).eval()
    with torch.no_grad():
        recipe = build_recipe(captured(layer, torch.randn(3, 8)), tessiq.gpu, use_cache=False)
    gemm = next(k for k in recipe.kernels.values() if isinstance(k, _C.GemmKernel))
    assert not gemm.packs_inputs


def test_a_broadcasting_chain_agrees_with_eager() -> None:
    # This crashed before the fusion guard: the group was built and the launch was handed an
    # operand of the wrong size. It runs one operation at a time now, which is slower and right.
    def model(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.relu(a - b) * 2.0

    inputs = (torch.randn(4, 8), torch.randn(8))
    compiled = torch.compile(
        model, backend="tessiq", dynamic=False, options={"device": str(tessiq.gpu)}
    )
    torch.testing.assert_close(compiled(*inputs), model(*inputs), rtol=1e-6, atol=1e-5)


def transposed_chain(x: torch.Tensor) -> torch.Tensor:
    return torch.relu(x.t() * 2.0) + 1.0


def broadcast_chain(x: torch.Tensor) -> torch.Tensor:
    return torch.relu(x.expand(4, 8) * 2.0)


def transposed_sum(x: torch.Tensor) -> torch.Tensor:
    return x.t().sum(dim=0)


@pytest.mark.parametrize(
    ("model", "shape"),
    [(transposed_chain, (8, 4)), (broadcast_chain, (1, 8)), (transposed_sum, (8, 4))],
)
def test_a_kernel_reading_a_strided_value_agrees_with_eager(
    model: Callable[[torch.Tensor], torch.Tensor], shape: tuple[int, ...]
) -> None:
    # Each reads a value laid out other than packed, which layout-assign copies first. None of
    # these compiled before: the copy reached the kernel under a name with a dot in it.
    x = torch.randn(*shape)
    recipe = build_recipe(captured(model, x), tessiq.gpu, use_cache=False)
    assert recipe.kernels
    assert all("." not in name for kernel in recipe.kernels.values() for name in kernel.inputs)

    compiled = torch.compile(
        model, backend="tessiq", dynamic=False, options={"device": str(tessiq.gpu)}
    )
    torch.testing.assert_close(compiled(x), model(x))


def test_a_transformer_block_agrees_with_eager() -> None:
    import test_models

    model = test_models.EncoderBlock().eval()
    example = torch.randn(2, 6, 32)
    with torch.no_grad():
        compiled = torch.compile(
            model, backend="tessiq", dynamic=False, options={"device": str(tessiq.gpu)}
        )
        torch.testing.assert_close(compiled(example), model(example), rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize(
    ("name", "shapes"),
    [
        ("row added to every row", ((4, 8), (8,))),
        ("one value per row", ((4, 8), (4, 1))),
        ("one value per row, rank 3", ((2, 6, 8), (2, 6, 1))),
        ("a middle axis", ((2, 6, 8), (6, 1))),
        ("a scalar", ((4, 8), (1, 1))),
    ],
)
def test_a_broadcasting_chain_agrees_with_eager_and_is_one_kernel(
    name: str, shapes: tuple[tuple[int, ...], ...]
) -> None:
    # Each of these used to end the group, which is what a softmax and a layer norm are made of.
    def model(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.relu(a - b) * 2.0

    inputs = tuple(torch.randn(*shape) for shape in shapes)
    recipe = build_recipe(captured(model, *inputs), tessiq.gpu, use_cache=False)
    assert len(recipe.kernels) == 1, f"{name}: {recipe.kernels}"

    compiled = torch.compile(
        model, backend="tessiq", dynamic=False, options={"device": str(tessiq.gpu)}
    )
    torch.testing.assert_close(compiled(*inputs), model(*inputs), rtol=1e-6, atol=1e-5)
