# Copyright 2026 The Tessiq Authors.
# SPDX-License-Identifier: Apache-2.0
"""`gemm_bias_gelu` from csrc/custom_ops: a CUTLASS GEMM whose epilogue adds the bias and applies GELU.

These check it against torch's linear layer and GELU called directly, that it declines what its
epilogue cannot read, that `rewrite-gemm-bias-gelu` turns a linear layer followed by GELU into this
one kernel and leaves alone what it would compute differently, and that the slices of a split K
take their semaphores from tessiq's allocator.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
import torch
from torch._dynamo.backends.common import aot_autograd
from torch.nn import functional

import tessiq
from tessiq.decompositions import decompositions
from tessiq.recipe import build_recipe

gemm_bias_gelu = torch.ops.tessiq.gemm_bias_gelu


def expected(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """In float64, so the kernel and torch's float32 are each compared with something better."""
    return functional.gelu(functional.linear(x.double(), weight.double(), bias.double())).float()


def fallbacks() -> int:
    return tessiq.library.fallbacks("tessiq::gemm_bias_gelu")


def arguments(m: int, k: int, n: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return torch.randn(m, k), torch.randn(n, k) / k**0.5, torch.randn(n)


def on_gpu(*tensors: torch.Tensor) -> list[torch.Tensor]:
    return [t.to(tessiq.gpu) for t in tensors]


def captured(fn: Callable[..., Any], *inputs: torch.Tensor) -> torch.fx.GraphModule:
    graphs: list[torch.fx.GraphModule] = []

    def backend(graph_module: torch.fx.GraphModule, _inputs: object) -> Callable[..., object]:
        graphs.append(graph_module)
        return graph_module.forward

    torch.compile(
        fn,
        backend=aot_autograd(fw_compiler=backend, decompositions=decompositions()),
        dynamic=False,
    )(*inputs)
    return graphs[0]


def kinds(fn: Callable[..., Any], *inputs: torch.Tensor, rewrites: list[str]) -> list[str]:
    recipe = build_recipe(captured(fn, *inputs), tessiq.gpu, use_cache=False, rewrites=rewrites)
    return sorted(type(kernel).__name__ for kernel in recipe.kernels.values())


#: What torch.compile is given to run the rewrite. Typed loosely, since torch's stub for options
#: does not allow a list.
REWRITING: dict[str, Any] = {"rewrites": ["gemm_bias_gelu"]}


def linear_gelu(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    return functional.gelu(functional.linear(x, weight, bias))


@pytest.fixture
def host_code() -> None:
    if tessiq.gpu_count() == 0:
        pytest.skip("needs a gpu")
    if "tessiq::gemm_bias_gelu" not in tessiq.library.precompiled():
        pytest.skip("this build had no nvcc, so gemm_bias_gelu has no host code")


@pytest.fixture(autouse=True)
def _fresh() -> None:
    torch._dynamo.reset()


def test_on_the_host_it_is_the_reference() -> None:
    x, weight, bias = arguments(8, 64, 32)
    torch.testing.assert_close(gemm_bias_gelu(x, weight, bias), linear_gelu(x, weight, bias))


def test_on_the_reference_device_it_runs_the_reference() -> None:
    x, weight, bias = arguments(8, 64, 32)
    before = fallbacks()
    out = gemm_bias_gelu(*(t.to(tessiq.cpu) for t in (x, weight, bias)))
    assert fallbacks() == before + 1
    torch.testing.assert_close(out.cpu(), linear_gelu(x, weight, bias))


@pytest.mark.cuda
@pytest.mark.usefixtures("host_code")
@pytest.mark.parametrize(
    ("m", "k", "n"),
    [
        (8, 64, 32),
        (1, 1, 1),
        # Extents no tile divides, and a K of 3.
        (129, 3, 257),
        # Few tiles and a deep K, so K is split into slices.
        (64, 2048, 512),
        (300, 700, 1000),
    ],
)
def test_on_a_gpu_it_runs_its_host_code(m: int, k: int, n: int) -> None:
    x, weight, bias = arguments(m, k, n)
    before = fallbacks()
    out = gemm_bias_gelu(*on_gpu(x, weight, bias))
    assert fallbacks() == before
    # Float32 arithmetic throughout, so it is as close to float64 as torch's own float32 is.
    reference = expected(x, weight, bias)
    torch_error = (linear_gelu(x, weight, bias) - reference).abs().max().item()
    kernel_error = (out.cpu() - reference).abs().max().item()
    assert kernel_error <= max(8 * torch_error, 1e-6), (kernel_error, torch_error)


@pytest.mark.cuda
@pytest.mark.usefixtures("host_code")
@pytest.mark.parametrize("bias", [torch.randn(1), torch.randn(4, 16)], ids=["one", "per element"])
def test_a_bias_that_broadcasts_is_declined_and_runs_the_reference(bias: torch.Tensor) -> None:
    # The reference broadcasts any bias that fits the output, but the epilogue reads one per column.
    x, weight = torch.randn(4, 8), torch.randn(16, 8)
    before = fallbacks()
    out = gemm_bias_gelu(*on_gpu(x, weight, bias))
    assert fallbacks() == before + 1
    torch.testing.assert_close(out.cpu(), linear_gelu(x, weight, bias))


@pytest.mark.cuda
@pytest.mark.usefixtures("host_code")
def test_a_type_it_does_not_declare_runs_the_reference() -> None:
    x, weight, bias = (t.double() for t in arguments(8, 64, 32))
    before = fallbacks()
    out = gemm_bias_gelu(*on_gpu(x, weight, bias))
    assert fallbacks() == before + 1
    torch.testing.assert_close(out.cpu(), linear_gelu(x, weight, bias))


@pytest.mark.cuda
@pytest.mark.usefixtures("host_code")
def test_the_rewrite_makes_a_linear_layer_and_gelu_one_kernel() -> None:
    inputs = arguments(256, 128, 512)
    assert kinds(linear_gelu, *inputs, rewrites=[]) == ["FusedKernel", "GemmKernel"]
    assert kinds(linear_gelu, *inputs, rewrites=["gemm_bias_gelu"]) == ["CustomKernel"]

    before = fallbacks()
    compiled = torch.compile(
        linear_gelu,
        backend="tessiq",
        options={"device": str(tessiq.gpu), **REWRITING},
    )
    out = compiled(*inputs)
    assert fallbacks() == before
    torch.testing.assert_close(out, linear_gelu(*inputs), rtol=1e-5, atol=1e-5)


@pytest.mark.cuda
@pytest.mark.usefixtures("host_code")
@pytest.mark.parametrize(
    ("name", "fn", "x"),
    [
        (
            "the tanh approximation",
            lambda x, w, b: functional.gelu(functional.linear(x, w, b), approximate="tanh"),
            torch.randn(16, 32),
        ),
        (
            "an addmm with a beta",
            lambda x, w, b: functional.gelu(torch.addmm(b, x, w.t(), beta=0.5)),
            torch.randn(16, 32),
        ),
        # Flattened by a view, with another view between the addmm and the GELU.
        ("a batched input", linear_gelu, torch.randn(2, 8, 32)),
    ],
)
def test_the_rewrite_leaves_alone_what_it_does_not_match(
    name: str, fn: Callable[..., torch.Tensor], x: torch.Tensor
) -> None:
    weight, bias = torch.randn(64, 32), torch.randn(64)
    assert "CustomKernel" not in kinds(fn, x, weight, bias, rewrites=["gemm_bias_gelu"]), name
    compiled = torch.compile(fn, backend="tessiq", options={"device": str(tessiq.gpu), **REWRITING})
    torch.testing.assert_close(compiled(x, weight, bias), fn(x, weight, bias), rtol=1e-5, atol=1e-5)


@pytest.mark.cuda
@pytest.mark.usefixtures("host_code")
@pytest.mark.parametrize(
    ("m", "k", "n", "blocks"),
    [(64, 2048, 1024, 2), (4096, 256, 1024, 1)],
    ids=["split", "not split"],
)
def test_a_split_k_takes_its_semaphores_from_tessiqs_allocator(
    m: int, k: int, n: int, blocks: int
) -> None:
    # Eight tiles of output and a K of 2048, so K is split and the slices take turns adding into the
    # output through a semaphore per tile in workspace. Five hundred tiles are enough without a
    # split, and there is no workspace.
    x, weight, bias = on_gpu(*arguments(m, k, n))
    gemm_bias_gelu(x, weight, bias)
    tessiq.synchronize(tessiq.gpu)
    before = tessiq.memory_stats(tessiq.gpu)

    out = gemm_bias_gelu(x, weight, bias)
    after = tessiq.memory_stats(tessiq.gpu)
    assert after.allocation_count - before.allocation_count == blocks
    # Only the result is still held.
    assert after.live_blocks == before.live_blocks + 1
    del out
