# Copyright 2026 The Tessiq Authors.
# SPDX-License-Identifier: Apache-2.0
"""`gemm_bias_gelu_tanh` from csrc/custom_ops: a linear layer and the tanh GELU as one cuBLASLt call.

Its host code launches no kernel of its own. These check it against torch's linear layer and tanh
GELU, that it copes with a pointer less aligned than cuBLASLt assumes, that it declines what its
epilogue cannot read, that `rewrite-gemm-bias-gelu-tanh` makes the layer one call and takes only
the tanh GELU, and that the workspace cuBLASLt asks for comes from tessiq's allocator.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
import torch
from torch._dynamo.backends.common import aot_autograd
from torch.nn import functional

import tessiq
from tessiq import _C
from tessiq.decompositions import decompositions
from tessiq.recipe import build_recipe

gemm_bias_gelu_tanh = torch.ops.tessiq.gemm_bias_gelu_tanh

#: What torch.compile is given to run a rewrite. Typed loosely, since torch's stub for options
#: does not allow a list.
REWRITING: dict[str, Any] = {"rewrites": ["gemm_bias_gelu_tanh"]}

#: How far cuBLASLt's GELU may be from float64 beyond torch's own float32. Its tanh is less exact
#: than torch's: about 5e-6 off even when the product is exact.
EPILOGUE_ERROR = 2e-5


def linear_gelu_tanh(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    return functional.gelu(functional.linear(x, weight, bias), approximate="tanh")


def linear_gelu(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    return functional.gelu(functional.linear(x, weight, bias))


def expected(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    return linear_gelu_tanh(x.double(), weight.double(), bias.double()).float()


def fallbacks() -> int:
    return tessiq.library.fallbacks("tessiq::gemm_bias_gelu_tanh")


def arguments(m: int, k: int, n: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return torch.randn(m, k), torch.randn(n, k) / k**0.5, torch.randn(n)


def on_gpu(*tensors: torch.Tensor) -> list[torch.Tensor]:
    return [t.to(tessiq.gpu) for t in tensors]


def assert_close_to_float64(
    out: torch.Tensor, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor
) -> None:
    reference = expected(x, weight, bias)
    torch_error = (linear_gelu_tanh(x, weight, bias) - reference).abs().max().item()
    error = (out.cpu() - reference).abs().max().item()
    assert error <= torch_error + EPILOGUE_ERROR, (error, torch_error)


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


def custom_ops(fn: Callable[..., Any], *inputs: torch.Tensor, rewrites: list[str]) -> list[str]:
    """The custom operation each custom kernel runs, and the class of every other kernel."""
    recipe = build_recipe(captured(fn, *inputs), tessiq.gpu, use_cache=False, rewrites=rewrites)
    return sorted(
        kernel.op if isinstance(kernel, _C.CustomKernel) else type(kernel).__name__
        for kernel in recipe.kernels.values()
    )


@pytest.fixture
def host_code() -> None:
    if tessiq.gpu_count() == 0:
        pytest.skip("needs a gpu")
    if "tessiq::gemm_bias_gelu_tanh" not in tessiq.library.precompiled():
        pytest.skip("this build had no nvcc, so gemm_bias_gelu_tanh has no host code")


@pytest.fixture(autouse=True)
def _fresh() -> None:
    torch._dynamo.reset()


def test_on_the_host_it_is_the_reference() -> None:
    x, weight, bias = arguments(8, 64, 32)
    torch.testing.assert_close(
        gemm_bias_gelu_tanh(x, weight, bias), linear_gelu_tanh(x, weight, bias)
    )


@pytest.mark.cuda
@pytest.mark.usefixtures("host_code")
@pytest.mark.parametrize(
    ("m", "k", "n"), [(8, 64, 32), (1, 1, 1), (129, 3, 257), (64, 2048, 512), (300, 700, 1000)]
)
def test_on_a_gpu_it_calls_cublaslt(m: int, k: int, n: int) -> None:
    x, weight, bias = arguments(m, k, n)
    before = fallbacks()
    out = gemm_bias_gelu_tanh(*on_gpu(x, weight, bias))
    assert fallbacks() == before
    assert_close_to_float64(out, x, weight, bias)


@pytest.mark.cuda
@pytest.mark.usefixtures("host_code")
def test_a_pointer_less_aligned_than_cublaslt_assumes_still_runs() -> None:
    # cuBLASLt assumes 256-byte pointers unless told otherwise. x here starts one float into its
    # storage.
    x, weight, bias = arguments(64, 128, 96)
    storage = torch.cat([torch.zeros(1), x.flatten()]).to(tessiq.gpu)
    offset = storage[1:].view(64, 128)
    assert offset.data_ptr() % 256 == 4
    before = fallbacks()
    out = gemm_bias_gelu_tanh(offset, *on_gpu(weight, bias))
    assert fallbacks() == before
    assert_close_to_float64(out, x, weight, bias)


@pytest.mark.cuda
@pytest.mark.usefixtures("host_code")
@pytest.mark.parametrize("bias", [torch.randn(1), torch.randn(4, 16)], ids=["one", "per element"])
def test_a_bias_that_broadcasts_is_declined_and_runs_the_reference(bias: torch.Tensor) -> None:
    x, weight = torch.randn(4, 8), torch.randn(16, 8)
    before = fallbacks()
    out = gemm_bias_gelu_tanh(*on_gpu(x, weight, bias))
    assert fallbacks() == before + 1
    torch.testing.assert_close(out.cpu(), linear_gelu_tanh(x, weight, bias))


@pytest.mark.cuda
@pytest.mark.usefixtures("host_code")
def test_a_type_it_does_not_declare_runs_the_reference() -> None:
    x, weight, bias = (t.double() for t in arguments(8, 64, 32))
    before = fallbacks()
    out = gemm_bias_gelu_tanh(*on_gpu(x, weight, bias))
    assert fallbacks() == before + 1
    torch.testing.assert_close(out.cpu(), linear_gelu_tanh(x, weight, bias))


@pytest.mark.cuda
@pytest.mark.usefixtures("host_code")
def test_the_rewrite_makes_a_linear_layer_and_tanh_gelu_one_call() -> None:
    # Without it the tanh GELU runs on the host, since the elementwise GELU is the exact one.
    inputs = arguments(256, 128, 512)
    assert custom_ops(linear_gelu_tanh, *inputs, rewrites=[]) == ["GemmKernel"]
    assert custom_ops(linear_gelu_tanh, *inputs, rewrites=["gemm_bias_gelu_tanh"]) == [
        "tessiq::gemm_bias_gelu_tanh"
    ]

    before = fallbacks()
    compiled = torch.compile(
        linear_gelu_tanh, backend="tessiq", options={"device": str(tessiq.gpu), **REWRITING}
    )
    out = compiled(*inputs)
    assert fallbacks() == before
    assert_close_to_float64(out, *inputs)


@pytest.mark.cuda
@pytest.mark.usefixtures("host_code")
def test_each_gelu_goes_to_the_operation_that_computes_it() -> None:
    if "tessiq::gemm_bias_gelu" not in tessiq.library.precompiled():
        pytest.skip("this build has no host code for gemm_bias_gelu")
    both = ["gemm_bias_gelu", "gemm_bias_gelu_tanh"]
    inputs = arguments(64, 32, 48)
    assert custom_ops(linear_gelu, *inputs, rewrites=both) == ["tessiq::gemm_bias_gelu"]
    assert custom_ops(linear_gelu_tanh, *inputs, rewrites=both) == ["tessiq::gemm_bias_gelu_tanh"]


@pytest.mark.cuda
@pytest.mark.usefixtures("host_code")
def test_the_workspace_cublaslt_asks_for_comes_from_tessiqs_allocator_and_goes_back() -> None:
    taken = []
    for m, k, n in [(129, 3, 257), (64, 4096, 4096), (512, 768, 3072)]:
        x, weight, bias = on_gpu(*arguments(m, k, n))
        gemm_bias_gelu_tanh(x, weight, bias)
        tessiq.synchronize(tessiq.gpu)
        before = tessiq.memory_stats(tessiq.gpu)

        out = gemm_bias_gelu_tanh(x, weight, bias)
        after = tessiq.memory_stats(tessiq.gpu)
        # The result, and at most one block of workspace, which is not still held.
        allocations = after.allocation_count - before.allocation_count
        assert allocations in (1, 2), (m, k, n, allocations)
        assert after.live_blocks == before.live_blocks + 1
        taken.append(allocations == 2)
        del out
    # Which algorithms want workspace is cuBLASLt's choice. On an RTX 5080 two of these do.
    if not any(taken):
        pytest.skip("cuBLASLt chose an algorithm with no workspace for every shape here")
