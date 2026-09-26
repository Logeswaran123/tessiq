# Copyright 2026 The Tessiq Authors.
# SPDX-License-Identifier: Apache-2.0
"""A precompiled custom operation: `layer_norm` from csrc/custom_ops, whose host code nvcc compiles.

Its host code, layer_norm.nvcc.cu, launches two kernels with each row's statistics held in
workspace between them. These check that it matches its reference called directly and in a
compiled graph, that it declines what its kernels cannot take, that its workspace comes from
tessiq's allocator and goes back to it, and that a build without nvcc still runs the reference.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
import torch
from torch._dynamo.backends.common import aot_autograd

import tessiq
from tessiq import _C
from tessiq.decompositions import decompositions
from tessiq.recipe import build_recipe

layer_norm = torch.ops.tessiq.layer_norm

EPS = 1e-5


def expected(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.layer_norm(x, (x.shape[-1],), weight, bias, EPS)


def fallbacks() -> int:
    return tessiq.library.fallbacks("tessiq::layer_norm")


def arguments(*shape: int, device: torch.device | None = None) -> tuple[torch.Tensor, ...]:
    x, weight, bias = torch.randn(*shape), torch.randn(shape[-1]), torch.randn(shape[-1])
    if device is None:
        return x, weight, bias
    return x.to(device), weight.to(device), bias.to(device)


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


def normalised(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    out: torch.Tensor = layer_norm(x, weight, bias, EPS)
    return out


@pytest.fixture
def host_code() -> None:
    if tessiq.gpu_count() == 0:
        pytest.skip("needs a gpu")
    if "tessiq::layer_norm" not in tessiq.library.precompiled():
        pytest.skip("this build had no nvcc, so layer_norm has no host code")


@pytest.fixture(autouse=True)
def _fresh() -> None:
    torch._dynamo.reset()


def test_on_the_host_it_is_the_reference() -> None:
    x, weight, bias = arguments(4, 64)
    torch.testing.assert_close(layer_norm(x, weight, bias, EPS), expected(x, weight, bias))


def test_on_the_reference_device_it_runs_the_reference() -> None:
    x, weight, bias = arguments(4, 64)
    before = fallbacks()
    out = layer_norm(*(t.to(tessiq.cpu) for t in (x, weight, bias)), EPS)
    assert fallbacks() == before + 1
    torch.testing.assert_close(out.cpu(), expected(x, weight, bias))


@pytest.mark.cuda
@pytest.mark.usefixtures("host_code")
@pytest.mark.parametrize("shape", [(4, 32), (3, 5, 300), (1, 1), (257, 1000)])
def test_on_a_gpu_it_runs_its_host_code(shape: tuple[int, ...]) -> None:
    x, weight, bias = arguments(*shape)
    before = fallbacks()
    out = layer_norm(*(t.to(tessiq.gpu) for t in (x, weight, bias)), EPS)
    assert fallbacks() == before
    torch.testing.assert_close(out.cpu(), expected(x, weight, bias), rtol=1e-5, atol=1e-5)


@pytest.mark.cuda
@pytest.mark.usefixtures("host_code")
def test_a_weight_that_broadcasts_is_declined_and_runs_the_reference() -> None:
    # The reference broadcasts a single weight and bias, but the kernel reads one of each per
    # column, so the host code hands the call back.
    x, weight, bias = torch.randn(4, 64), torch.randn(1), torch.randn(1)
    before = fallbacks()
    out = layer_norm(*(t.to(tessiq.gpu) for t in (x, weight, bias)), EPS)
    assert fallbacks() == before + 1
    reference = layer_norm(x, weight, bias, EPS)
    torch.testing.assert_close(out.cpu(), reference, rtol=1e-5, atol=1e-5)


@pytest.mark.cuda
@pytest.mark.usefixtures("host_code")
def test_a_type_it_does_not_declare_runs_the_reference() -> None:
    x, weight, bias = (t.double() for t in arguments(4, 64))
    before = fallbacks()
    out = layer_norm(*(t.to(tessiq.gpu) for t in (x, weight, bias)), EPS)
    assert fallbacks() == before + 1
    torch.testing.assert_close(out.cpu(), expected(x, weight, bias))


@pytest.mark.cuda
@pytest.mark.usefixtures("host_code")
def test_in_a_compiled_graph_it_is_a_kernel_of_its_own() -> None:
    def model(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        return torch.relu(layer_norm(x * 2.0, weight, bias, EPS))

    x, weight, bias = arguments(8, 128)
    recipe = build_recipe(captured(model, x, weight, bias), tessiq.gpu, use_cache=False)
    kinds = sorted(type(kernel).__name__ for kernel in recipe.kernels.values())
    assert kinds == ["CustomKernel", "FusedKernel", "FusedKernel"], kinds

    before = fallbacks()
    compiled = torch.compile(model, backend="tessiq", options={"device": str(tessiq.gpu)})
    out = compiled(x, weight, bias)
    assert fallbacks() == before
    torch.testing.assert_close(out, model(x, weight, bias), rtol=1e-5, atol=1e-5)


@pytest.mark.cuda
@pytest.mark.usefixtures("host_code")
def test_its_workspace_comes_from_tessiqs_allocator_and_goes_back() -> None:
    x, weight, bias = arguments(4096, 64, device=tessiq.gpu)
    layer_norm(x, weight, bias, EPS)
    tessiq.synchronize(tessiq.gpu)
    before = tessiq.memory_stats(tessiq.gpu)

    out = layer_norm(x, weight, bias, EPS)
    after = tessiq.memory_stats(tessiq.gpu)
    # Two blocks were taken, the result and the workspace holding each row's statistics, and only
    # the result is still held.
    assert after.allocation_count - before.allocation_count == 2
    assert after.live_blocks == before.live_blocks + 1
    del out


@pytest.mark.cuda
@pytest.mark.usefixtures("host_code")
def test_on_another_stream_its_workspace_goes_back_once_the_stream_finishes() -> None:
    if not torch.cuda.is_available():
        pytest.skip("needs torch to see the gpu, to make a stream")
    x, weight, bias = arguments(512, 256)
    recipe = build_recipe(captured(normalised, x, weight, bias), tessiq.gpu, use_cache=False)
    kernel = next(k for k in recipe.kernels.values() if isinstance(k, _C.CustomKernel))

    inputs = [t.to(tessiq.gpu) for t in (x, weight, bias)]
    out = torch.empty(512, 256, device=tessiq.gpu)
    before = tessiq.memory_stats(tessiq.gpu).live_blocks
    side = torch.cuda.Stream()  # type: ignore[no-untyped-call]
    kernel.launch(inputs, out, stream=side.cuda_stream)
    assert tessiq.memory_stats(tessiq.gpu).live_blocks == before
    side.synchronize()
    torch.testing.assert_close(out.cpu(), expected(x, weight, bias), rtol=1e-5, atol=1e-5)
