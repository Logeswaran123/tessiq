# Copyright 2026 The Tessiq Authors.
# SPDX-License-Identifier: Apache-2.0
"""A launch goes to the stream it is handed.

Checked with a CUDA graph capture on a side stream. Work queued on a capturing stream is recorded
rather than run, so the output is overwritten after the capture and before the replay: a kernel
that went to the stream it was given writes the right answer when the graph replays. One that
ignored the stream ran on the spot during the capture, left the graph empty, and the replay leaves
the overwritten values where they are.

The side stream is non-blocking, as every stream torch creates is, so a launch on the legacy
default stream during the capture is not an error. That is why the output has to be overwritten.
Without it the stray launch writes the right answer early and the test passes either way.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
import torch
from torch._decomp import core_aten_decompositions
from torch._dynamo.backends.common import aot_autograd

import tessiq
from tessiq.recipe import build_recipe

pytestmark = pytest.mark.cuda


@pytest.fixture(autouse=True)
def _needs_gpu() -> None:
    if tessiq.gpu_count() == 0 or not torch.cuda.is_available():
        pytest.skip("needs a gpu that torch can also see")
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


def replayed(kernel: Any, inputs: list[torch.Tensor], out: torch.Tensor) -> torch.Tensor:
    """Launches `kernel` inside a graph captured on a side stream, then replays the graph."""
    # Once outside the capture first. A library handle is created on first use, and creating one
    # allocates, which a capture does not allow.
    kernel.launch(inputs, out)
    torch.cuda.synchronize()

    side = torch.cuda.Stream()  # type: ignore[no-untyped-call]
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(side):
        graph.capture_begin()
        kernel.launch(inputs, out, stream=side.cuda_stream)
        graph.capture_end()
    torch.cuda.synchronize()
    out.copy_(torch.full(out.shape, float("nan")))

    graph.replay()
    torch.cuda.synchronize()
    return out


def test_a_fused_kernel_goes_to_the_stream_it_is_handed() -> None:
    def chain(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(torch.relu(a * b))

    a, b = torch.randn(4096), torch.randn(4096)
    recipe = build_recipe(captured(chain, a, b), tessiq.gpu, use_cache=False)
    (kernel,) = recipe.kernels.values()

    out = torch.empty(4096, device=tessiq.gpu)
    inputs = [a.to(tessiq.gpu), b.to(tessiq.gpu)]
    torch.testing.assert_close(replayed(kernel, inputs, out).cpu(), chain(a, b))


def test_a_matmul_goes_to_the_stream_it_is_handed() -> None:
    def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a @ b

    a, b = torch.randn(64, 32), torch.randn(32, 48)
    recipe = build_recipe(captured(matmul, a, b), tessiq.gpu, use_cache=False)
    (kernel,) = recipe.kernels.values()

    out = torch.empty(64, 48, device=tessiq.gpu)
    inputs = [a.to(tessiq.gpu), b.to(tessiq.gpu)]
    torch.testing.assert_close(replayed(kernel, inputs, out).cpu(), a @ b)


def test_precompiled_host_code_goes_to_the_stream_it_is_handed() -> None:
    if "tessiq::layer_norm" not in tessiq.library.precompiled():
        pytest.skip("this build had no nvcc, so layer_norm has no host code")

    def normalised(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = torch.ops.tessiq.layer_norm(x, weight, bias, 1e-5)
        return out

    x, weight, bias = torch.randn(64, 128), torch.randn(128), torch.randn(128)
    recipe = build_recipe(captured(normalised, x, weight, bias), tessiq.gpu, use_cache=False)
    (kernel,) = recipe.kernels.values()

    out = torch.empty(64, 128, device=tessiq.gpu)
    inputs = [x.to(tessiq.gpu), weight.to(tessiq.gpu), bias.to(tessiq.gpu)]
    expected = torch.nn.functional.layer_norm(x, (128,), weight, bias, 1e-5)
    torch.testing.assert_close(replayed(kernel, inputs, out).cpu(), expected, rtol=1e-5, atol=1e-5)
