# Copyright 2026 The Tessiq Authors.
# SPDX-License-Identifier: Apache-2.0
"""Every kind of compiled work behind one interface.

The executor asks a kernel what it reads, what it writes, how big the result is and how to run it,
and nothing else. These check that every kind answers those questions the same way, so that a new
kind can be added without the executor learning it exists.
"""

from __future__ import annotations

import pytest
import torch

import tessiq
from tessiq import _C
from tessiq.fx import capture
from test_models import EncoderBlock, ResidualBlock

pytestmark = pytest.mark.cuda

_SUBCLASS = {
    "fused": _C.FusedKernel,
    "reduction": _C.ReductionKernel,
    "gemm": _C.GemmKernel,
    "conv": _C.ConvKernel,
    # The encoder's softmax, which a custom operation replaces.
    "custom": _C.CustomKernel,
}


@pytest.fixture(autouse=True)
def _needs_gpu() -> None:
    if tessiq.gpu_count() == 0:
        pytest.skip("needs a gpu")
    torch._dynamo.reset()


def kernels_of(model: torch.nn.Module, example: torch.Tensor) -> list[_C.Kernel]:
    graph = capture(model.eval(), example)
    recipe = tessiq.build_recipe(graph, tessiq.gpu, use_cache=False)
    return list(recipe.kernels.values())


def compiled() -> list[_C.Kernel]:
    return kernels_of(EncoderBlock(), torch.randn(2, 16, 32)) + kernels_of(
        ResidualBlock(), torch.randn(1, 16, 8, 8)
    )


def test_the_two_blocks_between_them_use_every_kind() -> None:
    assert {kernel.kind for kernel in compiled()} == set(_SUBCLASS)


def test_each_kind_comes_back_as_its_own_class() -> None:
    for kernel in compiled():
        assert isinstance(kernel, _C.Kernel)
        assert isinstance(kernel, _SUBCLASS[kernel.kind]), kernel.kind


def test_every_kernel_computes_at_least_its_output() -> None:
    for kernel in compiled():
        assert kernel.output in kernel.members, kernel.kind
        assert kernel.inputs, kernel.kind


def test_only_a_fused_loop_computes_more_than_its_output() -> None:
    for kernel in compiled():
        if kernel.kind != "fused":
            assert kernel.members == [kernel.output], kernel.kind
