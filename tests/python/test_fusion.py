# Copyright 2026 The Tessiq Authors.
# SPDX-License-Identifier: Apache-2.0
"""Fusion on graphs PyTorch actually produces.

The golden files pin the pass against IR written by hand, using whichever operation names were
typed there. Only a real import proves those names match what `torch.compile` emits, which is the
failure this file exists to catch: an operation missing from the elementwise table does not error,
it quietly stops a chain fusing.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

import pytest
import torch
from torch._decomp import core_aten_decompositions
from torch._dynamo.backends.common import aot_autograd

import tessiq

PIPELINE = "mark-supported,cluster,fuse-elementwise"


def fused(fn: Callable[..., Any], *inputs: torch.Tensor) -> str:
    captured: list[torch.fx.GraphModule] = []

    def backend(graph_module: torch.fx.GraphModule, _inputs: object) -> Callable[..., object]:
        captured.append(graph_module)
        return graph_module.forward

    torch._dynamo.reset()
    torch.compile(
        fn,
        backend=aot_autograd(fw_compiler=backend, decompositions=core_aten_decompositions()),
        dynamic=False,
    )(*inputs)
    return tessiq.run_passes(tessiq.import_graph(captured[0]), PIPELINE)


def groups(ir: str) -> dict[int, int]:
    """How many operations landed in each fusion group."""
    counts: dict[int, int] = {}
    for match in re.finditer(r"tessiq\.fusion = (\d+)", ir):
        counts[int(match.group(1))] = counts.get(int(match.group(1)), 0) + 1
    return counts


def test_the_milestone_chain_is_one_group() -> None:
    ir = fused(
        lambda a, b, c: torch.sigmoid(torch.relu(a * b + c)),
        torch.randn(1024),
        torch.randn(1024),
        torch.randn(1024),
    )
    assert groups(ir) == {0: 4}, ir


def test_a_long_chain_stays_one_group() -> None:
    def model(a: torch.Tensor) -> torch.Tensor:
        x = a
        for _ in range(10):
            x = torch.tanh(torch.sigmoid(torch.relu(x)))
        return x

    ir = fused(model, torch.randn(256))
    assert groups(ir) == {0: 30}, ir


def test_every_supported_operation_fuses_under_its_aten_name() -> None:
    # One chain touching each operation the backend claims, under the spelling torch emits. A
    # missing entry in the elementwise table shows up as a smaller group rather than an error,
    # which is why the count is asserted and not just the shape.
    def model(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        x = a + b
        x = x - b
        x = x * b
        x = x / b
        x = torch.abs(x)
        x = -x
        x = torch.relu(x)
        x = torch.sigmoid(x)
        return torch.tanh(x)

    ir = fused(model, torch.randn(64), torch.rand(64) + 1.0)
    assert groups(ir) == {0: 9}, ir


def test_an_elementwise_operation_the_backend_does_not_claim_splits_the_group() -> None:
    # sqrt is elementwise, so it could fuse in principle, and the backend has no kernel for it, so
    # it is not in a cluster. Those are two different tables answering two different questions,
    # and this is what the difference looks like: the chain breaks where support ends, not where
    # fusability does.
    def model(a: torch.Tensor) -> torch.Tensor:
        return torch.tanh(torch.relu(torch.sqrt(torch.abs(a) + 1.0)))

    ir = fused(model, torch.randn(64))
    assert "aten.sqrt.default" in ir
    assert re.search(r"aten\.sqrt\.default[^\n]*tessiq\.supported = false", ir), ir
    counts = groups(ir)
    assert len(counts) == 2, ir


def test_a_matmul_splits_the_chain() -> None:
    def model(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(torch.tanh(torch.relu(a + b) @ b))

    ir = fused(model, torch.randn(4, 4), torch.randn(4, 4))
    counts = groups(ir)
    assert len(counts) == 2, ir
    assert sorted(counts.values()) == [2, 2], ir


def test_fusion_runs_in_the_compile_path() -> None:
    # The backend's pipeline includes fusion now, so the IR a compile produces carries the groups
    # and a recipe can turn them into kernels.
    ir, clusters = tessiq.partition(
        _capture(lambda a, b: torch.relu(a + b) * b, torch.randn(8), torch.randn(8))
    )
    assert clusters
    assert "tessiq.fusion" in ir


def _capture(fn: Callable[..., Any], *inputs: torch.Tensor) -> torch.fx.GraphModule:
    captured: list[torch.fx.GraphModule] = []

    def backend(graph_module: torch.fx.GraphModule, _inputs: object) -> Callable[..., object]:
        captured.append(graph_module)
        return graph_module.forward

    torch._dynamo.reset()
    torch.compile(fn, backend=aot_autograd(fw_compiler=backend), dynamic=False)(*inputs)
    return captured[0]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_dtype_does_not_prevent_fusion(dtype: torch.dtype) -> None:
    ir = fused(
        lambda a, b: torch.relu(a + b) * b,
        torch.randn(32, dtype=dtype),
        torch.randn(32, dtype=dtype),
    )
    assert groups(ir) == {0: 3}, ir
