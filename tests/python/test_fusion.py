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


def test_a_chain_of_transcendentals_is_one_group() -> None:
    # The same property as the chain above, over the operations added for coverage rather than the
    # original handful. Nothing here is special to fusion; the point is that widening the table
    # widened what fuses, without anything in the fusion pass changing.
    def model(a: torch.Tensor) -> torch.Tensor:
        x = torch.exp(a)
        x = torch.log1p(x)
        x = torch.sin(x)
        x = torch.cos(x)
        x = torch.atan(x)
        x = torch.erf(x)
        return torch.rsqrt(x + 2.0)

    ir = fused(model, torch.randn(64))
    assert groups(ir) == {0: 8}, ir


def test_a_matmul_splits_the_fused_chain_without_leaving_the_device() -> None:
    # A matmul is claimed and is not elementwise, so it ends the loop its neighbours share while
    # staying in their cluster. Two fusion groups, one cluster.
    def model(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(torch.tanh(torch.relu(a + b) @ b))

    ir = fused(model, torch.randn(4, 4), torch.randn(4, 4))
    counts = groups(ir)
    assert len(counts) == 2, ir
    assert sorted(counts.values()) == [2, 2], ir
    assert re.search(r"aten\.mm\.default[^\n]*tessiq\.supported = true", ir), ir


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


def fusion_group_of(ir: str, op: str) -> int | None:
    """The fusion group `op` landed in, or None if it was left out of every group."""
    for line in ir.splitlines():
        if op not in line:
            continue
        match = re.search(r"tessiq\.fusion = (\d+)", line)
        return int(match.group(1)) if match else None
    raise AssertionError(f"{op} is not in the graph:\n{ir}")


def test_a_reduction_ends_the_group_and_the_chain_before_it_still_fuses() -> None:
    # A reduction reads a whole axis to write one element, so it cannot share a loop that gives
    # each thread one index. What matters is that it ends the group without breaking the part in
    # front of it, which is where the work is.
    def model(a: torch.Tensor) -> torch.Tensor:
        return torch.sum(torch.relu(a * 2.0), dim=1)

    ir = fused(model, torch.randn(4, 8))
    assert groups(ir) == {0: 2}, ir
    assert fusion_group_of(ir, "aten.sum.dim_IntList") is None, ir
    assert fusion_group_of(ir, "aten.relu.default") == 0, ir


def test_a_reduction_between_two_chains_splits_them() -> None:
    def model(a: torch.Tensor) -> torch.Tensor:
        x = torch.relu(a * 2.0)
        s = torch.sum(x, dim=1, keepdim=True)
        return torch.tanh(torch.abs(s) + 1.0)

    ir = fused(model, torch.randn(4, 8))
    # Two groups either side of the reduction, of two and three operations.
    assert sorted(groups(ir).values()) == [2, 3], ir
    assert fusion_group_of(ir, "aten.sum.dim_IntList") is None, ir
    assert fusion_group_of(ir, "aten.mul.Tensor") != fusion_group_of(ir, "aten.tanh.default"), ir


def test_keepdim_is_carried_through_to_the_ir() -> None:
    # torch passes keepdim positionally, so it arrives as arg2 rather than under its own name.
    # The shape rule reads it from there, so it is worth pinning that it survives the import.
    def model(a: torch.Tensor) -> torch.Tensor:
        return torch.sum(a, dim=1, keepdim=True)

    ir = fused(model, torch.randn(4, 8))
    assert "arg1 = [1]" in ir, ir
    assert "arg2 = true" in ir, ir
    assert "f32[4,1]" in ir, ir


def test_a_movement_operation_joins_the_group_it_sits_in() -> None:
    # A view between two claimed operations is on the device with them, and costs nothing there.
    def model(a: torch.Tensor) -> torch.Tensor:
        return torch.relu(a * 2.0).reshape(2, 32).tanh()

    ir = fused(model, torch.randn(8, 8))
    assert "aten.view.default" in ir, ir
    assert re.search(r"aten\.view\.default[^\n]*tessiq\.cluster", ir), ir


def test_a_movement_operation_on_its_own_stays_on_the_host() -> None:
    # Claimed and not placed. Moving a tensor across to take a different view of it and moving
    # that back is two transfers for no work, so a group holding nothing but movement is left
    # where its input already was.
    def model(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.conv1d(a, b).reshape(4, 8)

    ir = fused(model, torch.randn(1, 4, 8), torch.randn(4, 4, 1))
    assert re.search(r"aten\.view\.default[^\n]*tessiq\.supported = true", ir), ir
    assert not re.search(r"aten\.view\.default[^\n]*tessiq\.cluster", ir), ir
