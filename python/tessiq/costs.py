# Copyright 2026 The Tessiq Authors.
# SPDX-License-Identifier: Apache-2.0
"""What each kernel costs the device, measured in isolation.

A stream mark costs about as much as a small kernel, so the profiler's per-kernel timings inside a
real call carry that bias. Here each kernel is launched many times between one pair of marks, which
spreads their cost over every launch. The kernels then run back to back on an idle device rather
than interleaved as in a call, so this measures the generated code, not the call.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from tessiq import _C
from tessiq.device import cuda_ordinal
from tessiq.recipe import Recipe

#: Launches per measurement, enough that the pair of marks is a small part of what it covers.
DEFAULT_REPEATS = 200


@dataclass(frozen=True)
class KernelCost:
    """What one kernel costs, and what it was measured over.

    Attributes:
        name: The graph node whose value the kernel writes.
        kind: ``fused``, ``reduction``, ``gemm``, ``conv`` or ``custom``.
        microseconds: Device time for one launch, averaged over `repeats` back to back.
        repeats: How many launches the pair of marks covered.
    """

    name: str
    kind: str
    microseconds: float
    repeats: int


def _tensor_like(node: torch.fx.Node, device: torch.device, *, packed: bool) -> torch.Tensor:
    """A tensor shaped and laid out the way this node's value will be.

    Taken from the fake tensor, so nothing has to run. `packed` forces the contiguous layout the
    executor gives a kernel that packs its inputs; otherwise the fake tensor's layout is kept, so a
    matmul's transposed operand stays transposed.
    """
    meta = node.meta["val"]
    if packed or meta.is_contiguous():
        return torch.empty(meta.shape, dtype=meta.dtype, device=device)
    return torch.empty_strided(meta.shape, meta.stride(), dtype=meta.dtype, device=device)


def _kind_of(kernel: _C.Kernel) -> str:
    return kernel.kind


def kernel_costs(recipe: Recipe, *, repeats: int = DEFAULT_REPEATS) -> list[KernelCost]:
    """Device time per launch for every kernel in `recipe`, slowest first.

    The inputs are uninitialised tensors of the recorded shapes and layouts, which is enough since
    the cost does not depend on the values.

    Args:
        recipe: A compiled graph. Its device has to be a GPU, since this measures device time.
        repeats: Launches per measurement.

    Returns:
        One entry per kernel that could be measured, longest first. A kernel whose inputs cannot be
        built, or which refuses the tensors it is handed, is left out rather than reported as zero.

    Raises:
        ValueError: If the recipe was not compiled for a GPU, or the build cannot time a device.
    """
    on_a_gpu = [d for d in recipe.devices.values() if cuda_ordinal(d) >= 0]
    if not on_a_gpu or not _C.device_timing_available():
        raise ValueError("kernel costs need a recipe compiled for a gpu, and a build with CUDA")
    device = on_a_gpu[0]
    ordinal = cuda_ordinal(device)

    nodes = {node.name: node for node in recipe.graph_module.graph.nodes}
    costs: list[KernelCost] = []
    for name, kernel in recipe.kernels.items():
        try:
            inputs = [
                _tensor_like(nodes[i], device, packed=kernel.packs_inputs) for i in kernel.inputs
            ]
            out = _tensor_like(nodes[name], device, packed=True)
            # Warm up past the first launch, which pays for whatever the driver does once.
            for _ in range(3):
                kernel.launch(inputs, out)
        except (KeyError, TypeError, ValueError):
            # Left out rather than reported as free.
            continue

        timer = _C.DeviceTimer(ordinal)
        timer.start()
        for _ in range(repeats):
            kernel.launch(inputs, out)
        timer.stop()
        costs.append(
            KernelCost(
                name=name,
                kind=_kind_of(kernel),
                microseconds=timer.microseconds() / repeats,
                repeats=repeats,
            )
        )

    return sorted(costs, key=lambda cost: -cost.microseconds)
