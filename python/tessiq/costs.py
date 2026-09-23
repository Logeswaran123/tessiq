# Copyright 2026 The Tessiq Authors.
# SPDX-License-Identifier: Apache-2.0
"""What each kernel costs the device, measured without the measurement getting in the way.

The profiler can mark the stream around a kernel inside a real call, and on this card a mark costs
about what a small kernel costs. So a per-kernel reading taken that way carries a mark's worth of
bias, which is fine for ranking kernels against each other and not fine for adding them up and
comparing the total against anything.

This takes the other approach. Each kernel is launched many times between one pair of marks, so
the cost of marking is divided by however many launches it covered and stops mattering. What it
gives up is the real call's interleaving: these are kernels run back to back on a device that is
never waiting, which is the right number for "what does our generated code cost" and the wrong one
for "what happens in a call".

Both are needed, and they answer different questions. This one answers whether the kernels are the
problem.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from tessiq import _C
from tessiq.device import cuda_ordinal
from tessiq.recipe import Recipe

#: Launches per measurement. Chosen so that one pair of marks, about sixteen microseconds, is
#: under a percent of what it covers for even the smallest kernel here.
DEFAULT_REPEATS = 200


@dataclass(frozen=True)
class KernelCost:
    """What one kernel costs, and what it was measured over.

    Attributes:
        name: The graph node whose value the kernel writes.
        kind: ``fused``, ``reduction`` or ``gemm``.
        microseconds: Device time for one launch, averaged over `repeats` back to back.
        repeats: How many launches the pair of marks covered.
    """

    name: str
    kind: str
    microseconds: float
    repeats: int


def _tensor_like(node: torch.fx.Node, device: torch.device, *, packed: bool) -> torch.Tensor:
    """A tensor shaped and laid out the way this node's value will be.

    Taken from the fake tensor rather than from a run, so nothing has to be executed to find out
    what a kernel will be handed. `packed` forces the contiguous layout, which is what a fused
    loop and a reduction require of their inputs and what the executor gives them. A matmul is the
    exception: it reads the layout it was given, so a transposed operand has to stay transposed or
    the measurement is of a different call.
    """
    meta = node.meta["val"]
    if packed or meta.is_contiguous():
        return torch.empty(meta.shape, dtype=meta.dtype, device=device)
    return torch.empty_strided(meta.shape, meta.stride(), dtype=meta.dtype, device=device)


def _kind_of(kernel: object) -> str:
    return type(kernel).__name__.removesuffix("Kernel").lower()


def kernel_costs(recipe: Recipe, *, repeats: int = DEFAULT_REPEATS) -> list[KernelCost]:
    """Device time per launch for every kernel in `recipe`, slowest first.

    Nothing is run beforehand: the tensors come from the shapes the compiler already recorded, so
    what a kernel reads is uninitialised memory of the right size and layout. That is enough,
    because a kernel's cost here is bandwidth and occupancy rather than the values in it.

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
            # A kernel whose operands this cannot reconstruct is left out. Reporting it as free
            # would be worse than not reporting it.
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
