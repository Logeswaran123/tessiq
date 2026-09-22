# Copyright 2026 The Tessiq Authors.
# SPDX-License-Identifier: Apache-2.0
"""Which operations are rewritten before the compiler sees them.

A decomposition is the cheapest way to cover an operation: instead of a kernel, it becomes the
operations the backend already runs. `core_aten_decompositions()` does most of it, and leaves a few
whole that are worth breaking up here because every piece they break into is something tessiq has.

The reverse is also possible and is a trap. Decomposing further than this reaches `prims`, a lower
operation set with its own names, and a model that was seven operations short becomes one that is
short a whole namespace.
"""

from __future__ import annotations

from typing import Any

import torch
from torch._decomp import core_aten_decompositions, get_decompositions

#: Left whole by core ATen and broken up here.
#:
#: A softmax is a max, a subtract, an exp, a sum and a divide. A layer norm is a mean, a variance
#: and an elementwise pass. Every one of those runs on the device, so the two operations that did
#: not are worth more as their parts.
_EXTRA = [
    torch.ops.aten._softmax,
    torch.ops.aten._softmax_backward_data,
    torch.ops.aten.native_layer_norm,
]


def _var_mean(
    self: torch.Tensor,
    dim: list[int] | None = None,
    *,
    correction: float | None = None,
    keepdim: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The variance and the mean, written in operations the backend claims.

    torch's own decomposition goes to `prims`, which is further down than anything else here. This
    stays in ATen: a mean, a squared difference, a sum, and a divide by however many elements went
    into it less the correction.
    """
    axes = list(range(self.dim())) if dim is None else list(dim)
    count = 1
    for axis in axes:
        count *= self.shape[axis]

    mean = torch.mean(self, dim=axes, keepdim=True)
    difference = self - mean
    total = torch.sum(difference * difference, dim=axes, keepdim=keepdim)
    divisor = float(count) - (1.0 if correction is None else float(correction))

    return total / divisor, mean if keepdim else torch.mean(self, dim=axes, keepdim=False)


def decompositions() -> dict[Any, Any]:
    """The table the backend hands AOTAutograd.

    Built fresh each time. AOTAutograd stores its keyword arguments and rewrites them on first use,
    so a shared dictionary would be modified under whoever holds it.
    """
    table = dict(core_aten_decompositions())
    table.update(get_decompositions(_EXTRA))
    table[torch.ops.aten.var_mean.correction] = _var_mean
    return table
