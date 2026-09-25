# Copyright 2026 The Tessiq Authors.
# SPDX-License-Identifier: Apache-2.0
"""Which operations are rewritten before the compiler sees them.

A decomposition covers an operation by rewriting it into ones the backend already runs.
`core_aten_decompositions()` does most of it; this adds a few it leaves whole. Decomposing further
would reach `prims`, a lower operation set tessiq does not cover.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from torch._decomp import core_aten_decompositions, get_decompositions
from torch.utils import _pytree

from tessiq import _C

#: Left whole by core ATen and broken up here.
#:
#: A softmax is a max, a subtract, an exp, a sum and a divide. A layer norm is a mean, a variance
#: and an elementwise pass. Each part runs on the device.
_EXTRA = [
    torch.ops.aten._softmax,
    torch.ops.aten._softmax_backward_data,
    torch.ops.aten.native_layer_norm,
    # In eval mode the running statistics are fixed, so a batch norm is a subtract, a reciprocal
    # square root, a scale and a shift.
    torch.ops.aten._native_batch_norm_legit_no_training,
    # The first rewrite lands on this one rather than on leaves, so both spellings are asked for.
    torch.ops.aten._native_batch_norm_legit,
]


def _var_mean(
    self: torch.Tensor,
    dim: list[int] | None = None,
    *,
    correction: float | None = None,
    keepdim: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The variance and the mean, written in operations the backend claims.

    torch's own decomposition goes to `prims`. This stays in ATen: a mean, a squared difference, a
    sum, and a divide by the element count less the correction.
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


def _unless_replaced(
    decomposition: Callable[..., Any], dtypes: frozenset[torch.dtype]
) -> Callable[..., Any]:
    """`decomposition`, except where a custom operation replaces the operation it decomposes.

    There it returns NotImplemented, which keeps the operation whole so the `override-<name>` pass
    can replace it. That is where every tensor has a type the custom operation declares, the same
    test the pass makes.
    """

    def decompose(*args: Any, **kwargs: Any) -> Any:
        tensors = [a for a in _pytree.tree_leaves((args, kwargs)) if isinstance(a, torch.Tensor)]
        if tensors and all(t.dtype in dtypes for t in tensors):
            return NotImplemented
        return decomposition(*args, **kwargs)

    return decompose


def _operation(spelling: str) -> Any:
    """`aten._softmax.default` as the operation object torch keys decompositions on."""
    namespace, rest = spelling.split(".", 1)
    name, overload = rest.rsplit(".", 1)
    return getattr(getattr(getattr(torch.ops, namespace), name), overload)


def decompositions() -> dict[Any, Any]:
    """The table the backend hands AOTAutograd.

    Built fresh each time, because AOTAutograd modifies the dictionary it is given.

    An operation a custom operation replaces is kept whole wherever the custom operation has the
    types for it, and decomposed as before everywhere else.
    """
    table = dict(core_aten_decompositions())
    table.update(get_decompositions(_EXTRA))
    table[torch.ops.aten.var_mean.correction] = _var_mean
    for _op, replaced, dtypes in _C.custom_op_overrides():
        operation = _operation(replaced)
        if operation in table:
            table[operation] = _unless_replaced(table[operation], frozenset(dtypes))
    return table
