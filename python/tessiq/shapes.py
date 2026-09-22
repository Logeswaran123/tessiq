# Copyright 2026 The Tessiq Authors.
# SPDX-License-Identifier: Apache-2.0
"""Telling a graph with known shapes from one without.

Dynamo captures a graph with concrete shapes the first time it sees a function, and recompiles
with symbolic ones the second time a dimension changes. That is the default, not an opt-in: a
model called with two sequence lengths gets a symbolic graph on the second call without anyone
asking for one.

Symbolic here means more than "unknown". A dimension arrives as its own graph input, shapes are
expressions over those inputs such as ``(s33, s50, 32)``, strides are expressions too, and there
are nodes doing arithmetic on them. Nothing in this compiler models any of that, and a kernel is
compiled for one set of shapes with every index in it folded to a constant.
"""

from __future__ import annotations

import torch


def _is_symbolic(value: object) -> bool:
    """Whether `value` is a symbolic size, or a tensor with one in its shape or its strides."""
    if isinstance(value, torch.SymInt):
        return True
    if isinstance(value, torch.Tensor):
        return any(isinstance(extent, torch.SymInt) for extent in value.shape) or any(
            isinstance(step, torch.SymInt) for step in value.stride()
        )
    if isinstance(value, (list, tuple)):
        return any(_is_symbolic(item) for item in value)
    return False


def is_static(graph_module: torch.fx.GraphModule) -> bool:
    """Whether every value in `graph_module` has a shape known at compile time.

    False for a graph Dynamo captured with symbolic shapes, which the compiler declines rather than
    guessing at. Checked on the whole graph rather than on its inputs, because a symbolic dimension
    can enter through an operation that computes one.
    """
    return not any(_is_symbolic(node.meta.get("val")) for node in graph_module.graph.nodes)
