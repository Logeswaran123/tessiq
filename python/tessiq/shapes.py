# Copyright 2026 The Tessiq Authors.
# SPDX-License-Identifier: Apache-2.0
"""Telling a graph with known shapes from one without.

Dynamo captures a graph with concrete shapes the first time it sees a function, and by default
recompiles with symbolic ones the second time a dimension changes.

In a symbolic graph a dimension arrives as its own graph input, shapes and strides are expressions
over those inputs such as ``(s33, s50, 32)``, and some nodes do arithmetic on them. The IR records
an unknown extent as ``?``. Elementwise loops, reductions and matmuls take their extents at launch.
A strided broadcast and a convolution need them when the kernel is built, so they run one
operation at a time.

This predicate is for logging and tests. Nothing decides whether to compile by asking it.
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

    False for a graph Dynamo captured with symbolic shapes. Checked on the whole graph rather than
    on its inputs, because a symbolic dimension can enter through an operation that computes one.
    """
    return not any(_is_symbolic(node.meta.get("val")) for node in graph_module.graph.nodes)
