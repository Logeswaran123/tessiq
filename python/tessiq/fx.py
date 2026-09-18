# Copyright 2026 The Tessiq Authors.
# SPDX-License-Identifier: Apache-2.0
"""Converting PyTorch FX graphs into tessiq IR."""

from __future__ import annotations

import torch

from tessiq import _C


def import_graph(graph_module: torch.fx.GraphModule, name: str = "forward") -> str:
    """Convert an FX graph into tessiq IR and return its textual form.

    The graph is expected to have been through AOTAutograd, which functionalises it and decomposes
    it to core ATen. The passes assume no operation mutates its operands, which holds for a
    functionalised graph and not for a raw one.

    Every node is imported, including operations tessiq has no kernel for. Deciding what to keep is
    partitioning, which runs later over the imported graph.

    Args:
        graph_module: The module a ``torch.compile`` backend receives.
        name: The name the resulting graph carries.

    Returns:
        The graph in tessiq's textual IR.

    Raises:
        ValueError: If a node carries no shape information, so its type is unknown.
        TypeError: If the graph uses a dtype or a node kind the IR cannot represent.
    """
    return str(_C.import_fx(graph_module, name))
