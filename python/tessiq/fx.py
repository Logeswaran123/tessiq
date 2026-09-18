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


def run_passes(ir: str, pipeline: str) -> str:
    """Run a named pass pipeline over textual IR and return the result.

    Text in and text out, for tests and for looking at what a pass did. The compiler itself keeps
    the graph in memory rather than round-tripping through text.

    Args:
        ir: A graph in tessiq's textual IR.
        pipeline: Comma-separated pass names, for example ``"mark-supported,cluster"``.

    Returns:
        The graph after the pipeline, in textual IR.

    Raises:
        ValueError: If the IR does not parse, or a pass rejects the graph.
        KeyError: If the pipeline names a pass that does not exist.
    """
    return str(_C.run_passes(ir, pipeline))


def pass_names() -> list[str]:
    """Names of the registered passes, sorted."""
    return list(_C.pass_names())
