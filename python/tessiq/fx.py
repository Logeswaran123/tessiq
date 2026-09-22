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


def partition(
    graph_module: torch.fx.GraphModule, name: str = "forward"
) -> tuple[str, dict[str, int]]:
    """Import an FX graph and decide what the backend takes.

    Args:
        graph_module: The module a ``torch.compile`` backend receives.
        name: The name the resulting graph carries.

    Returns:
        The partitioned graph in textual IR, and a map from FX node name to cluster number. A node
        absent from the map has no kernel and runs on the host.

    Raises:
        ValueError: If a node carries no shape information.
        TypeError: If the graph uses a dtype or node kind the IR cannot represent.
    """
    ir, clusters = _C.partition_fx(graph_module, name)
    return str(ir), dict(clusters)


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


def default_pipeline() -> list[str]:
    """The passes the backend runs to decide what it takes, in order.

    Read from the compiler rather than written down again here. A second copy of this list is a
    second thing that can be wrong, and the way it goes wrong is a tool describing a compilation
    that never happened.
    """
    return str(_C.default_pipeline()).split(",")


def capture(
    model: torch.nn.Module, example: torch.Tensor, *, dynamic: bool = False
) -> torch.fx.GraphModule:
    """The graph the backend would receive for `model` called with `example`.

    Runs Dynamo and AOTAutograd without compiling anything, so the result is the functionalised,
    decomposed graph the passes actually see. The decompositions are tessiq's own rather than core
    ATen's, which is what makes a softmax arrive as the five operations it is made of.

    Args:
        model: Any module. Put it in eval mode first unless training is what you mean to capture.
        example: One input, used to trace. Its values do not matter, only its shape and dtype.
        dynamic: True to capture with symbolic shapes, which is what Dynamo does by itself the
            second time a dimension changes.

    Returns:
        The graph module, with a fake tensor on every node's ``meta["val"]``.
    """
    from torch._dynamo.backends.common import aot_autograd

    from tessiq.decompositions import decompositions

    graphs: list[torch.fx.GraphModule] = []

    def collect(graph_module: torch.fx.GraphModule, _inputs: object) -> object:
        graphs.append(graph_module)
        return graph_module.forward

    torch._dynamo.reset()
    with torch.no_grad():
        torch.compile(
            model,
            backend=aot_autograd(fw_compiler=collect, decompositions=decompositions()),
            dynamic=dynamic,
        )(example)
    if not graphs:
        raise ValueError("tracing produced no graph; the model may have nothing to compile")
    return graphs[0]


def emit_cuda(ir: str) -> str:
    """CUDA source for every fusion group in `ir`, in group order.

    Args:
        ir: A graph in tessiq's textual IR, already through ``fuse-elementwise``. A graph with no
            fusion groups returns a comment saying so rather than an empty string.

    Returns:
        The source, one ``__global__`` function per group.

    Raises:
        ValueError: If the IR does not parse, or a group is not something one loop can express.
        TypeError: If a group uses a dtype the emitter has no spelling for.
    """
    return str(_C.emit_cuda(ir))
