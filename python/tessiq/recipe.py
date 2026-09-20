# Copyright 2026 The Tessiq Authors.
# SPDX-License-Identifier: Apache-2.0
"""A compiled graph, and what it takes to run one.

A recipe is the answer to every question about running a graph that does not depend on the actual
tensors: which device each operation runs on, where the arguments have to be moved and where they
already are, how big the arena is, and what shape the caller gets its results back in. Working
those out once rather than per call is the point.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from tessiq import _C
from tessiq.cache import lookup, signature, store
from tessiq.device import cpu

#: The cluster of a step the backend does not claim, matching `tessiq::ir::kHost`.
HOST = -1


@dataclass(frozen=True)
class Recipe:
    """A graph lowered for one device.

    Attributes:
        graph_module: The graph the recipe was built from. Steps name its nodes.
        devices: Where each node runs, keyed by node name. A node absent from this map runs on the
            host, which is also what an empty map means.
        needs_move: Nodes whose tensor arguments have to be moved before they run. A node in the
            middle of a cluster finds its operands where the previous step left them, so moving
            them again would be two transfers that cancel out.
        output_spec: One entry per value the graph returns: the producing node's name, or ``None``
            where the graph returns nothing. A backward graph returns ``None`` at every position
            whose input takes no gradient, and the autograd engine matches results to inputs by
            position, so the shape of the returned tuple is part of the calling convention rather
            than something the caller can work out.
        arena_bytes: What one allocation holding every intermediate would need.
        naive_bytes: What the same intermediates would cost with no reuse.
    """

    graph_module: torch.fx.GraphModule
    devices: dict[str, torch.device]
    needs_move: frozenset[str]
    output_spec: tuple[str | None, ...]
    arena_bytes: int
    naive_bytes: int


def _output_spec(graph_module: torch.fx.GraphModule) -> tuple[str | None, ...]:
    """Read the shape of the returned tuple off the FX graph.

    Taken from FX rather than from the IR on purpose. The IR records the values a graph computes,
    and a position that returns nothing computes nothing, so the IR cannot say how many results
    there are. See docs/design_notes/0002-output-structure-is-calling-convention.md.
    """
    for node in graph_module.graph.nodes:
        if node.op != "output":
            continue
        returned = node.args[0]
        if not isinstance(returned, (tuple, list)):
            returned = (returned,)
        return tuple(value.name if isinstance(value, torch.fx.Node) else None for value in returned)
    raise ValueError("graph has no output node")


def build_recipe(
    graph_module: torch.fx.GraphModule,
    device: torch.device | None = None,
    name: str = "forward",
    *,
    use_cache: bool = True,
) -> Recipe:
    """Compile an FX graph for `device`.

    Args:
        graph_module: The module a ``torch.compile`` backend receives.
        device: Where clustered operations run. Defaults to ``tessiq.cpu``.
        name: The name the resulting IR graph carries.
        use_cache: Whether to consult the recipe cache. False forces a compile, which is what a
            benchmark measuring compilation wants and what nothing else should.

    Returns:
        A recipe that can be run any number of times, from any thread. A cached recipe is shared
        between callers, which is safe because a recipe never changes after it is built.

    Raises:
        ValueError: If a node carries no shape information, if a shape is not static, or if the
            graph has no output node.
        TypeError: If the graph uses a dtype or node kind the IR cannot represent.
    """
    target = device if device is not None else cpu

    key = signature(graph_module, target) if use_cache else ""
    if use_cache:
        cached = lookup(key)
        if cached is not None:
            return cached

    recipe = _compile(graph_module, target, name)
    if use_cache:
        store(key, recipe)
    return recipe


def _compile(graph_module: torch.fx.GraphModule, target: torch.device, name: str) -> Recipe:
    host = torch.device("cpu")
    native = _C.build_recipe_fx(graph_module, name, 512)

    devices: dict[str, torch.device] = {}
    for step in native.steps:
        if not step.results:
            continue
        devices[step.results[0]] = host if step.cluster == HOST else target

    # A node needs a move only when something it reads is somewhere else. Everything that is not a
    # step is a graph input, which the caller hands us on the host.
    needs_move: set[str] = set()
    for node in graph_module.graph.nodes:
        if node.op != "call_function":
            continue
        here = devices.get(node.name, host)
        sources = [devices.get(other.name, host) for other in node.all_input_nodes]
        if any(source != here for source in sources) or (not sources and here != host):
            needs_move.add(node.name)

    return Recipe(
        graph_module=graph_module,
        devices=devices,
        needs_move=frozenset(needs_move),
        output_spec=_output_spec(graph_module),
        arena_bytes=native.arena_bytes,
        naive_bytes=native.naive_bytes,
    )
