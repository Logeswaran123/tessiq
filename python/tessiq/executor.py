# Copyright 2026 The Tessiq Authors.
# SPDX-License-Identifier: Apache-2.0
"""Running a partitioned graph.

Clustered nodes run with their tensors on a tessiq device; everything else runs on the host. Until
kernels exist, a tessiq op reaches PyTorch's CPU kernel through the fallback, so this is slower
than not compiling at all. What it buys is that the whole path is exercised and checked against
eager, which is the agreement a lowered recipe will have to keep.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

from tessiq.device import cpu


def _move(value: Any, device: torch.device) -> Any:
    """Move a tensor to ``device``, leaving anything else alone.

    Arguments are not all tensors. Scalars, shapes, dtypes and ``None`` all arrive here, and
    ``.to()`` on them would fail. Nested lists and tuples turn up in ops like ``cat`` and ``stack``.
    """
    if isinstance(value, torch.Tensor):
        return value if value.device == device else value.to(device)
    if isinstance(value, (list, tuple)):
        return type(value)(_move(item, device) for item in value)
    return value


class PartitionedExecutor(torch.fx.Interpreter):
    """Executes an FX graph, sending clustered nodes to a tessiq device.

    A node in a cluster has its tensor arguments moved to the device before it runs, so the
    operation dispatches through tessiq. Results stay there, so the next node in the same cluster
    finds them already in place and no transfer happens between them. A node on the host has its
    arguments moved back.

    Not thread-safe, and not reusable. One instance per call.

    Args:
        graph_module: The module a ``torch.compile`` backend receives.
        clusters: Map from FX node name to cluster number, as ``tessiq.partition`` returns it. A
            node absent from the map runs on the host.
        device: Where clustered nodes run. Defaults to ``tessiq.cpu``.
    """

    def __init__(
        self,
        graph_module: torch.fx.GraphModule,
        clusters: dict[str, int],
        device: torch.device | None = None,
    ) -> None:
        super().__init__(graph_module)
        self._clusters = clusters
        self._device = device if device is not None else cpu
        self._node: torch.fx.Node | None = None

    def run_node(self, n: torch.fx.Node) -> Any:
        # Recorded because call_function is handed the target and the arguments, not the node, and
        # the cluster is a property of the node.
        self._node = n
        return super().run_node(n)

    def call_function(self, target: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        device = self._device_for(self._node)
        return super().call_function(
            target,
            tuple(_move(value, device) for value in args),
            {name: _move(value, device) for name, value in kwargs.items()},
        )

    def output(self, target: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        # The graph returns to whoever called the model, so results come back to the host whatever
        # device the last node ran on.
        host = torch.device("cpu")
        return super().output(
            target,
            tuple(_move(value, host) for value in args),
            {name: _move(value, host) for name, value in kwargs.items()},
        )

    def _device_for(self, node: torch.fx.Node | None) -> torch.device:
        if node is not None and node.name in self._clusters:
            return self._device
        return torch.device("cpu")


def execute(
    graph_module: torch.fx.GraphModule,
    clusters: dict[str, int],
    device: torch.device | None = None,
) -> Callable[..., Any]:
    """Build a callable that runs a partitioned graph.

    Args:
        graph_module: The module a ``torch.compile`` backend receives.
        clusters: Map from FX node name to cluster number, as ``tessiq.partition`` returns it.
        device: Where clustered nodes run. Defaults to ``tessiq.cpu``.

    Returns:
        A callable taking the arguments ``graph_module.forward`` takes and returning what it
        returns, on the host. It holds no state between calls, so it can be called repeatedly and
        from more than one thread.
    """

    def run(*args: Any) -> Any:
        return PartitionedExecutor(graph_module, clusters, device).run(*args)

    return run
