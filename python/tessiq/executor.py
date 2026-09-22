# Copyright 2026 The Tessiq Authors.
# SPDX-License-Identifier: Apache-2.0
"""Running a recipe.

Everything that can be decided without seeing the tensors was decided when the recipe was built,
so a call is a walk over the graph doing only what the recipe says. Until kernels exist, a tessiq
operation reaches PyTorch's CPU kernel through the fallback, so this is slower than not compiling
at all. What it buys is that the whole path is exercised and checked against eager, which is the
agreement a lowered recipe will have to keep.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

from tessiq.recipe import Recipe


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


class RecipeExecutor(torch.fx.Interpreter):
    """Executes the graph a recipe was built from, following what the recipe decided.

    Not thread-safe, and not reusable. One instance per call, which is what `execute` does.

    Args:
        recipe: The compiled graph to run.
    """

    def __init__(self, recipe: Recipe) -> None:
        # Values are kept for the whole call rather than dropped once FX thinks they are dead. A
        # fused group reads its inputs at the node that writes the group's output, which is later
        # than the last use FX can see, since the uses in between belong to nodes the kernel
        # subsumes. With collection on, an input is freed before the kernel asks for it.
        super().__init__(recipe.graph_module, garbage_collect_values=not recipe.kernels)
        self._devices = recipe.devices
        self._needs_move = recipe.needs_move
        self._output_spec = recipe.output_spec
        self._kernels = recipe.kernels
        self._fused_away = recipe.fused_away
        self._nodes = {node.name: node for node in recipe.graph_module.graph.nodes}
        self._host = torch.device("cpu")
        self._node: torch.fx.Node | None = None

    def run_node(self, n: torch.fx.Node) -> Any:
        if n.name in self._fused_away:
            # A kernel computes this on its way to the group's output, so running it here would
            # compute the same value twice. Nothing reads what this returns: every consumer is
            # inside the same group, which is the condition that let it fuse.
            return None

        kernel = self._kernels.get(n.name)
        if kernel is not None:
            return self._run_kernel(kernel, n)

        # Recorded because call_function is handed the target and the arguments, not the node, and
        # where a step runs is a property of the node.
        self._node = n
        return super().run_node(n)

    def _run_kernel(self, kernel: Any, node: torch.fx.Node) -> torch.Tensor:
        """Runs a whole fusion group as one launch."""
        device = self._devices.get(node.name, self._host)
        # Most kernels walk one index across every buffer, which is only the same elements when
        # the layouts agree, so their inputs are packed first. A matmul is the exception: it reads
        # the layout it was given, and packing a transposed operand would move the data and leave
        # the transpose flag describing something that is no longer there.
        moved = [_move(self.env[self._nodes[name]], device) for name in kernel.inputs]
        inputs = [t.contiguous() for t in moved] if kernel.packs_inputs else moved

        meta = node.meta["val"]
        out = torch.empty(meta.shape, dtype=meta.dtype, device=device)

        # The launch goes to the default stream and the transfers the runtime does go to the
        # device's own, which is a blocking stream. The legacy default stream orders itself
        # against those, so a copy issued after this waits for the kernel without an explicit
        # synchronise here.
        kernel.launch(inputs, out)
        return out

    def call_function(self, target: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        name = self._node.name if self._node is not None else ""
        if name not in self._needs_move:
            # Everything this reads is already where it has to be, which is the common case in the
            # middle of a cluster. Moving again would be two transfers that cancel out.
            return super().call_function(target, args, kwargs)

        device = self._devices.get(name, self._host)
        return super().call_function(
            target,
            tuple(_move(value, device) for value in args),
            {key: _move(value, device) for key, value in kwargs.items()},
        )

    def output(self, target: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        produced = args[0]
        if not isinstance(produced, (tuple, list)):
            produced = (produced,)

        # Rebuilt from the recipe's spec rather than passed through, so every call checks that the
        # spec still describes the graph. strict=True is the check: a spec of the wrong length
        # raises here instead of handing the autograd engine a tuple whose positions have shifted.
        rebuilt = tuple(
            None if name is None else _move(value, self._host)
            for name, value in zip(self._output_spec, produced, strict=True)
        )
        return super().output(target, (rebuilt,), kwargs)


def execute(recipe: Recipe) -> Callable[..., Any]:
    """Build a callable that runs `recipe`.

    Args:
        recipe: A compiled graph, from `tessiq.build_recipe`.

    Returns:
        A callable taking the arguments the graph takes and returning what it returns, on the
        host. It holds no state between calls, so it can be called repeatedly and from more than
        one thread.
    """

    def run(*args: Any) -> Any:
        return RecipeExecutor(recipe).run(*args)

    return run
