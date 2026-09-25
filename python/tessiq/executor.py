# Copyright 2026 The Tessiq Authors.
# SPDX-License-Identifier: Apache-2.0
"""Running a recipe.

Everything that can be decided without seeing the tensors was decided when the recipe was built.
A call walks the graph, launching a kernel for each compiled group and running every other node
through PyTorch's dispatcher on the device the recipe chose.
"""

from __future__ import annotations

import dataclasses
import threading
import weakref
from collections.abc import Callable
from typing import Any

import torch

from tessiq import profiler
from tessiq.recipe import Recipe

#: The stream every launch goes to: the legacy default stream, whose handle is zero. It
#: synchronises with the blocking stream the runtime copies on, so kernels and transfers are
#: ordered without explicit synchronisation.
LEGACY_DEFAULT_STREAM = 0


def _move(value: Any, device: torch.device, resident: ResidentInputs | None = None) -> Any:
    """Move a tensor to ``device``, leaving anything else alone.

    Arguments are not all tensors. Scalars, shapes, dtypes and ``None`` all arrive here, and
    ``.to()`` on them would fail. Nested lists and tuples turn up in ops like ``cat`` and ``stack``.

    With a `resident` table, a tensor that has already been moved and has not changed since is
    reused rather than copied again.
    """
    if isinstance(value, torch.Tensor):
        if value.device == device:
            return value

        # Checked before anything is measured, so a reused copy is not recorded as a transfer.
        if resident is not None:
            kept = resident.lookup(value, device)
            if kept is not None:
                return kept

        collecting = profiler.current()
        if collecting is None:
            return value.to(device) if resident is None else resident.copy(value, device)

        with profiler.measure(collecting, f"{value.device} to {device}", "transfer", device) as n:
            moved = value.to(device) if resident is None else resident.copy(value, device)
            n.append(value.numel() * value.element_size())
            return moved
    if isinstance(value, (list, tuple)):
        return type(value)(_move(item, device, resident) for item in value)
    return value


class ResidentInputs:
    """Device copies of graph inputs that have not changed since the last call.

    AOTAutograd lifts a module's parameters to graph inputs, so without this every call would copy
    the weights to the device again.

    Keyed on the host tensor's identity and checked against its version counter, so a parameter an
    optimizer step writes to is copied again. The weak reference makes the identity safe: a freed
    tensor's id can be reused, and a dead reference misses rather than matching the wrong tensor.

    Dead entries are dropped on insert, so inputs that are new tensors each call do not accumulate.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._entries: dict[tuple[int, str], _Entry] = {}
        self._hits = 0
        self._misses = 0

    def move(self, value: torch.Tensor, device: torch.device) -> torch.Tensor:
        """The device copy of `value`, reusing the last one when nothing about it has changed."""
        kept = self.lookup(value, device)
        return kept if kept is not None else self.copy(value, device)

    def lookup(self, value: torch.Tensor, device: torch.device) -> torch.Tensor | None:
        """The device copy kept from an earlier call, or None when there is none still valid.

        Separate from `move` so the profiler can tell a reuse, which moves nothing, from a copy.
        """
        key = (id(value), str(device))
        with self._lock:
            found = self._entries.get(key)
            if (
                found is not None
                and found.source() is value
                and found.data_ptr == value.data_ptr()
                and found.version == value._version
            ):
                self._hits += 1
                return found.moved
        return None

    def copy(self, value: torch.Tensor, device: torch.device) -> torch.Tensor:
        """Copies `value` to `device` and keeps the copy for the next call."""
        moved = value.to(device)
        entry = _Entry(weakref.ref(value), value.data_ptr(), value._version, moved)
        key = (id(value), str(device))
        with self._lock:
            self._misses += 1
            self._entries = {k: v for k, v in self._entries.items() if v.source() is not None}
            self._entries[key] = entry
        return moved

    def stats(self) -> tuple[int, int]:
        """Inputs reused and inputs copied, since this callable was built."""
        with self._lock:
            return self._hits, self._misses


def _output_for(
    kernel: Any,
    node: torch.fx.Node,
    inputs: list[torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    """Allocates the tensor a kernel writes.

    The shape comes from the fake tensor AOTAutograd left on the node. In a graph traced for more
    than one shape that fake tensor has symbolic sizes, so the kernel works the shape out from its
    inputs instead.
    """
    if getattr(kernel, "has_dynamic_extent", False):
        return torch.empty(kernel.output_shape(inputs), dtype=inputs[0].dtype, device=device)
    meta = node.meta["val"]
    return torch.empty(meta.shape, dtype=meta.dtype, device=device)


@dataclasses.dataclass(frozen=True)
class _Entry:
    source: weakref.ref[torch.Tensor]
    data_ptr: int
    version: int
    moved: torch.Tensor


class RecipeExecutor(torch.fx.Interpreter):
    """Executes the graph a recipe was built from, following what the recipe decided.

    Not thread-safe, and not reusable. One instance per call, which is what `execute` does.

    Args:
        recipe: The compiled graph to run.
    """

    def __init__(self, recipe: Recipe, resident: ResidentInputs | None = None) -> None:
        # Values are kept for the whole call. A fused group reads its inputs at the node that
        # writes its output, later than the last use FX can see, so FX would free them too early.
        super().__init__(recipe.graph_module, garbage_collect_values=not recipe.kernels)
        self._devices = recipe.devices
        self._needs_move = recipe.needs_move
        self._output_spec = recipe.output_spec
        self._kernels = recipe.kernels
        self._fused_away = recipe.fused_away
        self._nodes = {node.name: node for node in recipe.graph_module.graph.nodes}
        self._host = torch.device("cpu")
        # Shared across calls of the same compiled graph, so weights stay on the device.
        self._resident = resident if resident is not None else ResidentInputs()
        self._node: torch.fx.Node | None = None

    def run_node(self, n: torch.fx.Node) -> Any:
        if n.name in self._fused_away:
            # The group's kernel computes this. Nothing reads the result, since every consumer is
            # inside the group.
            return None

        kernel = self._kernels.get(n.name)
        if kernel is not None:
            return self._run_kernel(kernel, n)

        # call_function gets the target and arguments but not the node, which decides placement.
        self._node = n

        collecting = profiler.current()
        if collecting is None or n.op != "call_function":
            return super().run_node(n)

        # Anything without a kernel runs through PyTorch's dispatcher on the device the recipe
        # chose. On a tessiq device that is a view for a movement operation and a host round trip
        # for anything else. The event records the device and is named after the operation.
        device = self._devices.get(n.name, self._host)
        with profiler.measure(collecting, str(n.target), "interpreted", device):
            return super().run_node(n)

    def _run_kernel(self, kernel: Any, node: torch.fx.Node) -> torch.Tensor:
        """Runs a whole fusion group as one launch."""
        device = self._devices.get(node.name, self._host)
        # Packed first unless the kernel reads layouts itself, as a matmul does.
        moved = [
            _move(self.env[self._nodes[name]], device, self._resident) for name in kernel.inputs
        ]
        inputs = [t.contiguous() for t in moved] if kernel.packs_inputs else moved

        out = _output_for(kernel, node, inputs, device)

        collecting = profiler.current()
        if collecting is None:
            kernel.launch(inputs, out, stream=LEGACY_DEFAULT_STREAM)
            return out

        with profiler.measure(collecting, node.name, "kernel", device):
            kernel.launch(inputs, out, stream=LEGACY_DEFAULT_STREAM)
        return out

    def call_function(self, target: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        name = self._node.name if self._node is not None else ""
        if name not in self._needs_move:
            # Everything is already on the right device, as in the middle of a cluster.
            return super().call_function(target, args, kwargs)

        device = self._devices.get(name, self._host)
        return super().call_function(
            target,
            tuple(_move(value, device, self._resident) for value in args),
            {key: _move(value, device, self._resident) for key, value in kwargs.items()},
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
        host.

        It keeps one thing between calls: the device copies of inputs that have not changed, which
        is how a module's weights stay on the device rather than crossing on every call. That table
        is locked, so the callable is still safe to use from more than one thread.
    """
    resident = ResidentInputs()

    def run(*args: Any) -> Any:
        return RecipeExecutor(recipe, resident).run(*args)

    run.resident = resident  # type: ignore[attr-defined]
    return run
