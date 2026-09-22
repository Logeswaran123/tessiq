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

import dataclasses
import threading
import weakref
from collections.abc import Callable
from typing import Any

import torch

from tessiq import profiler
from tessiq.recipe import Recipe


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

        collecting = profiler.current()
        if collecting is None:
            return value.to(device) if resident is None else resident.move(value, device)

        with profiler.measure(collecting, f"{value.device} to {device}", "transfer", device) as n:
            moved = value.to(device) if resident is None else resident.move(value, device)
            if moved is not value:
                n.append(value.numel() * value.element_size())
            return moved
    if isinstance(value, (list, tuple)):
        return type(value)(_move(item, device, resident) for item in value)
    return value


class ResidentInputs:
    """Device copies of graph inputs that have not changed since the last call.

    AOTAutograd lifts a module's parameters to graph inputs, so a compiled transformer block takes
    its twelve megabytes of weights as ordinary arguments and the executor moved all of them across
    on every call. They are the same tensors each time.

    Keyed on the host tensor's identity and checked against its version counter, so a parameter an
    optimizer step writes to is copied again rather than read stale. The weak reference is what
    makes the identity safe: a freed tensor's address can be handed to the next allocation, and a
    dead reference misses rather than matching the wrong tensor.

    Bounded by dropping dead entries on insert. The one argument that genuinely changes between
    calls is a new tensor each time, so without that the table would grow by one per call.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._entries: dict[tuple[int, str], _Entry] = {}
        self._hits = 0
        self._misses = 0

    def move(self, value: torch.Tensor, device: torch.device) -> torch.Tensor:
        """The device copy of `value`, reusing the last one when nothing about it has changed."""
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

        moved = value.to(device)
        entry = _Entry(weakref.ref(value), value.data_ptr(), value._version, moved)
        with self._lock:
            self._misses += 1
            self._entries = {k: v for k, v in self._entries.items() if v.source() is not None}
            self._entries[key] = entry
        return moved

    def stats(self) -> tuple[int, int]:
        """Inputs reused and inputs copied, since this callable was built."""
        with self._lock:
            return self._hits, self._misses


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
        # Shared with the other calls of the same compiled graph, which is what makes a weight
        # stay on the device rather than crossing again.
        self._resident = resident if resident is not None else ResidentInputs()
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

        collecting = profiler.current()
        if collecting is None or n.op != "call_function":
            return super().run_node(n)

        # Anything without a kernel runs one operation at a time through PyTorch's dispatcher,
        # wherever the recipe placed it. That is not the same as running on the host: a movement
        # operation placed on the device is interpreted there and costs almost nothing, while an
        # operation the device cannot run reaches the fallback and a round trip. The event carries
        # the device so the two are told apart, and is named after the operation rather than the
        # node, since what a reader wants is which operation is costing them.
        device = self._devices.get(n.name, self._host)
        with profiler.measure(collecting, str(n.target), "interpreted", device):
            return super().run_node(n)

    def _run_kernel(self, kernel: Any, node: torch.fx.Node) -> torch.Tensor:
        """Runs a whole fusion group as one launch."""
        device = self._devices.get(node.name, self._host)
        # Most kernels walk one index across every buffer, which is only the same elements when
        # the layouts agree, so their inputs are packed first. A matmul is the exception: it reads
        # the layout it was given, and packing a transposed operand would move the data and leave
        # the transpose flag describing something that is no longer there.
        moved = [
            _move(self.env[self._nodes[name]], device, self._resident) for name in kernel.inputs
        ]
        inputs = [t.contiguous() for t in moved] if kernel.packs_inputs else moved

        meta = node.meta["val"]
        out = torch.empty(meta.shape, dtype=meta.dtype, device=device)

        # The launch goes to the default stream and the transfers the runtime does go to the
        # device's own, which is a blocking stream. The legacy default stream orders itself
        # against those, so a copy issued after this waits for the kernel without an explicit
        # synchronise here.
        collecting = profiler.current()
        if collecting is None:
            kernel.launch(inputs, out)
            return out

        with profiler.measure(collecting, node.name, "kernel", device):
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
