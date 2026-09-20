# Copyright 2026 The Tessiq Authors.
# SPDX-License-Identifier: Apache-2.0
"""Reusing a recipe the compiler has already produced.

Compiling is not free, and a graph that has been compiled once has the same recipe every time it
comes back. The cache is keyed on what the recipe actually depends on, which is the structure of
the graph, the type of every value in it, and the device it was lowered for.
"""

from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from tessiq.recipe import Recipe

#: Recipes kept before the least recently used one is dropped.
#:
#: A recipe holds the graph it was built from, so an unbounded cache keeps every graph a process
#: has ever compiled. Sixty-four is enough that a model's forward and backward graphs at a handful
#: of shapes all stay resident, and small enough that a long-running server does not grow without
#: limit.
MAX_ENTRIES = 64


@dataclass(frozen=True)
class CacheStats:
    """Counters for the recipe cache.

    Attributes:
        hits: Compilations answered from the cache.
        misses: Compilations that had to build a recipe.
        entries: Recipes currently held.
        evictions: Recipes dropped to stay within `MAX_ENTRIES`.
    """

    hits: int = 0
    misses: int = 0
    entries: int = 0
    evictions: int = 0


class _Cache:
    """Bounded, least-recently-used, and safe to share between threads."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._entries: OrderedDict[str, Recipe] = OrderedDict()
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def get(self, key: str) -> Recipe | None:
        with self._lock:
            recipe = self._entries.get(key)
            if recipe is None:
                self._misses += 1
                return None
            self._entries.move_to_end(key)
            self._hits += 1
            return recipe

    def put(self, key: str, recipe: Recipe) -> None:
        with self._lock:
            self._entries[key] = recipe
            self._entries.move_to_end(key)
            while len(self._entries) > MAX_ENTRIES:
                self._entries.popitem(last=False)
                self._evictions += 1

    def stats(self) -> CacheStats:
        with self._lock:
            return CacheStats(
                hits=self._hits,
                misses=self._misses,
                entries=len(self._entries),
                evictions=self._evictions,
            )

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._hits = 0
            self._misses = 0
            self._evictions = 0


_CACHE = _Cache()


def _describe(value: object) -> str:
    """One node argument, in a form that distinguishes anything the recipe would differ on."""
    if isinstance(value, torch.fx.Node):
        return f"%{value.name}"
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_describe(item) for item in value) + "]"
    return repr(value)


def signature(graph_module: torch.fx.GraphModule, device: torch.device) -> str:
    """A key identifying every graph that would compile to the same recipe.

    Walks FX rather than the IR, which is the point: computing it has to be cheaper than the
    compilation it is trying to avoid, and importing the graph is most of that cost.

    Shapes and dtypes are in the key because the memory plan is built from them, so the same
    structure at a different shape is a different recipe. The device is in the key because it
    decides where the clustered steps run.

    Returns:
        A hex digest. Content-addressed, so two graphs share a key exactly when they would compile
        to the same recipe.
    """
    parts = [f"device={device}"]
    for node in graph_module.graph.nodes:
        value = node.meta.get("val")
        typing = (
            f"{value.dtype}{tuple(value.shape)}" if isinstance(value, torch.Tensor) else "untyped"
        )
        arguments = ",".join(_describe(argument) for argument in node.args)
        keywords = ",".join(f"{k}={_describe(v)}" for k, v in sorted(node.kwargs.items()))
        parts.append(f"{node.op}|{node.target}|{arguments}|{keywords}|{typing}")

    joined = "\n".join(parts)
    return hashlib.blake2b(joined.encode(), digest_size=16).hexdigest()


def lookup(key: str) -> Recipe | None:
    """The recipe stored under `key`, or None. Counts a hit or a miss either way."""
    return _CACHE.get(key)


def store(key: str, recipe: Recipe) -> None:
    """Keep `recipe` under `key`, evicting the least recently used if the cache is full."""
    _CACHE.put(key, recipe)


def cache_stats() -> CacheStats:
    """Hits, misses, entries and evictions since the process started or the cache was cleared."""
    return _CACHE.stats()


def clear_cache() -> None:
    """Drop every cached recipe and reset the counters.

    Mostly for tests, which need a known starting point. Clearing does not invalidate a recipe
    something else is already holding.
    """
    _CACHE.clear()
