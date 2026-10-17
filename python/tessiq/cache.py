# Copyright 2026 The Tessiq Authors.
# SPDX-License-Identifier: Apache-2.0
"""Reusing a recipe the compiler has already produced.

A graph compiled once has the same recipe every time it comes back. The cache is keyed on what
the recipe depends on: the structure of the graph, the type of every value in it, the device it
was lowered for and the rewrites asked for.
"""

from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from tessiq.recipe import Recipe

#: Recipes kept before the least recently used one is dropped.
#:
#: A recipe holds the graph it was built from, so the cache is bounded. Sixty-four holds a model's
#: forward and backward graphs at a handful of shapes.
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


def signature(
    graph_module: torch.fx.GraphModule, device: torch.device, rewrites: Sequence[str] = ()
) -> str:
    """A key identifying every graph that would compile to the same recipe.

    Walks FX rather than importing to IR, since import is most of the cost being avoided. Dtypes,
    shapes and strides are in the key because the memory plan and the kernels are built from them,
    the device because it decides where clusters run, and the rewrites because they change which
    operations are compiled.

    Returns:
        A hex digest of all of the above.
    """
    parts = [f"device={device}", f"rewrites={','.join(rewrites)}"]
    for node in graph_module.graph.nodes:
        value = node.meta.get("val")
        typing = (
            f"{value.dtype}{tuple(value.shape)}{value.stride()}"
            if isinstance(value, torch.Tensor)
            else "untyped"
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

    Mostly for tests. A recipe something else already holds stays valid.
    """
    _CACHE.clear()
