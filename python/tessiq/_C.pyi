# Copyright 2026 The Tessiq Authors.
# SPDX-License-Identifier: Apache-2.0
"""Type stubs for the native extension module.

Hand-written, because pybind11 does not emit stubs. Keep in step with csrc/bindings/module.cpp:
mypy checks the Python package against this file, so drift here is a silent loss of type coverage
rather than an error.
"""

from typing import Any

__version__: str

def version() -> str: ...
def set_log_level(level: str) -> None: ...
def get_log_level() -> str: ...
def _raise_test_error(code: str) -> None: ...

# --- backend ---------------------------------------------------------------------------------

def _init_backend() -> None: ...
def device_count() -> int: ...
def backend_name(index: int) -> str: ...
def empty_cache(index: int) -> int: ...

class AllocatorStats:
    allocated_bytes: int
    reserved_bytes: int
    peak_allocated_bytes: int
    peak_reserved_bytes: int
    allocation_count: int
    deallocation_count: int
    cache_hits: int
    cache_misses: int

def memory_stats(index: int) -> AllocatorStats: ...

# --- compiler --------------------------------------------------------------------------------

def import_fx(graph_module: object, name: str = ...) -> str: ...
def partition_fx(graph_module: object, name: str = ...) -> tuple[str, dict[str, int]]: ...
def run_passes(ir: str, pipeline: str) -> str: ...
def default_pipeline() -> str: ...
def pass_names() -> list[str]: ...
def emit_cuda(ir: str) -> str: ...

class Step:
    op: str
    results: list[str]
    cluster: int

class Buffer:
    name: str
    offset: int
    size: int

class Recipe:
    steps: list[Step]
    buffers: list[Buffer]
    arena_bytes: int
    naive_bytes: int
    def __str__(self) -> str: ...

class FusedKernel:
    group: int
    inputs: list[str]
    output: str
    members: list[str]
    elements: int
    has_dynamic_extent: bool
    packs_inputs: bool
    def output_shape(self, inputs: list[Any]) -> list[int]: ...
    def launch(self, inputs: list[Any], out: Any) -> None: ...

class ReductionKernel:
    output: str
    inputs: list[str]
    outer: int
    reduce: int
    inner: int
    blocks: int
    elements: int
    has_dynamic_extent: bool
    packs_inputs: bool
    def output_shape(self, inputs: list[Any]) -> list[int]: ...
    def launch(self, inputs: list[Any], out: Any) -> None: ...

class GemmKernel:
    output: str
    inputs: list[str]
    packs_inputs: bool
    transposed: tuple[bool, bool]
    extents: tuple[int, int, int, int]
    has_dynamic_extent: bool
    def output_shape(self, inputs: list[Any]) -> list[int]: ...
    def launch(self, inputs: list[Any], out: Any) -> None: ...

class CompiledGraph:
    recipe: Recipe
    kernels: list[FusedKernel]
    reductions: list[ReductionKernel]
    gemms: list[GemmKernel]

def build_recipe_fx(
    graph_module: object, name: str = ..., alignment: int = ..., ordinal: int = ...
) -> CompiledGraph: ...
