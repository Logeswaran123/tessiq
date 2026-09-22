# Copyright 2026 The Tessiq Authors.
# SPDX-License-Identifier: Apache-2.0
"""The operation table and the files generated from it.

The generated files are checked in, so they can be edited by hand. These make that fail rather
than drift, which is the whole reason for generating them: an operation defined in one place
cannot disagree with itself.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

import tessiq
from conftest import load_operations

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "ops.toml"
GENERATOR = ROOT / "tools" / "gen_ops.py"


def operations() -> list[dict[str, Any]]:
    return load_operations()


def test_the_generated_files_are_up_to_date() -> None:
    result = subprocess.run(
        [sys.executable, str(GENERATOR), "--check"],
        capture_output=True,
        text=True,
        check=False,
        cwd=ROOT,
    )
    assert result.returncode == 0, (
        f"{result.stderr}\nops.toml changed without regenerating. Run: python tools/gen_ops.py"
    )


def test_every_operation_is_named_the_same_way_twice() -> None:
    # The short name is for hand-written IR and the ATen names are what torch emits. A row missing
    # one of them does not error, it quietly stops a chain fusing, which is how aten.tanh.default
    # went missing once.
    for op in operations():
        assert op["aten"], f"{op['name']} has no ATen spelling"
        for aten in op["aten"]:
            assert str(aten).startswith("aten."), f"{op['name']}: {aten} is not an ATen name"


def test_elementwise_operations_say_what_they_compute() -> None:
    for op in operations():
        if op["category"] == "elementwise":
            assert op.get("expression"), f"{op['name']} is elementwise with no expression"
            arity = int(str(op["arity"]))
            for index in range(arity):
                assert f"${index}" in str(op["expression"]), (
                    f"{op['name']}: expression does not use ${index}"
                )


def test_only_elementwise_operations_have_an_expression() -> None:
    # An expression describes one element computed from the inputs at that position. An operation
    # that reads elsewhere has no such thing, and carrying one would invite a pass to use it.
    for op in operations():
        if op["category"] != "elementwise":
            assert not op.get("expression"), f"{op['name']} is not elementwise but has one"


def test_the_backend_claims_what_the_table_says() -> None:
    claimed = {
        name
        for op in operations()
        if op["kernel"]
        for name in (str(op["name"]), *(str(a) for a in op["aten"]))
    }
    # partition marks what the backend can run, so this is the table reaching a real graph.
    for name in sorted(claimed):
        if not name.startswith("aten."):
            continue
        graph = f"graph @g(%a: f32[4], %b: f32[4]) -> f32[4] {{\n  %x = {name}(%a, %b) : f32[4]\n  return %x\n}}\n"
        marked = tessiq.run_passes(graph, "mark-supported")
        assert "tessiq.supported = true" in marked, f"{name} is in the table and not claimed"


def test_an_operation_without_a_kernel_is_not_claimed() -> None:
    # The other direction of the test above. Named from the table rather than hardcoded, so that
    # giving one of these a kernel moves it to the claimed side instead of failing here.
    unclaimed = [op for op in operations() if not op["kernel"]]
    assert unclaimed, "every operation has a kernel, so this test no longer checks anything"
    for op in unclaimed:
        for name in (str(a) for a in op["aten"]):
            graph = f"graph @g(%a: f32[4], %b: f32[4]) -> f32[4] {{\n  %x = {name}(%a, %b) : f32[4]\n  return %x\n}}\n"
            marked = tessiq.run_passes(graph, "mark-supported")
            assert "tessiq.supported = false" in marked, f"{name} has no kernel and was claimed"


def test_the_coverage_document_matches_the_source() -> None:
    document = (ROOT / "docs" / "op_coverage.md").read_text()
    with_kernel = sum(1 for op in operations() if op["kernel"])
    assert f"{with_kernel} of {len(operations())} operations can run on the device" in document
    for op in operations():
        assert f"`{op['name']}`" in document, f"{op['name']} is missing from the document"


@pytest.mark.parametrize("category", ["elementwise", "matmul"])
def test_every_category_is_one_the_generator_knows(category: str) -> None:
    assert any(op["category"] == category for op in operations()), (
        f"no operation is {category}; the generator would still accept it, "
        "but nothing exercises that path"
    )


def test_reductions_have_kernels_but_no_expression() -> None:
    # A reduction has a kernel, and it is not an expression. An elementwise row says what one
    # element computes and the emitter drops it into a shared loop; a reduction is a whole kernel
    # of its own, so the table says which operation it is and the emitter writes the rest.
    reductions = [op for op in operations() if op["category"] == "reduction"]
    assert reductions, "the reduction category is empty"
    for op in reductions:
        assert op["kernel"], f"{op['name']} has no kernel"
        assert not op.get("expression"), f"{op['name']} has an expression"


def test_shape_inference_reads_the_dimensions_off_the_node() -> None:
    graph = (
        "graph @g(%a: f32[4,8,3]) -> f32[?,?] {\n"
        "  %s = aten.sum.dim_IntList(%a) {arg1 = [1]} : f32[?,?]\n"
        "  return %s\n"
        "}\n"
    )
    assert "f32[4,3]" in tessiq.run_passes(graph, "shape-infer")
