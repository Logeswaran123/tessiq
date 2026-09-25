# Copyright 2026 The Tessiq Authors.
# SPDX-License-Identifier: Apache-2.0
"""The operation table and the files generated from it.

The generated files are checked in, so they can be edited by hand. These make that fail rather
than drift.
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
    # one does not error; it stops a chain fusing.
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
    # An expression computes one element from the inputs at the same position, so only an
    # elementwise operation can have one.
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
        if not name.startswith("aten.") or name == "aten.convolution.default":
            # Convolution support also depends on the shape: only 2-D convolutions get a cuDNN
            # plan. It has its own test below.
            continue
        graph = f"graph @g(%a: f32[4], %b: f32[4]) -> f32[4] {{\n  %x = {name}(%a, %b) : f32[4]\n  return %x\n}}\n"
        marked = tessiq.run_passes(graph, "mark-supported")
        assert "tessiq.supported = true" in marked, f"{name} is in the table and not claimed"


def test_an_operation_the_table_has_never_heard_of_is_not_claimed() -> None:
    # Every row in the table is claimed, so this checks the other direction: an operation with no
    # row is declined.
    for name in (
        "aten.hypot.default",
        "aten.native_layer_norm.default",
        "aten.erfinv.default",
    ):
        graph = f"graph @g(%a: f32[4], %b: f32[4]) -> f32[4] {{\n  %x = {name}(%a, %b) : f32[4]\n  return %x\n}}\n"
        marked = tessiq.run_passes(graph, "mark-supported")
        assert "tessiq.supported = false" in marked, f"{name} is not in the table and was claimed"


def test_a_convolution_is_claimed_only_when_cudnn_would_have_a_plan() -> None:
    # A 2-D convolution has 4-D tensors and gets a cuDNN plan. A 1-D one has the same name and no
    # plan, so it is not claimed.
    two_d = (
        "graph @g(%a: f32[1,3,8,8], %w: f32[4,3,3,3]) -> f32[1,4,6,6] {\n"
        "  %c = aten.convolution.default(%a, %w) : f32[1,4,6,6]\n  return %c\n}\n"
    )
    one_d = (
        "graph @g(%a: f32[1,3,8], %w: f32[4,3,3]) -> f32[1,4,6] {\n"
        "  %c = aten.convolution.default(%a, %w) : f32[1,4,6]\n  return %c\n}\n"
    )
    assert "tessiq.supported = true" in tessiq.run_passes(two_d, "mark-supported")
    assert "tessiq.supported = false" in tessiq.run_passes(one_d, "mark-supported")


def test_the_coverage_document_matches_the_source() -> None:
    document = (ROOT / "docs" / "op_coverage.md").read_text()
    with_kernel = sum(1 for op in operations() if op["kernel"])
    assert f"{with_kernel} of {len(operations())} operations can run on the device" in document
    for op in operations():
        assert f"`{op['name']}`" in document, f"{op['name']} is missing from the document"


@pytest.mark.parametrize(
    "category", ["elementwise", "reduction", "movement", "matmul", "copy", "convolution"]
)
def test_every_category_is_one_the_generator_knows(category: str) -> None:
    assert any(op["category"] == category for op in operations()), (
        f"no operation is {category}; the generator would still accept it, "
        "but nothing exercises that path"
    )


def test_reductions_have_kernels_but_no_expression() -> None:
    # A reduction has a kernel but no expression: it is a whole kernel of its own, which the
    # emitter writes.
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
