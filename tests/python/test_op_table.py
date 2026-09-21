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
import tomllib

import tessiq

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "ops.toml"
GENERATOR = ROOT / "tools" / "gen_ops.py"


def operations() -> list[dict[str, Any]]:
    with SOURCE.open("rb") as handle:
        return list(tomllib.load(handle)["op"])


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
    graph = (
        "graph @g(%a: f32[4]) -> f32[4] {\n  %x = aten.sqrt.default(%a) : f32[4]\n  return %x\n}\n"
    )
    assert "tessiq.supported = false" in tessiq.run_passes(graph, "mark-supported")


def test_the_coverage_document_matches_the_source() -> None:
    document = (ROOT / "docs" / "op_coverage.md").read_text()
    with_kernel = sum(1 for op in operations() if op["kernel"])
    assert f"{with_kernel} of {len(operations())} operations have a kernel" in document
    for op in operations():
        assert f"`{op['name']}`" in document, f"{op['name']} is missing from the document"


@pytest.mark.parametrize("category", ["elementwise", "matmul"])
def test_every_category_is_one_the_generator_knows(category: str) -> None:
    assert any(op["category"] == category for op in operations()), (
        f"no operation is {category}; the generator would still accept it, "
        "but nothing exercises that path"
    )
