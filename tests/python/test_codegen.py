# Copyright 2026 The Tessiq Authors.
# SPDX-License-Identifier: Apache-2.0
"""Generated CUDA source, compiled.

The C++ tests check what the emitter writes. These check that what it writes is accepted by the
CUDA compiler, which is a different question and the one that matters: source that reads correctly
and does not compile is worth nothing.

Skipped when no CUDA toolkit is present, which is most machines and the CI runner.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

import tessiq

nvcc = shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"
needs_nvcc = pytest.mark.skipif(not Path(nvcc).exists(), reason="no CUDA toolkit")

PIPELINE = "mark-supported,cluster,fuse-elementwise"

CHAIN = """graph @g(%a: {t}[1024], %b: {t}[1024], %c: {t}[1024]) -> {t}[1024] {{
  %m = aten.mul.Tensor(%a, %b) : {t}[1024]
  %s = aten.add.Tensor(%m, %c) : {t}[1024]
  %r = aten.relu.default(%s) : {t}[1024]
  %g = aten.sigmoid.default(%r) : {t}[1024]
  return %g
}}
"""


def compile_cuda(source: str, tmp_path: Path, extra: list[str] | None = None) -> str:
    """Compile with nvcc and return the output, raising with the compiler's message on failure."""
    cu = tmp_path / "kernel.cu"
    cu.write_text(source)
    out = tmp_path / "kernel.out"
    result = subprocess.run(
        [nvcc, "-arch=sm_80", *(extra or ["-cubin"]), "-o", str(out), str(cu)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"{result.stderr}\n--- source ---\n{source}"
    return out.read_text(errors="replace") if out.suffix == ".ptx" else ""


@needs_nvcc
@pytest.mark.parametrize("dtype", ["f32", "f64"])
def test_the_generated_kernel_compiles(dtype: str, tmp_path: Path) -> None:
    source = tessiq.emit_cuda(tessiq.run_passes(CHAIN.format(t=dtype), PIPELINE))
    compile_cuda(source, tmp_path)


@needs_nvcc
def test_a_float_kernel_uses_no_double_arithmetic(tmp_path: Path) -> None:
    # The maths functions are spelled without a float suffix so one expression serves every
    # element type. That only works if overload resolution picks the float form: if it promoted to
    # double, a consumer GPU would run this at a fraction of the speed and nothing would say so.
    cu = tmp_path / "kernel.cu"
    cu.write_text(tessiq.emit_cuda(tessiq.run_passes(CHAIN.format(t="f32"), PIPELINE)))
    ptx = tmp_path / "kernel.ptx"
    subprocess.run(
        [nvcc, "-arch=sm_80", "-ptx", "-o", str(ptx), str(cu)], capture_output=True, check=True
    )

    text = ptx.read_text()
    assert ".f64" not in text, "the float kernel promoted to double"
    assert re.search(r"ex2\.approx[a-z.]*\.f32", text), "sigmoid did not become a float exponent"


@needs_nvcc
def test_every_supported_operation_compiles(tmp_path: Path) -> None:
    # One group touching each operation the emitter has an expression for. A malformed expression
    # is a compile error rather than a wrong number, so this is the cheapest way to check them.
    ops = "\n".join(
        [
            "  %o1 = aten.add.Tensor(%a, %b) : f32[64]",
            "  %o2 = aten.sub.Tensor(%o1, %b) : f32[64]",
            "  %o3 = aten.mul.Tensor(%o2, %b) : f32[64]",
            "  %o4 = aten.div.Tensor(%o3, %b) : f32[64]",
            "  %o5 = aten.abs.default(%o4) : f32[64]",
            "  %o6 = aten.sqrt.default(%o5) : f32[64]",
            "  %o7 = aten.neg.default(%o6) : f32[64]",
            "  %o8 = aten.relu.default(%o7) : f32[64]",
            "  %o9 = aten.sigmoid.default(%o8) : f32[64]",
            "  %o10 = aten.tanh.default(%o9) : f32[64]",
        ]
    )
    graph = f"graph @g(%a: f32[64], %b: f32[64]) -> f32[64] {{\n{ops}\n  return %o10\n}}\n"

    # sqrt has no kernel, so the chain splits. Both halves still have to compile.
    source = tessiq.emit_cuda(tessiq.run_passes(graph, PIPELINE))
    assert source.count("__global__") == 2, source
    compile_cuda(source, tmp_path)


def test_a_graph_with_no_groups_says_so() -> None:
    graph = "graph @g(%a: f32[8], %b: f32[8]) -> f32[8] {\n  %s = aten.add.Tensor(%a, %b) : f32[8]\n  return %s\n}\n"
    assert "no fusion groups" in tessiq.emit_cuda(tessiq.run_passes(graph, PIPELINE))
