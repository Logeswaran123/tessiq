# Copyright 2026 The Tessiq Authors.
# SPDX-License-Identifier: Apache-2.0
"""Custom operations defined by an expression: `fast_gelu` and `swiglu` from csrc/custom_ops.

Neither has a kernel file. The compiler writes each one's expression into the loop of whatever it
fuses the operation with, and runs the same code on its own for an eager call. `swiglu` is here for
what `fast_gelu` cannot show: two operands, which broadcast. Whether a call ran on the GPU or fell
back to the reference is read from `tessiq.library.fallbacks`, since both give the same answer.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

import tessiq
from tessiq import _C
from tessiq.recipe import build_recipe

CUSTOM_OPS = Path(__file__).resolve().parents[2] / "csrc" / "custom_ops"

fast_gelu = torch.ops.tessiq.fast_gelu
swiglu = torch.ops.tessiq.swiglu


def expected_gelu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(1.702 * x)


def expected_swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.silu(gate) * up


class Block(torch.nn.Module):
    """The custom operation between two built-in ones on each side, which fusion should swallow."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = torch.relu(fast_gelu(x * 2.0) + 1.0)
        return out


@pytest.fixture
def needs_gpu() -> None:
    if tessiq.gpu_count() == 0:
        pytest.skip("needs a gpu")


@pytest.fixture(autouse=True)
def _fresh() -> None:
    torch._dynamo.reset()


def test_every_directory_in_custom_ops_is_an_operation() -> None:
    directories = {p.name for p in CUSTOM_OPS.iterdir() if p.is_dir()}
    assert {"fast_gelu", "rms_norm", "swiglu"} <= directories
    assert tessiq.library.registered() == sorted(f"tessiq::{name}" for name in directories)


@pytest.mark.cuda
@pytest.mark.usefixtures("needs_gpu")
def test_every_operation_compiles() -> None:
    assert tessiq.library.check() == tessiq.library.registered()


def test_checking_an_unknown_operation_names_the_known_ones() -> None:
    with pytest.raises(KeyError, match="tessiq::fast_gelu"):
        tessiq.library.check("tessiq::missing")


def test_on_the_host_it_is_the_reference() -> None:
    x = torch.randn(100)
    torch.testing.assert_close(fast_gelu(x), expected_gelu(x))


@pytest.mark.cuda
@pytest.mark.usefixtures("needs_gpu")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_on_a_gpu_tensor_it_runs_there(dtype: torch.dtype) -> None:
    x = torch.randn(1000, dtype=dtype)
    before = tessiq.library.fallbacks("tessiq::fast_gelu")
    got = fast_gelu(x.to(tessiq.gpu))
    assert got.device == tessiq.gpu
    assert tessiq.library.fallbacks("tessiq::fast_gelu") == before, "the reference ran instead"
    torch.testing.assert_close(got.cpu(), expected_gelu(x))


@pytest.mark.cuda
@pytest.mark.usefixtures("needs_gpu")
def test_on_a_gpu_tensor_its_operands_broadcast() -> None:
    gate, up = torch.randn(4, 1), torch.randn(1, 5)
    before = tessiq.library.fallbacks("tessiq::swiglu")
    got = swiglu(gate.to(tessiq.gpu), up.to(tessiq.gpu))
    assert tessiq.library.fallbacks("tessiq::swiglu") == before
    assert got.shape == (4, 5)
    torch.testing.assert_close(got.cpu(), expected_swiglu(gate, up))


@pytest.mark.cuda
@pytest.mark.usefixtures("needs_gpu")
def test_on_an_empty_gpu_tensor_nothing_is_launched() -> None:
    got = swiglu(torch.randn(0, 3).to(tessiq.gpu), torch.randn(3).to(tessiq.gpu))
    assert got.shape == (0, 3)


def test_on_the_cpu_reference_device_it_is_the_reference() -> None:
    x = torch.randn(100)
    before = tessiq.library.fallbacks("tessiq::fast_gelu")
    got = fast_gelu(x.to(tessiq.cpu))
    assert got.device == tessiq.cpu
    assert tessiq.library.fallbacks("tessiq::fast_gelu") == before + 1
    torch.testing.assert_close(got.cpu(), expected_gelu(x))


@pytest.mark.cuda
@pytest.mark.usefixtures("needs_gpu")
def test_a_type_it_was_not_written_for_runs_the_reference() -> None:
    x = torch.randn(64, dtype=torch.float16)
    before = tessiq.library.fallbacks("tessiq::fast_gelu")
    got = fast_gelu(x.to(tessiq.gpu))
    assert tessiq.library.fallbacks("tessiq::fast_gelu") == before + 1
    torch.testing.assert_close(got.cpu(), expected_gelu(x))


@pytest.mark.cuda
@pytest.mark.usefixtures("needs_gpu")
def test_operands_of_different_types_run_the_reference() -> None:
    # The expression is compiled for one element type, so there is no kernel for a mix of two.
    gate, up = torch.randn(8), torch.randn(8, dtype=torch.float64)
    before = tessiq.library.fallbacks("tessiq::swiglu")
    got = swiglu(gate.to(tessiq.gpu), up.to(tessiq.gpu))
    assert tessiq.library.fallbacks("tessiq::swiglu") == before + 1
    assert got.dtype == torch.float64
    torch.testing.assert_close(got.cpu(), expected_swiglu(gate, up))


@pytest.mark.cuda
@pytest.mark.usefixtures("needs_gpu")
def test_compiled_it_fuses_with_its_neighbours() -> None:
    x = torch.randn(64, 128)
    graph = tessiq.capture(Block(), x)
    recipe = build_recipe(graph, tessiq.gpu, use_cache=False)

    (kernel,) = recipe.kernels.values()
    assert isinstance(kernel, _C.FusedKernel)
    custom = next(n.name for n in graph.graph.nodes if "fast_gelu" in str(n.target))
    assert custom in kernel.members


@pytest.mark.cuda
@pytest.mark.usefixtures("needs_gpu")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_compiled_it_matches_eager(dtype: torch.dtype) -> None:
    x = torch.randn(64, 128, dtype=dtype)
    compiled = torch.compile(Block(), backend="tessiq", options={"device": str(tessiq.gpu)})
    with torch.no_grad():
        torch.testing.assert_close(compiled(x), Block()(x))


@pytest.mark.cuda
@pytest.mark.usefixtures("needs_gpu")
def test_compiled_on_a_shape_that_changes() -> None:
    compiled = torch.compile(
        Block(), backend="tessiq", options={"device": str(tessiq.gpu)}, dynamic=True
    )
    with torch.no_grad():
        for rows in (8, 24, 40):
            x = torch.randn(rows, 32)
            torch.testing.assert_close(compiled(x), Block()(x))


def test_compiled_for_the_cpu_reference_device() -> None:
    x = torch.randn(16, 16)
    run = tessiq.execute(build_recipe(tessiq.capture(Block(), x), tessiq.cpu, use_cache=False))
    torch.testing.assert_close(run(x)[0], Block()(x))


@pytest.mark.cuda
@pytest.mark.usefixtures("needs_gpu")
def test_compiled_a_type_it_was_not_written_for_stays_on_the_host() -> None:
    x = torch.randn(32, dtype=torch.float16)
    graph = tessiq.capture(Block(), x)
    recipe = build_recipe(graph, tessiq.gpu, use_cache=False)
    custom = next(n.name for n in graph.graph.nodes if "fast_gelu" in str(n.target))
    assert recipe.devices.get(custom, torch.device("cpu")).type == "cpu"
    torch.testing.assert_close(tessiq.execute(recipe)(x)[0], Block()(x))
