# Copyright 2026 The Tessiq Authors.
# SPDX-License-Identifier: Apache-2.0
"""Operations registered from Python.

The two operations below are defined in this file and nowhere else: no row in ops.toml, no C++, no
rebuild. That is the claim these check. Each reference counts its calls, which is how a test tells
the kernel running from the reference running, since both give the same answer.
"""

from __future__ import annotations

import re

import pytest
import torch

import tessiq
from tessiq import _C
from tessiq.recipe import build_recipe

calls = {"fast_gelu": 0, "fma": 0}


@tessiq.library.elementwise(
    "tessiq_test::fast_gelu",
    "$0 * (scalar_t(1) / (scalar_t(1) + exp(scalar_t(-1.702) * $0)))",
    dtypes=(torch.float32, torch.float64),
)
def fast_gelu(x: torch.Tensor) -> torch.Tensor:
    calls["fast_gelu"] += 1
    return x * torch.sigmoid(1.702 * x)


@tessiq.library.elementwise("tessiq_test::fma", "($0 * $1 + $2)")
def fma(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    calls["fma"] += 1
    return a * b + c


def expected_gelu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(1.702 * x)


class Block(torch.nn.Module):
    """The custom operation between two built-in ones on each side, which fusion should swallow."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(fast_gelu(x * 2.0) + 1.0)


@pytest.fixture
def needs_gpu() -> None:
    if tessiq.gpu_count() == 0:
        pytest.skip("needs a gpu")


@pytest.fixture(autouse=True)
def _fresh() -> None:
    torch._dynamo.reset()


def test_both_are_registered_with_the_compiler() -> None:
    assert {"tessiq_test.fast_gelu.default", "tessiq_test.fma.default"} <= set(
        tessiq.library.registered()
    )


def test_on_the_host_it_is_the_reference() -> None:
    x = torch.randn(100)
    before = calls["fast_gelu"]
    torch.testing.assert_close(fast_gelu(x), expected_gelu(x))
    assert calls["fast_gelu"] == before + 1


@pytest.mark.cuda
@pytest.mark.usefixtures("needs_gpu")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_on_a_gpu_tensor_it_runs_the_kernel(dtype: torch.dtype) -> None:
    x = torch.randn(1000, dtype=dtype)
    before = calls["fast_gelu"]
    got = fast_gelu(x.to(tessiq.gpu))
    assert got.device == tessiq.gpu
    assert calls["fast_gelu"] == before, "the reference ran instead of the kernel"
    torch.testing.assert_close(got.cpu(), expected_gelu(x))


@pytest.mark.cuda
@pytest.mark.usefixtures("needs_gpu")
def test_on_a_gpu_tensor_its_operands_broadcast() -> None:
    a, b, c = torch.randn(4, 1), torch.randn(1, 5), torch.randn(5)
    before = calls["fma"]
    got = fma(a.to(tessiq.gpu), b.to(tessiq.gpu), c.to(tessiq.gpu))
    assert calls["fma"] == before
    torch.testing.assert_close(got.cpu(), a * b + c)


def test_on_the_cpu_reference_device_it_is_the_reference() -> None:
    x = torch.randn(100)
    before = calls["fast_gelu"]
    got = fast_gelu(x.to(tessiq.cpu))
    assert got.device == tessiq.cpu
    assert calls["fast_gelu"] == before + 1
    torch.testing.assert_close(got.cpu(), expected_gelu(x))


@pytest.mark.cuda
@pytest.mark.usefixtures("needs_gpu")
def test_a_type_the_expression_was_not_written_for_runs_the_reference() -> None:
    a, b, c = (torch.randn(8, dtype=torch.float64) for _ in range(3))
    before = calls["fma"]
    got = fma(a.to(tessiq.gpu), b.to(tessiq.gpu), c.to(tessiq.gpu))
    assert calls["fma"] == before + 1
    torch.testing.assert_close(got.cpu(), a * b + c)


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
    class Fma(torch.nn.Module):
        def forward(self, a: torch.Tensor) -> torch.Tensor:
            out: torch.Tensor = fma(a, a, a)
            return out

    a = torch.randn(32, dtype=torch.float64)
    graph = tessiq.capture(Fma(), a)
    recipe = build_recipe(graph, tessiq.gpu, use_cache=False)
    custom = next(n.name for n in graph.graph.nodes if "fma" in str(n.target))
    assert recipe.devices.get(custom, torch.device("cpu")).type == "cpu"
    torch.testing.assert_close(tessiq.execute(recipe)(a)[0], a * a + a)


@pytest.mark.parametrize(
    ("qualname", "expression", "message"),
    [
        ("tessiq_test::bare", "$0 * 1.702", "write scalar_t(1.702) instead"),
        ("tessiq_test::out_of_range", "$0 * $1", "its expression reads $1"),
        ("no_namespace", "$0", "write it as namespace::name"),
    ],
)
def test_a_registration_the_compiler_refuses_leaves_nothing_behind(
    qualname: str, expression: str, message: str
) -> None:
    def reference(x: torch.Tensor) -> torch.Tensor:
        return x

    with pytest.raises(ValueError, match=re.escape(message)):
        tessiq.library.elementwise(qualname, expression)(reference)
    assert not hasattr(torch.ops.tessiq_test, qualname.partition("::")[2] or qualname)


def test_a_reference_must_take_a_fixed_number_of_operands() -> None:
    def variadic(*xs: torch.Tensor) -> torch.Tensor:
        return xs[0]

    with pytest.raises(ValueError, match="positional"):
        tessiq.library.elementwise("tessiq_test::variadic", "$0")(variadic)


@pytest.mark.cuda
@pytest.mark.usefixtures("needs_gpu")
def test_an_expression_that_does_not_compile_fails_the_registration() -> None:
    def reference(x: torch.Tensor) -> torch.Tensor:
        return x

    with pytest.raises(RuntimeError, match="nvrtc"):
        tessiq.library.elementwise("tessiq_test::broken", "($0 +* $0)")(reference)
    assert not hasattr(torch.ops.tessiq_test, "broken")
