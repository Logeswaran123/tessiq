# Copyright 2026 The Tessiq Authors.
# SPDX-License-Identifier: Apache-2.0
"""Elementwise operations called eagerly on a tessiq tensor.

On a GPU each runs one generated kernel there. What the kernel does not cover goes through the CPU
fallback, which the fallback counts show.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch

import tessiq
from conftest import load_operations
from tessiq import _C

#: Every ATen spelling of every elementwise operation in ops.toml.
SPELLINGS = [
    spelling
    for op in load_operations()
    if op["category"] == "elementwise" and op["kernel"]
    for spelling in op["aten"]
]


@pytest.fixture
def needs_gpu() -> None:
    if tessiq.gpu_count() == 0:
        pytest.skip("needs a gpu")


def host_calls() -> int:
    return sum(_C.cpu_fallback_counts().values())


def run(
    fn: Callable[..., torch.Tensor], *args: object, device: torch.device | None = None
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """`fn` on the host and on `device`, and how many calls in between went through the host."""
    device = device or tessiq.gpu
    moved = [arg.to(device) if isinstance(arg, torch.Tensor) else arg for arg in args]
    before = host_calls()
    got = fn(*moved)
    fell_back = host_calls() - before
    return fn(*args), got.cpu(), fell_back


def same(expected: torch.Tensor, got: torch.Tensor) -> None:
    torch.testing.assert_close(got, expected, equal_nan=True)
    assert got.stride() == expected.stride()


def call(spelling: str) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """A call of `spelling` exactly, with a tensor or a number as its second operand."""
    _, name, overload = spelling.split(".")
    op = getattr(getattr(torch.ops.aten, name), overload)
    if overload == "default":
        return lambda x, _: op(x)
    if overload == "Tensor":
        return lambda x, y: op(x, y)
    return lambda x, _: op(x, 0.75)


@pytest.mark.cuda
@pytest.mark.usefixtures("needs_gpu")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("spelling", SPELLINGS)
def test_every_elementwise_operation_runs_on_the_gpu(spelling: str, dtype: torch.dtype) -> None:
    torch.manual_seed(0)
    # Inside every operation's domain: log, sqrt, asin and acos all take (0, 1).
    x = torch.rand(64, 33, dtype=dtype) * 0.8 + 0.1
    y = torch.rand(64, 33, dtype=dtype) * 0.8 + 0.1
    expected, got, fell_back = run(call(spelling), x, y)
    assert fell_back == 0
    same(expected, got)


@pytest.mark.cuda
@pytest.mark.usefixtures("needs_gpu")
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize(
    "fn",
    [torch.add, torch.sub, torch.mul, lambda x, _: torch.neg(x), lambda x, _: torch.relu(x)],
    ids=["add", "sub", "mul", "neg", "relu"],
)
def test_integer_arithmetic_runs_on_the_gpu(
    fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], dtype: torch.dtype
) -> None:
    top = torch.iinfo(dtype).max
    x = torch.tensor([-7, 0, 3, top, top - 1, -top], dtype=dtype)
    y = torch.tensor([2, -5, 3, 1, top, 9], dtype=dtype)
    expected, got, fell_back = run(fn, x, y)
    assert fell_back == 0
    assert torch.equal(got, expected)


@pytest.mark.cuda
@pytest.mark.usefixtures("needs_gpu")
@pytest.mark.parametrize(
    ("left", "right"),
    [
        ((64, 512), (512,)),
        ((64, 512), (64, 1)),
        ((1, 512), (64, 1)),
        ((4, 6, 8), (4, 1, 8)),
        ((64, 512), ()),
    ],
    ids=["row", "column", "outer", "middle", "zero-dim"],
)
def test_operands_broadcast_on_the_gpu(left: tuple[int, ...], right: tuple[int, ...]) -> None:
    torch.manual_seed(0)
    expected, got, fell_back = run(torch.sub, torch.randn(left), torch.randn(right))
    assert fell_back == 0
    same(expected, got)


@pytest.mark.cuda
@pytest.mark.usefixtures("needs_gpu")
@pytest.mark.parametrize(
    "fn",
    [
        lambda x: x * 0.5,
        lambda x: 2 + x,
        lambda x: x / 3,
        lambda x: x**2.5,
        lambda x: x - torch.tensor(1.5),
        lambda x: torch.tensor(4.0, dtype=torch.float64) * x,
    ],
    ids=["mul", "radd", "div", "pow", "zero-dim-host", "zero-dim-double"],
)
def test_a_number_is_passed_to_the_kernel(fn: Callable[[torch.Tensor], torch.Tensor]) -> None:
    torch.manual_seed(0)
    expected, got, fell_back = run(fn, torch.rand(1000) + 0.5)
    assert fell_back == 0
    assert got.dtype == expected.dtype
    same(expected, got)


@pytest.mark.cuda
@pytest.mark.usefixtures("needs_gpu")
@pytest.mark.parametrize(
    "make",
    [
        lambda: torch.randn(48, 64).t(),
        lambda: torch.randn(2, 3, 4, 5).to(memory_format=torch.channels_last),
        lambda: torch.randn(8, 64)[:, ::2],
        lambda: torch.randn(1, 64).expand(8, 64),
    ],
    ids=["transposed", "channels-last", "strided", "expanded"],
)
def test_the_result_has_the_layout_torch_gives(make: Callable[[], torch.Tensor]) -> None:
    torch.manual_seed(0)
    x = make()
    for fn in (torch.exp, lambda t: t * t.contiguous(), lambda t: t - t[:1]):
        expected, got, fell_back = run(fn, x)
        assert fell_back == 0
        same(expected, got)


@pytest.mark.cuda
@pytest.mark.usefixtures("needs_gpu")
def test_special_values_match_the_host() -> None:
    x = torch.tensor([float("nan"), float("inf"), -float("inf"), -0.0, 0.0, 0.5, 2.5, -1.5])
    for fn in (torch.relu, torch.neg, torch.abs, torch.sqrt, torch.log, torch.exp, torch.round):
        expected, got, fell_back = run(fn, x)
        assert fell_back == 0
        same(expected, got)
        # The sign of a NaN means nothing, and the two compute it differently.
        numbers = ~expected.isnan()
        assert torch.equal(torch.signbit(got[numbers]), torch.signbit(expected[numbers])), fn


@pytest.mark.cuda
@pytest.mark.usefixtures("needs_gpu")
@pytest.mark.parametrize(
    ("fn", "x"),
    [
        (lambda x: torch.add(x, x, alpha=2), torch.rand(16)),
        (lambda x: torch.nn.functional.gelu(x, approximate="tanh"), torch.rand(16)),
        (lambda x: x + 2.5, torch.arange(16)),
        (lambda x: x + x.double(), torch.rand(16)),
        (torch.abs, torch.arange(-8, 8)),
        (lambda x: x / x, torch.arange(1, 16)),
        (torch.exp, torch.rand(16, dtype=torch.float16)),
        (lambda x: x.add_(1.0), torch.rand(16)),
    ],
    ids=["alpha", "gelu-tanh", "promotes", "mixed-types", "int-abs", "int-div", "half", "inplace"],
)
def test_what_the_kernel_does_not_cover_runs_on_the_host(
    fn: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor
) -> None:
    expected, got, fell_back = run(fn, x.clone())
    assert fell_back >= 1
    assert got.dtype == expected.dtype
    torch.testing.assert_close(got, expected)


def test_on_the_cpu_reference_device_it_runs_on_the_host() -> None:
    torch.manual_seed(0)
    expected, got, fell_back = run(torch.add, torch.randn(16), torch.randn(16), device=tessiq.cpu)
    assert fell_back == 1
    same(expected, got)


@pytest.mark.cuda
@pytest.mark.usefixtures("needs_gpu")
def test_an_empty_result_launches_nothing() -> None:
    expected, got, fell_back = run(torch.exp, torch.empty(0, 3))
    assert fell_back == 0
    assert got.shape == expected.shape


@pytest.mark.cuda
@pytest.mark.usefixtures("needs_gpu")
def test_shapes_that_do_not_broadcast_raise_torchs_error() -> None:
    with pytest.raises(RuntimeError, match="must match the size"):
        torch.randn(3, device=tessiq.gpu) + torch.randn(4, device=tessiq.gpu)


@pytest.mark.cuda
@pytest.mark.usefixtures("needs_gpu")
def test_gradients_flow_through_eager_kernels() -> None:
    torch.manual_seed(0)
    x = torch.rand(32, requires_grad=True)
    on_gpu = x.detach().to(tessiq.gpu).requires_grad_()
    torch.autograd.backward(torch.exp(x * 2.0), torch.ones(32))
    torch.autograd.backward(torch.exp(on_gpu * 2.0), torch.ones(32, device=tessiq.gpu))
    assert on_gpu.grad is not None
    assert x.grad is not None
    torch.testing.assert_close(on_gpu.grad.cpu(), x.grad)
