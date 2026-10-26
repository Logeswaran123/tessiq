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


def call(
    spelling: str, form: str = "functional"
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """A call of `spelling` exactly, or of its in-place or out= form, with a tensor or a number as
    its second operand."""
    _, name, overload = spelling.split(".")
    if form == "in-place":
        op = getattr(
            getattr(torch.ops.aten, name + "_"), {"Tensor_Scalar": "Scalar"}.get(overload, overload)
        )
    elif form == "out":
        op = getattr(
            getattr(torch.ops.aten, name),
            "out" if overload in ("default", "Tensor") else overload + "_out",
        )
    else:
        op = getattr(getattr(torch.ops.aten, name), overload)

    def called(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        operands = [x] if overload == "default" else [x, y] if overload == "Tensor" else [x, 0.75]
        result: torch.Tensor = (
            op(*operands, out=torch.empty_like(x)) if form == "out" else op(*operands)
        )
        return result

    return called


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
@pytest.mark.parametrize("form", ["in-place", "out"])
@pytest.mark.parametrize("spelling", SPELLINGS)
def test_in_place_and_out_forms_run_on_the_gpu(spelling: str, form: str) -> None:
    torch.manual_seed(0)
    x = torch.rand(64, 33) * 0.8 + 0.1
    y = torch.rand(64, 33) * 0.8 + 0.1
    expected, got, fell_back = run(call(spelling, form), x, y)
    assert fell_back == 0
    same(expected, got)


@pytest.mark.cuda
@pytest.mark.usefixtures("needs_gpu")
def test_an_in_place_call_returns_its_tensor_and_writes_through_a_view() -> None:
    x = torch.arange(24, dtype=torch.float32).view(4, 6)
    on_gpu = x.to(tessiq.gpu)
    ones = torch.ones(4).to(tessiq.gpu)
    before = host_calls()
    rows = on_gpu[1:3]
    assert rows.mul_(2.0) is rows
    on_gpu[:, 1:].T.add_(ones)
    assert host_calls() == before + 1, "only the strided column view falls back"
    x[1:3].mul_(2.0)
    x[:, 1:].T.add_(torch.ones(4))
    assert torch.equal(on_gpu.cpu(), x)


@pytest.mark.cuda
@pytest.mark.usefixtures("needs_gpu")
@pytest.mark.parametrize(
    "fn",
    [
        lambda x, y: torch.add(x, y, out=torch.empty(0, device=x.device)),
        lambda x, y: x.add_(y, alpha=2),
        lambda x, y: x[:1].add_(y),
    ],
    ids=["out-resized", "alpha", "broadcast-into-self"],
)
def test_forms_the_kernel_cannot_write_run_on_the_host(
    fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> None:
    torch.manual_seed(0)
    x, y = torch.rand(4, 8), torch.rand(4, 8)
    try:
        expected: torch.Tensor | None = fn(x.clone(), y)
    except RuntimeError:
        expected = None
    on_gpu = x.to(tessiq.gpu)
    before = host_calls()
    if expected is None:
        with pytest.raises(RuntimeError):
            fn(on_gpu, y.to(tessiq.gpu))
    else:
        torch.testing.assert_close(fn(on_gpu, y.to(tessiq.gpu)).cpu(), expected)
    assert host_calls() == before + 1


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
    ],
    ids=["alpha", "gelu-tanh", "promotes", "mixed-types", "int-abs", "int-div", "half"],
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


# --- matmuls -------------------------------------------------------------------------------------


def close_products(got: torch.Tensor, expected: torch.Tensor) -> None:
    """A product sums `k` terms, which cuBLAS adds in another order than the CPU does."""
    torch.testing.assert_close(got, expected, rtol=1e-5, atol=1e-4)


@pytest.mark.cuda
@pytest.mark.usefixtures("needs_gpu")
@pytest.mark.parametrize(
    ("fn", "shapes"),
    [
        (torch.mm, [(64, 48), (48, 80)]),
        (lambda a, b: a.t() @ b, [(48, 64), (48, 80)]),
        (lambda a, b: a @ b.t(), [(64, 48), (80, 48)]),
        (lambda a, b: a.t() @ b.t(), [(48, 64), (80, 48)]),
        (lambda a, b: a[:, ::2] @ b, [(64, 96), (48, 80)]),
        (torch.addmm, [(80,), (64, 48), (48, 80)]),
        (torch.addmm, [(64, 80), (64, 48), (48, 80)]),
        (torch.addmm, [(64, 1), (64, 48), (48, 80)]),
        (torch.addmm, [(), (64, 48), (48, 80)]),
        (torch.nn.functional.linear, [(64, 48), (80, 48), (80,)]),
        (torch.bmm, [(4, 16, 8), (4, 8, 12)]),
        (lambda a, b: torch.bmm(a, b.transpose(1, 2)), [(4, 16, 8), (4, 12, 8)]),
        (torch.matmul, [(2, 4, 6, 8), (2, 4, 8, 6)]),
    ],
    ids=[
        "mm",
        "lhs-transposed",
        "rhs-transposed",
        "both-transposed",
        "strided",
        "addmm-row",
        "addmm-full",
        "addmm-column",
        "addmm-zero-dim",
        "linear",
        "bmm",
        "bmm-transposed",
        "matmul-4d",
    ],
)
def test_matmuls_run_on_the_gpu(
    fn: Callable[..., torch.Tensor], shapes: list[tuple[int, ...]]
) -> None:
    torch.manual_seed(0)
    expected, got, fell_back = run(fn, *(torch.randn(shape) for shape in shapes))
    assert fell_back == 0
    close_products(got, expected)


@pytest.mark.cuda
@pytest.mark.usefixtures("needs_gpu")
@pytest.mark.parametrize(
    ("fn", "dtype"),
    [
        (lambda c, a, b: torch.addmm(c, a, b, beta=0.5), torch.float32),
        (lambda c, a, b: torch.addmm(c, a, b, alpha=2), torch.float32),
        (lambda _, a, b: torch.mm(a, b), torch.float64),
    ],
    ids=["beta", "alpha", "float64"],
)
def test_matmuls_cublas_does_not_take_run_on_the_host(
    fn: Callable[..., torch.Tensor], dtype: torch.dtype
) -> None:
    torch.manual_seed(0)
    args = (
        torch.randn(8, dtype=dtype),
        torch.randn(4, 6, dtype=dtype),
        torch.randn(6, 8, dtype=dtype),
    )
    expected, got, fell_back = run(fn, *args)
    assert fell_back == 1
    close_products(got, expected)


@pytest.mark.cuda
@pytest.mark.usefixtures("needs_gpu")
def test_a_linear_layer_trains_on_the_gpu() -> None:
    torch.manual_seed(0)
    layer = torch.nn.Linear(48, 80)
    x = torch.randn(64, 48)
    on_gpu = torch.nn.Linear(48, 80).to(tessiq.gpu)
    on_gpu.load_state_dict(layer.state_dict())
    torch.autograd.backward(layer(x), torch.ones(64, 80))
    torch.autograd.backward(on_gpu(x.to(tessiq.gpu)), torch.ones(64, 80).to(tessiq.gpu))
    for theirs, ours in zip(layer.parameters(), on_gpu.parameters(), strict=True):
        assert theirs.grad is not None
        assert ours.grad is not None
        close_products(ours.grad.cpu(), theirs.grad)


# --- convolution ---------------------------------------------------------------------------------


@pytest.mark.cuda
@pytest.mark.usefixtures("needs_gpu")
@pytest.mark.parametrize(
    ("fn", "shapes"),
    [
        (torch.nn.functional.conv2d, [(2, 16, 8, 8), (32, 16, 3, 3)]),
        (
            lambda x, w, b: torch.nn.functional.conv2d(x, w, b, padding=1),
            [(2, 16, 8, 8), (32, 16, 3, 3), (32,)],
        ),
        (
            lambda x, w: torch.nn.functional.conv2d(x, w, stride=2, dilation=2, padding=2),
            [(2, 16, 9, 9), (32, 16, 3, 3)],
        ),
        (lambda x, w: torch.nn.functional.conv2d(x, w, groups=4), [(2, 16, 8, 8), (32, 4, 3, 3)]),
        (
            lambda x, w: torch.nn.functional.conv2d(x, w, groups=16, padding=1),
            [(2, 16, 8, 8), (16, 1, 3, 3)],
        ),
        (torch.nn.functional.conv1d, [(2, 4, 16), (8, 4, 3)]),
    ],
    ids=["plain", "bias-padding", "stride-dilation", "groups", "depthwise", "conv1d"],
)
def test_convolutions_run_on_the_gpu(
    fn: Callable[..., torch.Tensor], shapes: list[tuple[int, ...]]
) -> None:
    torch.manual_seed(0)
    expected, got, fell_back = run(fn, *(torch.randn(shape) for shape in shapes))
    assert fell_back == 0
    close_products(got, expected)
    assert got.stride() == expected.stride()


@pytest.mark.cuda
@pytest.mark.usefixtures("needs_gpu")
@pytest.mark.parametrize(
    ("fn", "x", "w"),
    [
        (
            torch.nn.functional.conv2d,
            torch.randn(2, 16, 8, 8).to(memory_format=torch.channels_last),
            torch.randn(32, 16, 3, 3),
        ),
        (torch.nn.functional.conv_transpose2d, torch.randn(2, 16, 8, 8), torch.randn(16, 8, 3, 3)),
        (
            torch.nn.functional.conv2d,
            torch.randn(2, 16, 8, 8).double(),
            torch.randn(32, 16, 3, 3).double(),
        ),
    ],
    ids=["channels-last", "transposed", "float64"],
)
def test_convolutions_cudnn_does_not_take_run_on_the_host(
    fn: Callable[..., torch.Tensor], x: torch.Tensor, w: torch.Tensor
) -> None:
    expected, got, fell_back = run(fn, x, w)
    assert fell_back == 1
    close_products(got, expected)
    assert got.stride() == expected.stride()
