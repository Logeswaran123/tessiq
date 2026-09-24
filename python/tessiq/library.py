# Copyright 2026 The Tessiq Authors.
# SPDX-License-Identifier: Apache-2.0
"""Operations added from Python, which the backend then compiles as its own.

An operation registered here is two things at once. It is a PyTorch operation, so a model calls it
like any other and `torch.compile` captures it as a single node. It is also a row in the compiler's
operation table, so the passes claim it, fuse it with its neighbours and emit its expression in the
same loop as theirs. None of tessiq's C++ changes and nothing is rebuilt.

    @tessiq.library.elementwise(
        "demo::fast_gelu",
        "$0 * (scalar_t(1) / (scalar_t(1) + exp(scalar_t(-1.702) * $0)))",
    )
    def fast_gelu(x):
        return x * torch.sigmoid(1.702 * x)

The decorated function is the reference: what the operation computes, written in PyTorch. It is
what runs on the host, on the CPU reference device, and for any element type the expression was not
written for, and it is what the kernel is tested against.
"""

from __future__ import annotations

import inspect
import threading
from collections.abc import Callable, Sequence
from typing import Any

import torch

from tessiq import _C
from tessiq.device import DEVICE_NAME, cuda_ordinal, gpu, gpu_count

#: The element types a registered expression can run on, and their names in the IR.
_DTYPE_NAMES = {torch.float32: "f32", torch.float64: "f64"}


def elementwise(
    qualname: str,
    expression: str,
    *,
    dtypes: Sequence[torch.dtype] = (torch.float32,),
) -> Callable[[Callable[..., torch.Tensor]], Any]:
    """Registers an elementwise operation, decorating the function that computes it in PyTorch.

    Args:
        qualname: The operation's name as `torch.library` spells it, `namespace::name`.
        expression: What one element of the result is, in the form the compiler's own table uses.
            `$0` is the first operand's element and `$1` the second's. A constant is written
            `scalar_t(1.702)` rather than `1.702`, which C++ reads as a double and which would turn
            a float32 kernel into double arithmetic. Maths functions are spelled `exp` rather than
            `expf`, so that one expression serves every element type.
        dtypes: The element types the expression was written for, float32 and float64. A call on
            any other type runs the reference instead.

    Returns:
        A decorator. It takes the reference, whose parameters are the operation's tensor operands,
        and returns the operation, which is called like a function. The operation has no backward:
        register one with its `register_autograd` before using it in training.

    Raises:
        ValueError: The expression reads an operand that does not exist, never reads one that does,
            or writes a floating-point constant bare. Also a name the compiler cannot use, or one
            already registered differently.
        RuntimeError: On a machine with a GPU, the expression does not compile. The message is the
            compiler's own log.
    """
    names = [_dtype_name(dtype) for dtype in dtypes]
    spelling = _spelling(qualname)

    def register(reference: Callable[..., torch.Tensor]) -> Any:
        arity = _arity(reference)
        # The compiler checks the expression before torch hears of the operation, and a GPU
        # compiles it, so an expression that is refused or does not compile fails the import that
        # registered it rather than a model an hour later, and leaves no torch operation behind.
        _C.register_elementwise(spelling, qualname, expression, arity, names)
        device = _OnTheDevice(spelling, arity, reference, frozenset(dtypes))
        if gpu_count() > 0:
            for dtype in dtypes:
                device.kernel(dtype, cuda_ordinal(gpu))

        schema = "(" + ", ".join(f"Tensor x{i}" for i in range(arity)) + ") -> Tensor"
        operation = torch.library.custom_op(qualname, reference, mutates_args=(), schema=schema)
        operation.register_fake(_broadcast_result)
        operation.register_kernel(DEVICE_NAME, device)
        return operation

    return register


def registered() -> list[str]:
    """Every operation registered in this process, as the compiler spells it."""
    return list(_C.registered_ops())


class _OnTheDevice:
    """What a registered operation does when it is called eagerly on a tessiq tensor.

    On a GPU it runs the kernel the compiler builds for the operation on its own, which is the same
    code a compiled graph fuses. That kernel reads every operand at the result's position, so the
    operands are broadcast and packed first. On the CPU reference device, and for an element type
    the expression was not written for, it runs the reference.
    """

    def __init__(
        self,
        spelling: str,
        arity: int,
        reference: Callable[..., torch.Tensor],
        dtypes: frozenset[torch.dtype],
    ) -> None:
        self._spelling = spelling
        self._arity = arity
        self._reference = reference
        self._dtypes = dtypes
        self._kernels: dict[tuple[torch.dtype, int], _C.Kernel] = {}
        self._lock = threading.Lock()

    def kernel(self, dtype: torch.dtype, ordinal: int) -> _C.Kernel:
        """The kernel for `dtype` on CUDA device `ordinal`, compiled on first use."""
        key = (dtype, ordinal)
        with self._lock:
            found = self._kernels.get(key)
            if found is None:
                found = _C.compile_elementwise(
                    self._spelling, self._arity, _dtype_name(dtype), ordinal
                )
                self._kernels[key] = found
            return found

    def __call__(self, *operands: torch.Tensor) -> torch.Tensor:
        device = operands[0].device
        dtype = operands[0].dtype
        ordinal = cuda_ordinal(device)
        alike = all(t.dtype == dtype and t.device == device for t in operands)
        if ordinal < 0 or dtype not in self._dtypes or not alike:
            return self._reference(*(t.cpu() for t in operands)).to(device)

        shape = torch.broadcast_shapes(*(t.shape for t in operands))  # type: ignore[no-untyped-call]
        packed = [t.expand(shape).contiguous().view(-1) for t in operands]
        out = torch.empty(shape, dtype=dtype, device=device)
        self.kernel(dtype, ordinal).launch(packed, out.view(-1))
        return out


def _broadcast_result(*operands: torch.Tensor) -> torch.Tensor:
    """The result's shape and type without computing it, which is all tracing needs."""
    shape = torch.broadcast_shapes(*(t.shape for t in operands))  # type: ignore[no-untyped-call]
    return torch.empty(shape, dtype=operands[0].dtype, device=operands[0].device)


def _spelling(qualname: str) -> str:
    """`demo::fast_gelu` as a graph carries it, `demo.fast_gelu.default`."""
    namespace, sep, name = qualname.partition("::")
    if not sep or not namespace or not name:
        raise ValueError(f"{qualname!r} is not a qualified name; write it as namespace::name")
    return f"{namespace}.{name}.default"


def _dtype_name(dtype: torch.dtype) -> str:
    try:
        return _DTYPE_NAMES[dtype]
    except KeyError:
        raise ValueError(
            f"a registered expression runs on float32 and float64, not {dtype}"
        ) from None


def _arity(reference: Callable[..., torch.Tensor]) -> int:
    """How many tensor operands the reference takes, which is the operation's arity."""
    parameters = inspect.signature(reference).parameters.values()
    kinds = (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    if any(p.kind not in kinds or p.default is not inspect.Parameter.empty for p in parameters):
        raise ValueError(
            f"{reference.__name__} must take its tensor operands as plain positional parameters"
        )
    return len(parameters)
