# Copyright 2026 The Tessiq Authors.
# SPDX-License-Identifier: Apache-2.0
"""The custom operations compiled into tessiq, and what they did.

A custom operation is defined in C++, in its own directory under `csrc/custom_ops/`: a `.cpp` that
gives its schema, what it computes in ATen, and how to launch its kernel, beside the `.cu` file
that holds the kernel. `csrc/custom_ops/README.md` walks through adding one. Once tessiq is
imported, each is `torch.ops.tessiq.<name>`, and it runs on the GPU both called directly and inside
`torch.compile`.

An operation can also replace a torch one, such as `softmax` replacing `aten::_softmax`. A model
calling the torch operation then runs the custom one wherever it has the types for it.

Nothing about an operation is defined here. This module lists them, compiles them ahead of their
first call, and says how often they ran their reference instead of their kernel.
"""

from __future__ import annotations

from collections.abc import Sequence

from tessiq import _C
from tessiq.device import cuda_ordinal, gpu, gpu_count


def registered() -> list[str]:
    """Every custom operation, as `tessiq::<name>`, sorted."""
    return list(_C.custom_ops())


def overrides() -> dict[str, str]:
    """Each custom operation that replaces a torch one, and the torch operation it replaces.

    Keyed by the custom operation, `tessiq::<name>`. The torch operation is spelled as a graph
    carries it, such as `aten._softmax.default`.
    """
    return {op: replaced for op, replaced, _dtypes in _C.custom_op_overrides()}


def rewrites() -> dict[str, str]:
    """Each custom operation with a pattern, and the name of the pass that rewrites to it.

    A rewrite finds the computation an operation's pattern describes, written out in plain PyTorch,
    and replaces it with one call to the operation. It runs only when a compile asks for it:
    ``torch.compile(model, backend="tessiq", options={"rewrites": ["rms_norm"]})``.
    """
    return dict(_C.custom_op_rewrites())


def rewrite_passes(names: Sequence[str]) -> list[str]:
    """The passes that run the rewrites `names` asks for, in the order asked, each once.

    Args:
        names: Operations, as ``rms_norm`` or ``tessiq::rms_norm``, or their passes, as
            ``rewrite-rms-norm``.

    Raises:
        ValueError: A name with no rewrite. The message lists the ones that have one.
    """
    if isinstance(names, str):
        names = [names]
    known: dict[str, str] = {}
    for op, pass_name in rewrites().items():
        known[op] = known[op.partition("::")[2]] = known[pass_name] = pass_name
    passes: list[str] = []
    for name in names:
        if name not in known:
            available = ", ".join(sorted(op.partition("::")[2] for op in rewrites()))
            raise ValueError(
                f"no custom operation has a rewrite called {name!r}; these do: {available}"
            )
        if known[name] not in passes:
            passes.append(known[name])
    return passes


def check(name: str | None = None) -> list[str]:
    """Compiles custom operations now, rather than the first time each one runs.

    A kernel is compiled on first use, so a mistake in one would otherwise surface in the middle of
    a model. The test suite calls this for every operation, and so can anyone adding one.

    Args:
        name: One operation to compile, as `tessiq::<name>`, or every one when omitted.

    Returns:
        The operations compiled. Empty on a machine with no GPU, where there is nothing to compile
        for.

    Raises:
        KeyError: `name` is not a custom operation.
        RuntimeError: An operation does not compile. The message names it and carries the
            compiler's own log.
    """
    names = registered()
    if name is not None:
        if name not in names:
            raise KeyError(f"{name} is not a custom operation; these are: {', '.join(names)}")
        names = [name]
    if gpu_count() == 0:
        return []
    for each in names:
        _C.compile_custom_op(each, cuda_ordinal(gpu))
    return names


def fallbacks(name: str) -> int:
    """How many calls of `name` on a tessiq tensor ran its reference instead of its kernel.

    A call runs the reference on the CPU reference device, and on a GPU for an element type the
    operation was not written for. The answer is the same either way, which is why this is counted:
    nothing else says that a model is running the slow path.
    """
    return _C.custom_op_fallbacks(name)
