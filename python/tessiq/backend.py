# Copyright 2026 The Tessiq Authors.
# SPDX-License-Identifier: Apache-2.0
"""The ``torch.compile`` backend.

Declared as a ``torch_dynamo_backends`` entry point, so ``torch.compile(model, backend="tessiq")``
resolves whether or not the caller imported tessiq first. Nothing here calls
``register_backend``: that would collide with the entry point, since torch rejects a second
registration under the same name.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import torch
from torch._decomp import core_aten_decompositions
from torch._dynamo.backends.common import aot_autograd
from torch._functorch._aot_autograd.utils import make_boxed_func

from tessiq.device import DEVICE_NAME, _resolve
from tessiq.executor import execute
from tessiq.recipe import build_recipe

_OPTIONS = frozenset({"device"})


def compile_graph(
    graph_module: torch.fx.GraphModule,
    example_inputs: Sequence[Any],
    *,
    mode: str | None = None,
    options: dict[str, Any] | None = None,
) -> Callable[..., Any]:
    """Compile one graph captured by Dynamo.

    Args:
        graph_module: The graph Dynamo captured.
        example_inputs: Fake tensors standing in for the arguments, used by AOTAutograd to trace.
        mode: Not supported. tessiq has one compilation strategy.
        options: ``{"device": d}`` picks the tessiq device clusters run on. Defaults to
            ``tessiq.cpu``.

    Returns:
        A callable Dynamo installs in place of the captured graph.

    Raises:
        ValueError: If ``mode`` is given, if ``options`` holds a key that is not recognised, or if
            ``options["device"]`` does not name a tessiq device.
    """
    if mode is not None:
        raise ValueError(
            f"backend '{DEVICE_NAME}' has no compilation modes, got mode={mode!r}; "
            f"drop the argument"
        )
    device = _device_from(options)

    def compile_forward(traced: torch.fx.GraphModule, _inputs: Sequence[Any]) -> Callable[..., Any]:
        # Boxed, so AOTAutograd hands the arguments over in a list it can drop its own reference
        # to. Unboxed it holds every input alive for the length of the call, which in training is
        # the whole activation set.
        return make_boxed_func(execute(build_recipe(traced, device)))

    # AOTAutograd is what makes the graph importable: it functionalises the mutations Dynamo leaves
    # in and lowers to core ATen, which is the op set the passes and the support table assume. The
    # decompositions are the reason a model using an op tessiq never heard of can still cluster,
    # because they rewrite it into ones tessiq has.
    #
    # Built per call because AotAutograd stores its kwargs and rewrites them on the first use.
    compiler = aot_autograd(fw_compiler=compile_forward, decompositions=core_aten_decompositions())
    return compiler(graph_module, example_inputs)


def _device_from(options: dict[str, Any] | None) -> torch.device:
    if not options:
        return torch.device(DEVICE_NAME, 0)
    unknown = sorted(set(options) - _OPTIONS)
    if unknown:
        raise ValueError(
            f"unknown option(s) for backend '{DEVICE_NAME}': {', '.join(unknown)}; "
            f"supported: {', '.join(sorted(_OPTIONS))}"
        )
    return torch.device(DEVICE_NAME, _resolve(options.get("device")))
