# Copyright 2026 The Tessiq Authors.
# SPDX-License-Identifier: Apache-2.0
"""The ``torch.compile`` backend.

Declared as a ``torch_dynamo_backends`` entry point, so ``torch.compile(model, backend="tessiq")``
resolves whether or not the caller imported tessiq first. Nothing here calls
``register_backend``: that would collide with the entry point, since torch rejects a second
registration under the same name.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from typing import Any

import torch
from torch._dynamo.backends.common import aot_autograd
from torch._functorch._aot_autograd.utils import make_boxed_func

from tessiq.decompositions import decompositions
from tessiq.device import DEVICE_NAME, _resolve
from tessiq.executor import execute
from tessiq.library import rewrite_passes
from tessiq.recipe import build_recipe
from tessiq.shapes import is_static

_log = logging.getLogger(__name__)

_OPTIONS = frozenset({"device", "rewrites"})


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
            ``tessiq.cpu``. ``{"rewrites": ["rms_norm", ...]}`` runs those custom operations'
            rewrites, which replace each computation their patterns describe with one call.

    Returns:
        A callable Dynamo installs in place of the captured graph.

    Raises:
        ValueError: If ``mode`` is given, if ``options`` holds a key that is not recognised, if
            ``options["device"]`` does not name a tessiq device, or if ``options["rewrites"]``
            names an operation with no rewrite.
    """
    if mode is not None:
        raise ValueError(
            f"backend '{DEVICE_NAME}' has no compilation modes, got mode={mode!r}; "
            f"drop the argument"
        )
    device = _device_from(options)
    rewrites = tuple((options or {}).get("rewrites", ()))
    # Checked here, so a misspelt name fails the torch.compile call rather than the first run.
    rewrite_passes(rewrites)

    def compile_forward(traced: torch.fx.GraphModule, _inputs: Sequence[Any]) -> Callable[..., Any]:
        try:
            recipe = build_recipe(traced, device, rewrites=rewrites)
            _log.debug(
                "compiled a graph with %s shapes: %d kernel(s)",
                "known" if is_static(traced) else "symbolic",
                len(recipe.kernels),
            )
        except TypeError as error:
            # TypeError means the compiler cannot represent the graph (a malformed one raises
            # ValueError instead), so run it uncompiled. Dynamo recompiles with symbolic shapes the
            # second time a dimension changes, so this path is not rare.
            _log.debug("running the graph without compiling it: %s", error)
            return traced.forward

        # Boxed, so AOTAutograd can drop its references to the inputs during the call. Unboxed it
        # keeps them all alive, which in training is every saved activation.
        return make_boxed_func(execute(recipe))

    # AOTAutograd functionalises the graph and lowers it to core ATen, the op set the passes
    # assume. The decompositions rewrite ops tessiq does not know into ones it does.
    #
    # Built per call because aot_autograd stores its kwargs and rewrites them on first use.
    compiler = aot_autograd(fw_compiler=compile_forward, decompositions=decompositions())
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
