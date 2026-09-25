#!/usr/bin/env python3
# Copyright 2026 The Tessiq Authors.
# SPDX-License-Identifier: Apache-2.0
"""Print how one model compiles.

The compiler already answers every part of this one call at a time. This runs them together and
lines the answers up, so that "why is this operation on the host" is one command rather than four
and a text editor.

    python tools/tessiq_explain.py encoder --device tessiq:1
    python tools/tessiq_explain.py encoder --sections schedule
    python tools/tessiq_explain.py encoder --dynamic --sections summary,kernels --source
    python tools/tessiq_explain.py rmsnorm --device tessiq:1 --rewrites rms_norm,fast_gelu

The models are the blocks the test suite uses, because they are the ones the rest of the project
quotes numbers for. `tessiq.explain(model, example)` takes any module.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests" / "python"))

import torch  # noqa: E402

from tessiq.explain import SECTIONS, explain  # noqa: E402
from test_models import EncoderBlock, ResidualBlock, RMSNormBlock  # noqa: E402

MODELS = {
    "encoder": (EncoderBlock, (2, 6, 32)),
    "residual": (ResidualBlock, (1, 16, 8, 8)),
    "rmsnorm": (RMSNormBlock, (2, 8, 64)),
}


def parse_shape(text: str) -> tuple[int, ...]:
    try:
        return tuple(int(part) for part in text.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"not a shape: {text!r}, want 4,16,32") from error


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("model", choices=sorted(MODELS), help="which block to report on")
    parser.add_argument(
        "--shape", type=parse_shape, help="input shape, comma separated. Defaults per model."
    )
    parser.add_argument(
        "--device",
        default=None,
        help="where clustered operations would run, such as tessiq:1. Defaults to the host "
        "reference, which has no kernels.",
    )
    parser.add_argument(
        "--dynamic",
        action="store_true",
        help="report on the graph Dynamo captures once a dimension has changed",
    )
    parser.add_argument(
        "--sections",
        default=",".join(SECTIONS),
        help=f"which parts to print, comma separated, from: {', '.join(SECTIONS)}",
    )
    parser.add_argument(
        "--source", action="store_true", help="print the generated CUDA for each kernel"
    )
    parser.add_argument(
        "--rewrites",
        default="",
        help="custom operations whose rewrites to run, comma separated, such as rms_norm",
    )
    args = parser.parse_args()

    factory, default_shape = MODELS[args.model]
    torch.manual_seed(0)
    model = factory().eval()
    example = torch.randn(*(args.shape or default_shape))
    device = torch.device(args.device) if args.device else None

    print(
        explain(
            model,
            example,
            device=device,
            dynamic=args.dynamic,
            sections=tuple(args.sections.split(",")),
            source=args.source,
            rewrites=[name for name in args.rewrites.split(",") if name],
        ),
        end="",
    )


if __name__ == "__main__":
    main()
