# Copyright 2026 The Tessiq Authors.
# SPDX-License-Identifier: Apache-2.0
"""What folding the bias and GELU into the GEMM's epilogue is worth.

Compiles a linear layer followed by GELU three ways for a tessiq GPU:

- Without a rewrite it is two kernels, cuBLAS's SGEMM with the bias and then a generated GELU
  kernel, which reads the product back.
- With `rewrite-gemm-bias-gelu` it is tessiq::gemm_bias_gelu, a CUTLASS GEMM that applies the bias
  and GELU before it writes.
- With the tanh GELU and `rewrite-gemm-bias-gelu-tanh` it is tessiq::gemm_bias_gelu_tanh, one
  cuBLASLt call whose epilogue does the same. The tanh GELU is a different function, so this column
  compares the libraries' fused epilogues rather than two ways of computing one answer. Without the
  rewrite the tanh GELU runs on the host.

All three do float32 arithmetic.

What is timed is the device: each kernel launched many times between one pair of stream marks, as
`tessiq.costs` does. A whole compiled call also copies its result back to the host, which for these
outputs takes longer than any version's kernels and would hide the difference.

Run with: python bench/gemm_bias_gelu.py
"""

from __future__ import annotations

import statistics

import torch

import tessiq
from tessiq.costs import kernel_costs

SAMPLES = 9

# Tokens, input features, output features: the up-projection of transformer MLPs of a few sizes,
# and a small batch through a wide layer.
SHAPES = [(512, 768, 3072), (4096, 1024, 4096), (8192, 768, 3072), (64, 4096, 4096)]


#: What each column compiles: its GELU and the rewrite it asks for.
VARIANTS = [
    ("cuBLAS + GELU", "none", []),
    ("gemm_bias_gelu", "none", ["gemm_bias_gelu"]),
    ("gemm_bias_gelu_tanh", "tanh", ["gemm_bias_gelu_tanh"]),
]


class Layer(torch.nn.Module):
    def __init__(self, inputs: int, outputs: int, approximate: str) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(inputs, outputs)
        self.approximate = approximate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.gelu(self.linear(x), approximate=self.approximate)


def device_time(recipe: tessiq.Recipe) -> tuple[float, float, float]:
    """Median, lowest and highest over SAMPLES of the kernels' summed device time, in us."""
    samples = sorted(
        sum(cost.microseconds for cost in kernel_costs(recipe, repeats=50)) for _ in range(SAMPLES)
    )
    return statistics.median(samples), samples[0], samples[-1]


def main() -> None:
    needed = {"tessiq::gemm_bias_gelu", "tessiq::gemm_bias_gelu_tanh"}
    if tessiq.gpu_count() == 0 or not needed <= set(tessiq.library.precompiled()):
        print("needs a GPU and a build with nvcc, nothing to measure")
        return

    print(f"torch {torch.__version__}, device time per call in us, median of {SAMPLES} (range)")
    print(f"{'m x k -> n':>20}" + "".join(f"{label:>29}" for label, _, _ in VARIANTS))

    torch.manual_seed(0)
    kernels = {}
    for m, k, n in SHAPES:
        x = torch.randn(m, k)
        cells = []
        for label, approximate, rewrites in VARIANTS:
            torch.manual_seed(1)
            layer = Layer(k, n, approximate).eval()
            with torch.no_grad():
                expected = layer.double()(x.double()).float()
                layer.float()
            recipe = tessiq.build_recipe(
                tessiq.capture(layer, x), tessiq.gpu, use_cache=False, rewrites=rewrites
            )
            # Every version has to give the right numbers before any is timed.
            with torch.no_grad():
                (out,) = tessiq.execute(recipe)(*layer.parameters(), x)
            error = (out - expected).abs().max().item()
            assert error < 1e-4, (label, error)
            kernels[label] = sorted(cost.kind for cost in kernel_costs(recipe, repeats=1))

            median, low, high = device_time(recipe)
            ratio = "" if not cells else f" {cells[0][0] / median:.2f}x"
            cells.append((median, f"{median:.1f} ({low:.0f}-{high:.0f}){ratio}"))
        print(f"{f'{m} x {k} -> {n}':>20}" + "".join(f"{cell:>29}" for _, cell in cells))

    print()
    for label, kinds in kernels.items():
        print(f"{label + ':':<21}{', '.join(kinds)}")


if __name__ == "__main__":
    main()
