# Copyright 2026 The Tessiq Authors.
# SPDX-License-Identifier: Apache-2.0
"""What folding the bias and GELU into the GEMM's epilogue is worth.

Compiles a linear layer followed by GELU twice for a tessiq GPU. Without the rewrite it is two
kernels, cuBLAS's SGEMM with the bias and then a generated GELU kernel, which reads the product
back. With `rewrite-gemm-bias-gelu` it is one, tessiq::gemm_bias_gelu, a CUTLASS GEMM that applies
the bias and GELU before it writes. Both do float32 arithmetic.

What is timed is the device: each kernel launched many times between one pair of stream marks, as
`tessiq.costs` does. A whole compiled call also copies its result back to the host, which for these
outputs takes longer than both versions' kernels and would hide the difference.

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


class Layer(torch.nn.Module):
    def __init__(self, inputs: int, outputs: int) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(inputs, outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.gelu(self.linear(x))


def device_time(recipe: tessiq.Recipe) -> tuple[float, float, float]:
    """Median, lowest and highest over SAMPLES of the kernels' summed device time, in us."""
    samples = sorted(
        sum(cost.microseconds for cost in kernel_costs(recipe, repeats=50)) for _ in range(SAMPLES)
    )
    return statistics.median(samples), samples[0], samples[-1]


def main() -> None:
    if tessiq.gpu_count() == 0 or "tessiq::gemm_bias_gelu" not in tessiq.library.precompiled():
        print("needs a GPU and a build with nvcc, nothing to measure")
        return

    print(f"torch {torch.__version__}, device time per call, median of {SAMPLES} (range)")
    print(f"{'m x k -> n':>20} {'cuBLAS + GELU':>26} {'gemm_bias_gelu':>26} {'ratio':>6}")

    torch.manual_seed(0)
    for m, k, n in SHAPES:
        layer = Layer(k, n).eval()
        x = torch.randn(m, k)
        with torch.no_grad():
            expected = layer.double()(x.double()).float()
            layer.float()

        timings = []
        for rewrites in ([], ["gemm_bias_gelu"]):
            recipe = tessiq.build_recipe(
                tessiq.capture(layer, x), tessiq.gpu, use_cache=False, rewrites=rewrites
            )
            # Both have to give the right numbers before either is timed.
            with torch.no_grad():
                (out,) = tessiq.execute(recipe)(*layer.parameters(), x)
            error = (out - expected).abs().max().item()
            assert error < 1e-4, (rewrites, error)
            kinds = sorted(cost.kind for cost in kernel_costs(recipe, repeats=1))
            timings.append((device_time(recipe), kinds))

        ((b_median, b_low, b_high), b_kinds), ((r_median, r_low, r_high), r_kinds) = timings
        print(
            f"{f'{m} x {k} -> {n}':>20} "
            f"{b_median:>9.1f} us ({b_low:.0f}-{b_high:.0f}) "
            f"{r_median:>9.1f} us ({r_low:.0f}-{r_high:.0f}) "
            f"{b_median / r_median:>5.2f}x"
        )

    print(f"\nkernels without the rewrite: {', '.join(b_kinds)}")
    print(f"kernels with it:             {', '.join(r_kinds)}")


if __name__ == "__main__":
    main()
