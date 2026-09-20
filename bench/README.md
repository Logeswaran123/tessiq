# Benchmarks

Anything quoted as a number in the README or the docs has a script in here that reproduces it.
Report a distribution, not one timing, and drop the warmup.

- `compile_overhead.py` measures what the recipe cache saves on compilation and what the
  move analysis saves per call. Both are overhead rather than arithmetic, which is all there
  is to measure until kernels exist.

- `micro/` holds per-pattern comparisons, fused against unfused.
- `models/` holds model-level runs, fusion on and off.
- `emitters/` compares NVRTC against Triton on the same clusters. See
  [PHASES.md](../PHASES.md), phase 9.
