# Benchmarks

Empty until phase 6.

Anything quoted as a number in the README or the docs has a script in here that reproduces it.
Report a distribution, not one timing, and drop the warmup.

- `micro/` — per-pattern, fused against unfused
- `models/` — model-level, fusion on and off
- `emitters/` — NVRTC against Triton on the same clusters ([PLAN.md §9.2](../PLAN.md))
