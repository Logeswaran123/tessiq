# Benchmarks

*Arrives in phase 5 — see [`PLAN.md` §7.9](../PLAN.md).*

Every performance number published in the README or the docs maps to a script here that anyone can
run. Benchmarks report a distribution rather than a single timing, with warmup excluded.

Planned layout:

- `micro/` — per-pattern: fused versus unfused, emitter versus emitter
- `models/` — model-level, fusion on and off
- `emitters/` — NVRTC versus Triton on identical clusters ([`PLAN.md` §9.2](../PLAN.md))
