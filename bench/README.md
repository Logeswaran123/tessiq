# Benchmarks

Anything quoted as a number in the README or the docs has a script in here that reproduces it.
Report a distribution, not one timing, and drop the warmup.

- `compile_overhead.py` measures what the recipe cache saves on compilation and what the
  move analysis saves per call. Both are overhead rather than arithmetic, which is all there
  is to measure until kernels exist.
- `fusion.py` compares a generated kernel against the same graph with the kernels removed,
  which is what fusion buys inside tessiq.
- `against_torch.py` compares against PyTorch itself, both the kernel alone and the whole
  call. The two answers point opposite ways, which is why both are there.
- `where_the_time_goes.py` splits a call into kernels, transfers and host work, and writes a
  chrome trace. It answers why a block is behind rather than by how much.

- `where_the_gap_is.py` splits the distance to eager CUDA into what the generated code costs the
  device and what everything around it costs the host. The kernels are replayed many times under
  one pair of stream marks, so the cost of marking divides away.

- `gemm_bias_gelu.py` times a linear layer followed by GELU as cuBLAS's SGEMM and a GELU kernel,
  and as one CUTLASS GEMM with the bias and GELU in its epilogue. It measures device time, since a
  compiled call's copy of the result back to the host would hide the difference.

- `micro/` holds per-pattern comparisons, fused against unfused.
- `models/` holds model-level runs, fusion on and off.
- `emitters/` compares NVRTC against Triton on the same clusters. See
  [PHASES.md](../PHASES.md), phase 14.
