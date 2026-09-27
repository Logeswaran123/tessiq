# Benchmarks

Every performance number in the docs comes from one of these scripts. Run them from the repo root
after installing tessiq, e.g. `python bench/against_torch.py`. All except `compile_overhead.py`
need a GPU.

| Script | Measures |
|---|---|
| [`against_torch.py`](against_torch.py) | Generated kernel vs Inductor on `(a * b + c).relu().sigmoid()`; whole calls from host tensors for that chain and a transformer encoder block |
| [`fusion.py`](fusion.py) | A fused kernel vs the same recipe with kernels removed (each op through the host fallback) |
| [`reduction.py`](reduction.py) | Reduction kernels vs PyTorch across axis layouts |
| [`gemm_bias_gelu.py`](gemm_bias_gelu.py) | Linear + GELU device time: cuBLAS + GELU kernel, CUTLASS `gemm_bias_gelu`, cuBLASLt `gemm_bias_gelu_tanh` |
| [`compile_overhead.py`](compile_overhead.py) | Recipe build vs cache hit, and the cost of moving arguments at every step |
| [`where_the_time_goes.py`](where_the_time_goes.py) | Host-side breakdown of one call into kernels, transfers and interpreted steps. Pass a path to write a Chrome trace |
| [`where_the_gap_is.py`](where_the_gap_is.py) | Gap to eager CUDA, split into kernel device time and host time |

Scripts report medians and exclude warmup. Device time is measured by launching each kernel many
times between one pair of events (`tessiq.kernel_costs`), since a compiled call also includes the
copy of its results to the host.
