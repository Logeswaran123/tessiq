# Benchmarks

Every performance number in the docs comes from one of these scripts. Run them from the repo root
after installing tessiq, e.g. `python bench/against_torch.py`. All except `compile_overhead.py`
need a GPU.

| Script | Measures |
|---|---|
| [`against_torch.py`](against_torch.py) | Generated kernel vs Inductor on `(a * b + c).relu().sigmoid()`, in L2 and from DRAM; whole calls from host tensors for that chain and a transformer encoder block |
| [`fusion.py`](fusion.py) | A fused kernel vs the same recipe with kernels removed (an eager kernel per op), whole calls from host tensors |
| [`eager.py`](eager.py) | Eager elementwise calls on a GPU tensor against the host fallback they replaced and PyTorch's CUDA kernels, in L2 and from DRAM; whole model blocks run eagerly on tessiq, CUDA and the CPU |
| [`reduction.py`](reduction.py) | Reduction kernels vs PyTorch across axis layouts, in L2 and from DRAM |
| [`gemm_bias_gelu.py`](gemm_bias_gelu.py) | Linear + GELU device time: cuBLAS + GELU kernel, CUTLASS `gemm_bias_gelu`, cuBLASLt `gemm_bias_gelu_tanh` |
| [`compile_overhead.py`](compile_overhead.py) | Recipe build vs cache hit, and the cost of moving arguments at every step |
| [`where_the_time_goes.py`](where_the_time_goes.py) | Host-side breakdown of one call into kernels, transfers and interpreted steps. Pass a path to write a Chrome trace |
| [`where_the_gap_is.py`](where_the_gap_is.py) | Gap to eager CUDA: the kernels' back-to-back device time (`tessiq.kernel_costs`, gaps between kernels included) against the call |

Scripts report medians and exclude warmup. Kernel comparisons use CUPTI's record of each kernel
([`kernel_timing.py`](kernel_timing.py)), so a raw launch and a framework call are timed the same
way, at working sets chosen from the GPU's L2 size so that neither sits near it. GB/s is 10^9
bytes per second.
