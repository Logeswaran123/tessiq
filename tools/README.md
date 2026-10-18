# Tools

## tessiq-opt

Runs a pass pipeline over textual IR. Built with the C++ core, no PyTorch needed; after a CMake
build it's at `build/tools/tessiq-opt`. Reads a file or stdin.

```console
$ cat graph.tsr
graph @demo(%a: f32[4,8], %b: f32[4,8]) -> f32[4,8] {
  %s0 = add(%a, %b) : f32[4,8]
  %s1 = add(%a, %b) : f32[4,8]
  %unused = neg(%a) : f32[4,8]
  %out = mul(%s0, %s1) : f32[4,8]
  return %out
}
$ build/tools/tessiq-opt --passes=cse,dead-node-elimination graph.tsr
graph @demo(%a: f32[4,8], %b: f32[4,8]) -> f32[4,8] {
  %s0 = add(%a, %b) : f32[4,8]
  %out = mul(%s0, %s0) : f32[4,8]
  return %out
}
```

Other flags: `--list`, `--verify-only`, `--stats` (per-pass timing), `--memory-plan`, and
`--emit-cuda` (CUDA source per fusion group). Custom-op `rewrite-*` and `override-*` passes are
registered by the Python extension, so run those with `tessiq.run_passes()`.

## lint_cpp.py

Runs clang-format and clang-tidy over the tracked C++ sources, as CI does. clang-tidy reads the
build's compilation database, so configure first.

```bash
python tools/lint_cpp.py --build-dir build            # both
python tools/lint_cpp.py --format                     # clang-format only
python tools/lint_cpp.py --tidy --build-dir build -j 8
```

## tessiq_explain.py

Runs `tessiq.explain` on a block from `tests/python/test_models.py` (`encoder`, `residual`,
`rmsnorm`):

```bash
python tools/tessiq_explain.py encoder --device tessiq:1
python tools/tessiq_explain.py residual --device tessiq:1 --sections schedule
python tools/tessiq_explain.py rmsnorm --device tessiq:1 --rewrites rms_norm,fast_gelu
```

`--device` defaults to `tessiq:0`, which has no kernels. `--sections` takes any of `summary`,
`graph`, `passes`, `kernels`, `schedule`. `--source` prints generated CUDA, `--dynamic` reports on
the symbolic recompile, and `--shape` overrides the input shape.

The schedule shows where each op runs:

```
  convolution    aten.convolution.default   kernel %convolution (moves its inputs)
  _to_copy       aten._to_copy.default      host
  add            aten.add.Tensor            kernel %mul (moves its inputs)
  sqrt           aten.sqrt.default          kernel %mul
  mul            aten.mul.Tensor            kernel %mul
  unsqueeze      aten.unsqueeze.default     interpreted on the device (moves its inputs)
```

`kernel %mul` means the op is covered by that kernel; `add`, `sqrt` and `mul` are fused into one.
`interpreted on the device` means no kernel, run through the dispatcher on device tensors (cheap
for views). `host` costs a transfer each way. `(moves its inputs)` marks steps whose arguments are
copied to their device first.

## model_coverage.py

```console
$ python tools/model_coverage.py
residual block: 37 of 41 operations on the device (90%), 37 claimable
  left on the host                                 count
  aten._to_copy.default                                4

encoder block: 76 of 76 operations on the device (100%), 76 claimable
```

Claimable means the backend implements it. On the device can be lower: a cluster of nothing but
views stays where its input is.

## gen_ops.py

Regenerates `csrc/tessiq/passes/op_table.inc` and `docs/op_coverage.md` from `ops.toml`.
`--check` verifies they are current; a test runs it.
