# Tools

| | | |
|---|---|---|
| `tessiq-opt` | Reads textual IR, runs a pass pipeline, prints the result | **built** |
| `gen_ops.py` | Generates the C++ op table and `docs/op_coverage.md` from `ops.toml` | **built** |
| `tessiq_explain.py` | Shows a model's whole compilation: graph, per-pass diffs, kernels, where each step runs | **built** |
| `model_coverage.py` | How much of a real model the backend places on the device | **built** |

`tessiq-opt` is the one that makes the pass suite testable. Text in, text out, no PyTorch and no
GPU involved, so a pass test is a diffable file rather than a fixture.

`tessiq_explain.py` answers the question the others make you assemble by hand. Every part of it is
already reachable one call at a time, and lining four answers up in a text editor is enough work
that "why is this operation on the host" usually goes unasked.

```
$ python tools/tessiq_explain.py encoder --device tessiq:1 --sections schedule
  ...
  expand                   aten.expand.default                interpreted on the device
  clone                    aten.clone.default                 host (moves its inputs)
  view_6                   aten.view.default                  interpreted on the device
  bmm                      aten.bmm.default                   kernel %bmm
  div                      aten.div.Tensor                    kernel %div
  amax                     aten.amax.default                  kernel %amax
  sub                      aten.sub.Tensor                    kernel %exp
  exp                      aten.exp.default                   kernel %exp
```

Three things can happen to a step and they are easy to confuse. It can be covered by a kernel, and
`sub` and `exp` above share one. It can sit on the device with no kernel, which for an operation
that only takes a different view of a tensor costs almost nothing. Or it can be on the host, which
for anything the device was holding costs a transfer each way. `clone` is the one doing that here.

`tessiq.explain(model, example)` is the same report for any module rather than the blocks the test
suite uses.
