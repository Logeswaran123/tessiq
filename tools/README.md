# Tools

| | | |
|---|---|---|
| `tessiq-opt` | Reads textual IR, runs a pass pipeline, prints the result | phase 3 |
| `gen_ops.py` | Generates C++ and docs from `ops/*.yaml` | phase 7 |
| `tessiq-explain` | Shows a model's whole compilation: graph, per-pass diffs, kernel schedule | phase 8 |

`tessiq-opt` is the one that makes the pass suite testable. Text in, text out, no PyTorch and no
GPU involved, so a pass test is a diffable file rather than a fixture.
