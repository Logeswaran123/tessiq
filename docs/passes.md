# Passes

A pass reads the IR graph and changes it or annotates it. This page covers the order a compile runs
them in and why, the passes that exist but do not run, how to run any of them yourself, and how to
add one.

## The pipeline a compile runs

Every `torch.compile` with `backend="tessiq"` runs these over each graph, top to bottom, each once:

```
rewrite-<name>      only the ones the compile asks for, in the order it names them
override-<name>     one per custom operation that replaces a torch one, sorted by name
mark-supported
layout-assign
cluster
fuse-elementwise
```

With what ships today, a compile that asks for both rewrites runs:

```
rewrite-rms-norm, rewrite-fast-gelu, override-softmax,
mark-supported, layout-assign, cluster, fuse-elementwise
```

The order is assembled in `csrc/tessiq/torch/partition.cpp`, in three steps:

- `kPipeline` is the fixed part, from `mark-supported` down.
- `default_pipeline()` puts the override passes in front of it. `override_passes()` in `custom.cpp`
  lists them.
- `prepare()` puts the rewrites the compile named in front of that.

`tessiq.default_pipeline()` returns the first two parts. `tessiq.explain(model, x, rewrites=[...])`
runs all three, one pass at a time, and prints what each one changed.

### Why each one sits where it does

| pass | what it does | why here |
|---|---|---|
| `rewrite-<name>` | Replaces a computation written out op by op, such as RMSNorm, with one call to a custom operation. | First, so every later pass sees the custom operation and none sees the operations it absorbed. A pattern is written against the torch operations AOTAutograd emits, including one an override would replace. |
| `override-<name>` | Makes each node of a torch operation, such as `aten._softmax`, a node of the custom operation that replaces it, where the custom operation declares the types. | Before `mark-supported`, so the node is claimed as the custom operation. |
| `mark-supported` | Records on each node, as `tessiq.supported`, whether the backend can run it: the operation has a kernel, the dtypes are ones it takes, and its extents are ones it can handle. It changes nothing else. | Everything after it reads that annotation. It needs nothing before it. |
| `layout-assign` | Inserts a `tessiq.contiguous` copy wherever a claimed elementwise or reduction operation reads a value that is not packed. | After `mark-supported`, because only claimed readers need the copy. Before `cluster`, because the copy is a node that clustering has to see. |
| `cluster` | Groups neighbouring claimed nodes into clusters, which run on the device as a unit, and keeps the graph of clusters free of cycles. | Needs `tessiq.supported`. |
| `fuse-elementwise` | Within a cluster, merges chains of elementwise operations whose intermediate values have one reader into fusion groups. Each group becomes one kernel. | Needs `tessiq.cluster`. It is last because nothing after it changes the graph. |

After the pipeline, the compiler stops changing the graph. It plans memory from it and builds
kernels for it: a fused loop per fusion group, a reduction, a matmul or a convolution per node that
is one, and a launch per call to a custom operation with a kernel of its own.

Each pass's source states what it assumes and what it guarantees, under `Precondition:` and
`Postcondition:`. The order above follows from those.

### What a compile can change

Only which rewrites run, with `options={"rewrites": ["rms_norm"]}`. The overrides run in every
compile, and the fixed part is the backend's, not a per-graph choice. A pass that is not in the
pipeline runs only when you name it yourself, as below.

One other pipeline exists. When an elementwise custom operation such as `fast_gelu` is called
outside `torch.compile`, it compiles a one-node graph with `mark-supported, cluster,
fuse-elementwise`. `layout-assign` is left out because every operand there is already packed.

## Passes that exist and do not run in a compile

| pass | what it does |
|---|---|
| `dead-node-elimination` | Removes nodes whose results nothing reads. |
| `cse` | Replaces a repeated computation with the first one. |
| `const-fold` | Evaluates scalar arithmetic on constants. |
| `algebraic-simplify` | Replaces an operation whose result is already available, such as `x * 1`. |
| `shape-infer` | Propagates shapes and dtypes, and checks the ones already recorded. |

They are in the registry, tested with golden files, and run when a pipeline names them.

## Running passes yourself

Every pass is registered by name, and a pipeline is a comma-separated list of names.

From the command line, `tessiq-opt` reads textual IR and runs a pipeline over it:

```
$ tessiq-opt --list
$ tessiq-opt --passes=mark-supported,layout-assign,cluster,fuse-elementwise graph.tsr
$ tessiq-opt --passes=mark-supported,cluster,fuse-elementwise --emit-cuda graph.tsr
```

`tessiq-opt` is built from the core library alone, with no PyTorch, so it lists the core passes and
not `override-<name>` or `rewrite-<name>`. Those are registered by the Python extension when
tessiq is imported.

From Python, which has every pass:

```python
import tessiq

graph = tessiq.capture(model, x)  # the FX graph a compile receives
ir = tessiq.import_graph(graph)  # the same graph as textual IR
tessiq.pass_names()  # every registered pass
tessiq.run_passes(ir, "rewrite-rms-norm,mark-supported")
tessiq.explain(model, x, sections=("passes",), rewrites=["rms_norm"])
```

`explain`'s passes section runs the compile's pipeline one pass at a time and prints each pass's
changed lines, which is the quickest way to see why a node ended up where it did.

## Adding a pass

There are three ways, depending on what the pass is for.

**A rewrite for a custom operation.** To have the compiler find a computation written out in
PyTorch and replace it with a custom operation, give the operation a pattern: a `.tsr` file in its
directory and `.pattern()` in its `.cpp`. That registers `rewrite-<name>`, which a compile runs
when it names it. See "Rewriting a computation written out in PyTorch" in
[csrc/custom_ops/README.md](../csrc/custom_ops/README.md).

**A replacement for a torch operation.** To have a custom operation stand in for a torch operation
everywhere, add `.replaces("aten::<op>")` to its definition. That registers `override-<name>`,
which every compile runs. See "Replacing a torch operation" in the same README.

Both need a rebuild, and neither needs any change outside the operation's directory.

**A compiler pass.** For anything else, write a pass in C++:

1. Create `csrc/tessiq/passes/<name>.cpp` with a class that derives from `Pass` and implements
   `name()`, `description()` and `run()`. `run()` returns whether it changed the graph. Above the
   class, write what the pass assumes on entry, what it guarantees on exit, and why the rewrite
   keeps the graph computing the same thing:

   ```cpp
   /// Removes nodes whose results nothing reads.
   ///
   /// Precondition: use lists agree with operands, which the verifier checks.
   /// Postcondition: every remaining node has at least one result that is used or returned.
   ///
   /// Legality: removing a node whose result nobody reads cannot change what the graph computes.
   class DeadNodeElimination final : public Pass {
    public:
     [[nodiscard]] std::string_view name() const override { return "dead-node-elimination"; }
     [[nodiscard]] std::string_view description() const override {
       return "remove nodes whose results are never used";
     }
     Result<bool> run(ir::Graph& graph) override;
   };

   TESSIQ_REGISTER_PASS(DeadNodeElimination)
   ```

2. Add the file to `tessiq_ir` in `csrc/CMakeLists.txt`. The library is linked whole, so the
   registration survives even though nothing calls the pass by symbol.
3. Add golden files in `tests/cpp/golden/`, named `<pass>_<what the case shows>.tsr`. Each is a
   `// RUN:` line naming a pipeline, an input graph, `// ---`, and the output exactly as it prints.
   The test runner finds them on its own. Include cases where the pass must leave the graph alone.
4. Rebuild. The pass is now in `tessiq-opt --list` and `tessiq.pass_names()`, and any pipeline can
   name it.
5. To have every compile run it, add it to `kPipeline` in `csrc/tessiq/torch/partition.cpp`, at the
   place its precondition needs, and say why there in the comment.

The core, which includes the passes, builds as C++17 and does not include PyTorch. Debug builds run
the verifier after every pass, so a pass that leaves the graph malformed fails at that pass.
