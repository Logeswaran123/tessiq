# ADR 0003: registering the backend takes over the process's accelerator

- **Status:** accepted
- **Date:** 2026-09-22

## Context

PyTorch has a notion of *the* accelerator: one device type per process, exposed as
`torch.accelerator.current_accelerator()`. Registering a PrivateUse1 backend makes it that
accelerator, and there is no way to register one without doing so.

```python
>>> import torch
>>> torch.accelerator.current_accelerator()
device(type='cuda')
>>> import tessiq
>>> torch.accelerator.current_accelerator()
device(type='tessiq')
```

The consequence is not confined to tessiq's own tensors. The autograd engine asks the accelerator
for the stream a node ran on, and a CUDA tensor is no longer on the accelerator, so a backward
pass over CUDA tensors fails inside the engine:

```
RuntimeError: opt_ready_stream && opt_parent_stream INTERNAL ASSERT FAILED
  at "torch/csrc/autograd/engine.cpp"
```

Importing tessiq is enough to cause it. Using tessiq is not required, and neither is compiling
anything.

## Decision

Accept it, and say so here.

Implementing more of `DeviceGuardImplInterface` does not help. The engine is not failing because
tessiq's guard is incomplete; it is failing because it looks for a CUDA tensor's stream among the
accelerator's devices and tessiq is the accelerator. `getDefaultStream` was implemented and
reverted while establishing that, because nothing reachable called it.

## Consequences

**A process can use tessiq or CUDA autograd, not both.** Forward passes on CUDA are unaffected,
which is why `bench/against_torch.py` can compare against Inductor: it measures inference. A
benchmark that trained on both would have to fork.

**The test suite is unaffected**, because it compares tessiq against eager on the CPU, which is
the stricter reference anyway.

**It cannot be fixed in tessiq.** It follows from PyTorch supporting one accelerator per process
and from PrivateUse1 being how an out-of-tree backend registers. A backend built into PyTorch as
its own device type would not have it.
