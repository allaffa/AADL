# Reproducible paper experiments

This directory contains thin configuration files, not independent training
programs. All families use `AADL.experiments`, which keeps these concerns
orthogonal:

- **workload**: data, model, loss, and scientific metrics;
- **method**: optimizer and optional AADL configuration;
- **execution**: serial or native PyTorch DDP;
- **recording**: timing, environment, epoch metrics, and controller decisions.

Run the small controlled example from the repository root:

```bash
python -m AADL.experiments examples/paper/configs/controlled.json
python -m AADL.experiments --list-workloads
```

Run the same configuration with native DDP by setting
`execution.distributed` to `true` and launching it normally:

```bash
torchrun --standalone --nproc-per-node=4 \
  -m AADL.experiments examples/paper/configs/controlled-ddp.json
```

Each run writes one JSON record on rank zero. The record contains the complete
configuration, seed, source revision, software/hardware metadata, epoch
metrics, wall time, and adaptive-sketch controller decisions. This is the
canonical input for tables and plots; examples must not invent their own log
format.

In DDP runs, ordinary backward passes and gradient synchronization remain
entirely native PyTorch operations. On Anderson safeguard evaluations the
runner all-reduces only the detached loss scalar. This makes acceptance and
adaptive retry counts identical across ranks without triggering another
gradient synchronization or duplicating DDP logic.

The built-in `*.synthetic` workloads are fast protocol checks. They represent
the controlled, vision, graph, and transformer interfaces without downloads.
Publication experiments should register real dataset workloads through
`register_workload`; optional packages such as torchvision, fairchem-core, or
HydraGNN should be imported inside their workload factory so the core AADL
installation remains usable without them.

Kernel-cost studies use
`AADL.experiments.benchmarks.benchmark_anderson_kernel`, keeping numerical
microbenchmarks separate from end-to-end convergence experiments.
