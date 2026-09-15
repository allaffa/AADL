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

The built-in `*.synthetic` workloads are fast protocol checks. Real-data
adapters are provided for `vision.cifar10`, `vision.cifar100`,
`vision.imagenet`, `transformer.wikitext103`, `transformer.glue-sst2`, and
`graph.ogbg-molhiv`. Optional packages are imported only when their workload is
selected, so the core AADL installation remains lightweight.

Kernel-cost studies use
`AADL.experiments.benchmarks.benchmark_anderson_kernel`, keeping numerical
microbenchmarks separate from end-to-end convergence experiments.

## Preparing real data

Install the optional Python packages after the machine-specific PyTorch stack:

```bash
python -m pip install -r requirements-paper.txt
```

Download public datasets in a serial allocation before launching DDP. This
avoids concurrent extraction into the same cache:

```bash
python examples/paper/scripts/prepare_data.py cifar10 data/cifar10
python examples/paper/scripts/prepare_data.py cifar100 data/cifar100
python examples/paper/scripts/prepare_data.py wikitext103 data/huggingface
python examples/paper/scripts/prepare_data.py glue-sst2 data/huggingface
python examples/paper/scripts/prepare_data.py ogbg-molhiv data/ogb
```

ImageNet is not downloaded. Arrange it as `ROOT/train/<class>/...` and
`ROOT/val/<class>/...`, then validate it with:

```bash
python examples/paper/scripts/prepare_data.py imagenet /path/to/imagenet
```

Dataset roots can be overridden without editing committed configurations:

```bash
examples/paper/scripts/run_serial.sh examples/paper/configs/cifar10.json \
  --set workload_options.root=/datasets/cifar10
examples/paper/scripts/run_ddp.sh 4 examples/paper/configs/cifar10-noniid-ddp.json \
  --set workload_options.root=/datasets/cifar10
```

`partition="iid"` deterministically shards shuffled samples. For
classification datasets, `partition="dirichlet"` uses labels and
`dirichlet_alpha` to create heterogeneous rank-local data while retaining an
equal number of samples per rank so DDP executes matching collective counts.

Use `run_comparison.sh` to generate plain optimizer, full Anderson, and
adaptive-sketch records from the same seed and workload configuration. Every
configuration includes a conventional learning-rate scheduler and records the
actual learning rate each epoch.

## Materials workflows

HydraGNN and fairchem/UMA are intentionally delegated to their native data and
training tools because they own graph construction, force/energy losses, and
HPC communication. After installing them with the machine-specific scripts:

```bash
examples/paper/scripts/run_hydragnn.sh /path/to/hydragnn-config.json
examples/paper/scripts/prepare_uma_data.sh TRAIN_DIR VALID_DIR OUTPUT_DIR omat e
examples/paper/scripts/run_fairchem.sh /path/to/uma-finetune.yaml runner.device=cuda
```

These two scripts prepare native baselines; they do **not** silently claim to
apply AADL. A publishable Anderson comparison requires wrapping the optimizer
inside HydraGNN/fairchem's training construction point, which should be added
as a dedicated integration after fixing the exact upstream versions and model
configurations used by the study.

For every dataset, record its upstream revision/version, checksum or snapshot,
license, split, preprocessing, and any sample cap in the manuscript artifact.
