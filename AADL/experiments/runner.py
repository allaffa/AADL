"""One training protocol for every registered workload family."""

from datetime import datetime, timezone
import random
import time

import numpy as np
import torch

from AADL import accelerate
from .config import ExperimentConfig
from .distributed import maybe_wrap_ddp, prepare_distributed
from .records import new_record, write_record
from .registry import create_workload


def _device(config, context):
    requested = config.execution.device
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    if device.type == "cuda":
        index = context.local_rank if context.enabled else (device.index or 0)
        torch.cuda.set_device(index)
        device = torch.device("cuda", index)
    return device


def _optimizer(config, parameters):
    choices = {
        "sgd": torch.optim.SGD,
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW,
    }
    return choices[config.method.optimizer](
        parameters, lr=config.method.learning_rate,
        **config.method.optimizer_options,
    )


def _scheduler(config, optimizer):
    values = dict(config.method.scheduler)
    name = values.pop("name", None)
    if name is None:
        if values:
            raise ValueError("scheduler options require a scheduler name")
        return None
    choices = {
        "cosine": torch.optim.lr_scheduler.CosineAnnealingLR,
        "step": torch.optim.lr_scheduler.StepLR,
        "multistep": torch.optim.lr_scheduler.MultiStepLR,
    }
    try:
        scheduler = choices[name]
    except KeyError as error:
        raise ValueError(f"unsupported scheduler: {name}") from error
    if name == "cosine":
        values.setdefault("T_max", config.epochs)
    return scheduler(optimizer, **values)


def _controller_snapshot(optimizer, epoch, step):
    diagnostics = getattr(optimizer, "acc_sketch_last_diagnostics", None)
    if diagnostics is None:
        return None
    call = getattr(optimizer, "acc_call_counter", 0)
    if (call <= getattr(optimizer, "acc_wait_iterations", 0)
            or call % getattr(optimizer, "acc_frequency", 1)):
        return None
    return {
        "epoch": epoch,
        "step": step,
        "fraction": getattr(optimizer, "acc_last_sketch_fraction", None),
        "diagnostics": diagnostics,
    }


def run_experiment(config: ExperimentConfig):
    """Run an experiment and return ``(record, result_path_or_none)``."""
    config.validate()
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    context = prepare_distributed(
        config.execution.distributed, config.execution.backend,
    )
    device = _device(config, context)
    workload = create_workload(config.workload, config.workload_options)
    if workload.family != config.family:
        raise ValueError(
            f"workload {config.workload!r} belongs to {workload.family!r}, "
            f"not {config.family!r}"
        )
    instance = workload.build(device, config.seed)
    model = maybe_wrap_ddp(instance.model.to(device), context, device)
    optimizer = _optimizer(config, model.parameters())
    if config.method.acceleration:
        accelerate(optimizer, **config.method.acceleration)
    # Construct after AADL installs its optimizer-step wrapper so PyTorch's
    # scheduler observes the final step method and does not flag it as replaced.
    scheduler = _scheduler(config, optimizer)

    record = new_record(config, context)
    global_step = 0
    started = time.perf_counter()
    try:
        for epoch in range(config.epochs):
            epoch_started = time.perf_counter()
            losses = []
            model.train()
            for batch in instance.train_batches(
                    epoch, context.rank, context.world_size):
                def closure():
                    optimizer.zero_grad()
                    loss = instance.loss(model, batch)
                    if torch.is_grad_enabled():
                        loss.backward()
                        return loss
                    # Safeguard decisions must be identical on every rank.
                    # Reducing only loss scalars avoids a second gradient sync
                    # while ensuring adaptive retries execute in lockstep.
                    if context.enabled:
                        loss = loss.detach().clone()
                        torch.distributed.all_reduce(loss)
                        loss /= context.world_size
                    return loss

                loss = optimizer.step(closure)
                losses.append(float(loss.detach()))
                snapshot = _controller_snapshot(optimizer, epoch, global_step)
                if snapshot is not None and config.output.save_trace:
                    record["controller"].append(snapshot)
                global_step += 1
            if not losses:
                raise ValueError(
                    "workload produced no full batches; reduce batch_size or world_size"
                )
            metrics = instance.evaluate(model)
            record["epochs"].append({
                "epoch": epoch,
                "mean_train_loss": sum(losses) / len(losses),
                "steps": len(losses),
                "seconds": time.perf_counter() - epoch_started,
                "metrics": metrics,
                "learning_rates": [group["lr"] for group in optimizer.param_groups],
            })
            if scheduler is not None:
                scheduler.step()
        record["status"] = "completed"
        record["summary"] = record["epochs"][-1]["metrics"]
    except Exception as error:
        record["status"] = "failed"
        record["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        record["finished_at"] = datetime.now(timezone.utc).isoformat()
        record["total_seconds"] = time.perf_counter() - started
        result_path = (
            write_record(record, config.output.directory)
            if context.is_primary else None
        )
    return record, result_path
