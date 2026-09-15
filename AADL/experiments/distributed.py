"""Small adapter around PyTorch's distributed runtime; no DDP reimplementation."""

from dataclasses import dataclass
import os

import torch


@dataclass(frozen=True)
class DistributedContext:
    enabled: bool
    rank: int = 0
    world_size: int = 1
    local_rank: int = 0

    @property
    def is_primary(self):
        return self.rank == 0


def prepare_distributed(requested: bool, backend=None) -> DistributedContext:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if not requested:
        if world_size > 1:
            raise RuntimeError("WORLD_SIZE > 1 but execution.distributed is false")
        return DistributedContext(False)
    if not torch.distributed.is_available():
        raise RuntimeError("this PyTorch build has no distributed support")
    if not torch.distributed.is_initialized():
        if world_size <= 1:
            raise RuntimeError("distributed execution must be launched with torchrun or mpiexec")
        default_backend = "nccl" if torch.cuda.is_available() else "gloo"
        torch.distributed.init_process_group(backend=backend or default_backend)
    return DistributedContext(
        True,
        torch.distributed.get_rank(),
        torch.distributed.get_world_size(),
        int(os.environ.get("LOCAL_RANK", "0")),
    )


def maybe_wrap_ddp(model, context, device):
    if not context.enabled:
        return model
    if device.type == "cuda":
        return torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[context.local_rank], output_device=context.local_rank,
        )
    return torch.nn.parallel.DistributedDataParallel(model)
