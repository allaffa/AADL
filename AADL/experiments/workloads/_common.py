from dataclasses import dataclass
import math

import torch


def batches(features, targets, batch_size, seed, epoch, rank=0, world_size=1):
    generator = torch.Generator(device="cpu").manual_seed(seed + epoch)
    order = torch.randperm(features.size(0), generator=generator)
    # Equivalent to a drop-last DistributedSampler: every rank executes the
    # same number of backward passes, which DDP collectives require.
    usable = (order.numel() // world_size) * world_size
    order = order[:usable][rank::world_size]
    return [
        (features[index].to(features.device), targets[index].to(targets.device))
        for index in order.split(batch_size) if index.numel() == batch_size
    ]


def classification_metrics(model, features, targets):
    model.eval()
    with torch.no_grad():
        logits = model(features)
        loss = torch.nn.functional.cross_entropy(logits, targets)
        accuracy = (logits.argmax(dim=-1) == targets).float().mean()
    return {"loss": float(loss), "accuracy": float(accuracy)}


@dataclass
class Options:
    values: dict

    def integer(self, name, default, minimum=1):
        value = self.values.pop(name, default)
        if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
        return value

    def number(self, name, default, minimum=None):
        value = self.values.pop(name, default)
        if (not isinstance(value, (int, float)) or isinstance(value, bool)
                or not math.isfinite(value)
                or (minimum is not None and value < minimum)):
            raise ValueError(f"{name} must be a finite number >= {minimum}")
        return value

    def string(self, name, default=None, required=False):
        value = self.values.pop(name, default)
        if required and not value:
            raise ValueError(f"{name} is required")
        if value is not None and not isinstance(value, str):
            raise ValueError(f"{name} must be a string")
        return value

    def boolean(self, name, default=False):
        value = self.values.pop(name, default)
        if not isinstance(value, bool):
            raise ValueError(f"{name} must be a boolean")
        return value

    def finish(self):
        if self.values:
            names = ", ".join(sorted(self.values))
            raise ValueError(f"unknown workload options: {names}")


def partition_indices(labels, rank, world_size, seed, epoch, policy="iid", alpha=0.5):
    """Return an equal-length rank shard under IID or label-Dirichlet policy."""
    labels = torch.as_tensor(labels, device="cpu").view(-1)
    generator = torch.Generator(device="cpu").manual_seed(seed + epoch)
    if policy == "iid":
        order = torch.randperm(labels.numel(), generator=generator)
        usable = (order.numel() // world_size) * world_size
        return order[:usable][rank::world_size]
    if policy != "dirichlet":
        raise ValueError("partition must be 'iid' or 'dirichlet'")
    if alpha <= 0:
        raise ValueError("dirichlet_alpha must be positive")
    # Seed NumPy locally because torch.distributions does not accept a generator.
    import numpy as np
    rng = np.random.default_rng(seed)
    shards = [[] for _ in range(world_size)]
    for label in labels.unique(sorted=True):
        indices = torch.where(labels == label)[0].numpy()
        rng.shuffle(indices)
        probabilities = rng.dirichlet(np.full(world_size, alpha))
        cuts = (np.cumsum(probabilities)[:-1] * len(indices)).astype(int)
        for target, part in zip(shards, np.split(indices, cuts)):
            target.extend(part.tolist())
    equal = min(len(shard) for shard in shards)
    # Equal sample counts prevent mismatched DDP collective counts.
    selected = torch.tensor(shards[rank][:equal], dtype=torch.long)
    if selected.numel():
        selected = selected[torch.randperm(selected.numel(), generator=generator)]
    return selected
