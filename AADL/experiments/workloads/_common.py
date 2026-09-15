from dataclasses import dataclass

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

    def integer(self, name, default):
        value = self.values.pop(name, default)
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
        return value

    def finish(self):
        if self.values:
            names = ", ".join(sorted(self.values))
            raise ValueError(f"unknown workload options: {names}")
