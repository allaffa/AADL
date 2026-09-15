"""Download-free vision reference workload; real datasets can use the same contract."""

import torch

from ..registry import WorkloadInstance, register_workload
from ._common import Options, batches, classification_metrics


class SyntheticVisionWorkload:
    family = "vision"
    name = "vision.synthetic"

    def __init__(self, values):
        options = Options(values)
        self.samples = options.integer("samples", 128)
        self.batch_size = options.integer("batch_size", 32)
        self.classes = options.integer("classes", 4)
        options.finish()

    def build(self, device, seed):
        generator = torch.Generator(device="cpu").manual_seed(seed)
        features = torch.randn(self.samples, 1, 8, 8, generator=generator)
        projection = torch.randn(64, self.classes, generator=generator)
        targets = (features.flatten(1) @ projection).argmax(dim=-1)
        features, targets = features.to(device), targets.to(device)
        model = torch.nn.Sequential(
            torch.nn.Conv2d(1, 8, 3, padding=1), torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d((2, 2)), torch.nn.Flatten(),
            torch.nn.Linear(32, self.classes),
        ).to(device)

        return WorkloadInstance(
            model,
            lambda epoch, rank, world_size: batches(
                features, targets, self.batch_size, seed, epoch, rank, world_size,
            ),
            lambda module, batch: torch.nn.functional.cross_entropy(module(batch[0]), batch[1]),
            lambda module: classification_metrics(module, features, targets),
        )


@register_workload(SyntheticVisionWorkload.name)
def create(options):
    return SyntheticVisionWorkload(options)
