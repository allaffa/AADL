"""Deterministic ill-conditioned regression for mechanism and ablation studies."""

import torch

from ..registry import WorkloadInstance, register_workload
from ._common import Options, batches


class QuadraticWorkload:
    family = "controlled"
    name = "controlled.quadratic"

    def __init__(self, values):
        options = Options(values)
        self.dimension = options.integer("dimension", 32)
        self.samples = options.integer("samples", 128)
        self.batch_size = options.integer("batch_size", 32)
        options.finish()

    def build(self, device, seed):
        generator = torch.Generator(device="cpu").manual_seed(seed)
        scales = torch.logspace(0, -3, self.dimension)
        features = torch.randn(self.samples, self.dimension, generator=generator) * scales
        target_weight = torch.randn(self.dimension, 1, generator=generator)
        targets = features @ target_weight
        features, targets = features.to(device), targets.to(device)
        model = torch.nn.Linear(self.dimension, 1, bias=False).to(device)

        def loss(module, batch):
            x, y = batch
            return torch.nn.functional.mse_loss(module(x), y)

        def evaluate(module):
            module.eval()
            with torch.no_grad():
                value = loss(module, (features, targets))
            return {"loss": float(value)}

        return WorkloadInstance(
            model,
            lambda epoch, rank, world_size: batches(
                features, targets, self.batch_size, seed, epoch, rank, world_size,
            ),
            loss,
            evaluate,
        )


@register_workload(QuadraticWorkload.name)
def create(options):
    return QuadraticWorkload(options)
