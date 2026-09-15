"""Dependency-free message-passing workload for graph-family smoke studies."""

import torch

from ..registry import WorkloadInstance, register_workload
from ._common import Options, batches, classification_metrics


class GraphClassifier(torch.nn.Module):
    def __init__(self, nodes, width, classes):
        super().__init__()
        adjacency = torch.eye(nodes).roll(1, 0) + torch.eye(nodes).roll(-1, 0)
        adjacency += torch.eye(nodes)
        self.register_buffer("adjacency", adjacency / adjacency.sum(dim=-1, keepdim=True))
        self.input = torch.nn.Linear(3, width)
        self.output = torch.nn.Linear(width, classes)

    def forward(self, features):
        hidden = torch.relu(self.input(features))
        hidden = torch.einsum("ij,bjk->bik", self.adjacency, hidden)
        return self.output(hidden.mean(dim=1))


class SyntheticGraphWorkload:
    family = "graph"
    name = "graph.synthetic"

    def __init__(self, values):
        options = Options(values)
        self.samples = options.integer("samples", 128)
        self.batch_size = options.integer("batch_size", 16)
        self.nodes = options.integer("nodes", 8)
        self.classes = options.integer("classes", 3)
        options.finish()

    def build(self, device, seed):
        generator = torch.Generator(device="cpu").manual_seed(seed)
        features = torch.randn(self.samples, self.nodes, 3, generator=generator)
        targets = (features[..., 0].sum(dim=1) > 0).long().remainder(self.classes)
        features, targets = features.to(device), targets.to(device)
        model = GraphClassifier(self.nodes, 16, self.classes).to(device)
        return WorkloadInstance(
            model,
            lambda epoch, rank, world_size: batches(
                features, targets, self.batch_size, seed, epoch, rank, world_size,
            ),
            lambda module, batch: torch.nn.functional.cross_entropy(module(batch[0]), batch[1]),
            lambda module: classification_metrics(module, features, targets),
        )


@register_workload(SyntheticGraphWorkload.name)
def create(options):
    return SyntheticGraphWorkload(options)
