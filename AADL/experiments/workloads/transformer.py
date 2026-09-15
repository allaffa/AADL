"""Small sequence workload for studying history cost and mixed precision."""

import torch

from ..registry import WorkloadInstance, register_workload
from ._common import Options, batches, classification_metrics


class SequenceClassifier(torch.nn.Module):
    def __init__(self, vocabulary, classes):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocabulary, 16)
        layer = torch.nn.TransformerEncoderLayer(
            16, 4, dim_feedforward=32, dropout=0.0, batch_first=True,
        )
        self.encoder = torch.nn.TransformerEncoder(layer, 1)
        self.head = torch.nn.Linear(16, classes)

    def forward(self, tokens):
        return self.head(self.encoder(self.embedding(tokens)).mean(dim=1))


class SyntheticTransformerWorkload:
    family = "transformer"
    name = "transformer.synthetic"

    def __init__(self, values):
        options = Options(values)
        self.samples = options.integer("samples", 128)
        self.batch_size = options.integer("batch_size", 16)
        self.sequence_length = options.integer("sequence_length", 12)
        self.vocabulary = options.integer("vocabulary", 32)
        self.classes = options.integer("classes", 4)
        options.finish()

    def build(self, device, seed):
        generator = torch.Generator(device="cpu").manual_seed(seed)
        features = torch.randint(
            self.vocabulary, (self.samples, self.sequence_length), generator=generator,
        )
        targets = features.sum(dim=1).remainder(self.classes)
        features, targets = features.to(device), targets.to(device)
        model = SequenceClassifier(self.vocabulary, self.classes).to(device)
        return WorkloadInstance(
            model,
            lambda epoch, rank, world_size: batches(
                features, targets, self.batch_size, seed, epoch, rank, world_size,
            ),
            lambda module, batch: torch.nn.functional.cross_entropy(module(batch[0]), batch[1]),
            lambda module: classification_metrics(module, features, targets),
        )


@register_workload(SyntheticTransformerWorkload.name)
def create(options):
    return SyntheticTransformerWorkload(options)
