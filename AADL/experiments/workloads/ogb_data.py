"""OGB graph-property workloads through the official PyG adapter."""

from pathlib import Path

import torch

from ..registry import WorkloadInstance, register_workload
from ._common import Options, partition_indices


class OGBGraphModel(torch.nn.Module):
    def __init__(self, tasks, width, layers, atom_encoder, convolution):
        super().__init__()
        self.input = atom_encoder(width)
        self.layers = torch.nn.ModuleList(
            convolution(torch.nn.Sequential(
                torch.nn.Linear(width, width * 2), torch.nn.ReLU(),
                torch.nn.Linear(width * 2, width),
            )) for _ in range(layers)
        )
        self.output = torch.nn.Linear(width, tasks)

    def forward(self, batch):
        from torch_geometric.nn import global_mean_pool
        hidden = self.input(batch.x)
        for layer in self.layers:
            hidden = torch.relu(layer(hidden, batch.edge_index))
        return self.output(global_mean_pool(hidden, batch.batch))


class OGBWorkload:
    family = "graph"
    name = "graph.ogbg-molhiv"

    def __init__(self, values):
        options = Options(values)
        self.root = Path(options.string("root", required=True)).expanduser()
        self.batch_size = options.integer("batch_size", 64)
        self.workers = options.integer("workers", 0, minimum=0)
        self.width = options.integer("width", 128)
        self.layers = options.integer("layers", 3)
        self.partition = options.string("partition", "iid")
        self.alpha = options.number("dirichlet_alpha", 0.5, minimum=0.0)
        options.finish()

    def build(self, device, seed):
        try:
            from ogb.graphproppred import Evaluator, PygGraphPropPredDataset
            from ogb.graphproppred.mol_encoder import AtomEncoder
            from torch_geometric.loader import DataLoader
            from torch_geometric.nn import GINConv
        except ImportError as error:
            raise ImportError(
                "OGB workloads require ogb and torch-geometric; "
                "install requirements-paper.txt"
            ) from error
        dataset = PygGraphPropPredDataset(name="ogbg-molhiv", root=str(self.root))
        split = dataset.get_idx_split()
        train_indices = torch.as_tensor(split["train"])
        train_labels = dataset.data.y[train_indices].view(-1).nan_to_num(-1)
        validation = dataset[split["valid"]]
        model = OGBGraphModel(
            dataset.num_tasks, self.width, self.layers, AtomEncoder, GINConv,
        ).to(device)
        evaluator = Evaluator("ogbg-molhiv")

        def train_batches(epoch, rank, world_size):
            local = partition_indices(
                train_labels, rank, world_size, seed, epoch,
                self.partition, self.alpha,
            )
            subset = dataset[train_indices[local]]
            yield from DataLoader(
                subset, self.batch_size, shuffle=False, drop_last=True,
                num_workers=self.workers,
            )

        def loss(module, batch):
            batch = batch.to(device)
            prediction = module(batch)
            target = batch.y.view_as(prediction).float()
            mask = torch.isfinite(target)
            return torch.nn.functional.binary_cross_entropy_with_logits(
                prediction[mask], target[mask],
            )

        def evaluate(module):
            predictions, targets = [], []
            module.eval()
            with torch.no_grad():
                for batch in DataLoader(validation, self.batch_size, num_workers=self.workers):
                    batch = batch.to(device)
                    predictions.append(module(batch).cpu())
                    targets.append(batch.y.view(-1, dataset.num_tasks).cpu())
            result = evaluator.eval({
                "y_true": torch.cat(targets).numpy(),
                "y_pred": torch.cat(predictions).numpy(),
            })
            return {name: float(value) for name, value in result.items()}

        return WorkloadInstance(model, train_batches, loss, evaluate)


@register_workload(OGBWorkload.name)
def create(options):
    return OGBWorkload(options)
