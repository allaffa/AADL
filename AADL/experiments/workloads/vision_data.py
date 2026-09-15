"""Real torchvision datasets, loaded only when explicitly selected."""

from pathlib import Path

import torch

from ..registry import WorkloadInstance, register_workload
from ._common import Options, partition_indices


def _torchvision():
    try:
        import torchvision
        from torchvision import transforms
    except ImportError as error:
        raise ImportError(
            "vision datasets require torchvision; install requirements-paper.txt"
        ) from error
    return torchvision, transforms


def _targets(dataset):
    values = getattr(dataset, "targets", None)
    if values is not None:
        return torch.as_tensor(values)
    samples = getattr(dataset, "samples", None)
    if samples is not None:
        return torch.tensor([label for _, label in samples])
    raise ValueError("dataset does not expose classification targets")


class VisionDatasetWorkload:
    family = "vision"

    def __init__(self, values, dataset_name):
        options = Options(values)
        self.dataset_name = dataset_name
        self.name = f"vision.{dataset_name}"
        self.root = Path(options.string("root", required=True)).expanduser()
        self.batch_size = options.integer("batch_size", 128)
        self.workers = options.integer("workers", 0, minimum=0)
        self.download = options.boolean("download", False)
        self.partition = options.string("partition", "iid")
        self.alpha = options.number("dirichlet_alpha", 0.5, minimum=0.0)
        self.max_eval_batches = options.integer("max_eval_batches", 100)
        options.finish()

    def build(self, device, seed):
        torchvision, transforms = _torchvision()
        if self.dataset_name == "imagenet":
            if self.download:
                raise ValueError("ImageNet cannot be downloaded by torchvision")
            train_transform = transforms.Compose([
                transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
            eval_transform = transforms.Compose([
                transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
            train = torchvision.datasets.ImageFolder(self.root / "train", train_transform)
            validation = torchvision.datasets.ImageFolder(self.root / "val", eval_transform)
            classes = len(train.classes)
            model = torchvision.models.resnet50(num_classes=classes)
        else:
            dataset_class = {
                "cifar10": torchvision.datasets.CIFAR10,
                "cifar100": torchvision.datasets.CIFAR100,
            }[self.dataset_name]
            normalize = transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616),
            )
            train = dataset_class(
                self.root, train=True, download=self.download,
                transform=transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize,
                ]),
            )
            validation = dataset_class(
                self.root, train=False, download=self.download,
                transform=transforms.Compose([transforms.ToTensor(), normalize]),
            )
            classes = 10 if self.dataset_name == "cifar10" else 100
            model = torchvision.models.resnet18(num_classes=classes)
            model.conv1 = torch.nn.Conv2d(3, 64, 3, 1, 1, bias=False)
            model.maxpool = torch.nn.Identity()
        labels = _targets(train)
        model.to(device)

        def train_batches(epoch, rank, world_size):
            indices = partition_indices(
                labels, rank, world_size, seed, epoch, self.partition, self.alpha,
            )
            subset = torch.utils.data.Subset(train, indices.tolist())
            loader = torch.utils.data.DataLoader(
                subset, self.batch_size, shuffle=False, num_workers=self.workers,
                drop_last=True, pin_memory=device.type == "cuda",
            )
            for features, targets in loader:
                yield features.to(device, non_blocking=True), targets.to(device, non_blocking=True)

        def loss(module, batch):
            return torch.nn.functional.cross_entropy(module(batch[0]), batch[1])

        def evaluate(module):
            loader = torch.utils.data.DataLoader(
                validation, self.batch_size, num_workers=self.workers,
                pin_memory=device.type == "cuda",
            )
            total_loss = total_correct = total = 0
            module.eval()
            with torch.no_grad():
                for batch_index, (features, targets) in enumerate(loader):
                    if batch_index >= self.max_eval_batches:
                        break
                    features, targets = features.to(device), targets.to(device)
                    logits = module(features)
                    total_loss += float(torch.nn.functional.cross_entropy(
                        logits, targets, reduction="sum",
                    ))
                    total_correct += int((logits.argmax(-1) == targets).sum())
                    total += targets.numel()
            return {"loss": total_loss / total, "accuracy": total_correct / total}

        return WorkloadInstance(model, train_batches, loss, evaluate)


for _name in ("cifar10", "cifar100", "imagenet"):
    register_workload(f"vision.{_name}")(
        lambda options, dataset_name=_name: VisionDatasetWorkload(options, dataset_name)
    )
