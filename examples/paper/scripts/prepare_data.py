#!/usr/bin/env python3
"""Download public datasets or validate manually provisioned ImageNet data."""

import argparse
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=("cifar10", "cifar100", "imagenet", "wikitext103", "glue-sst2", "ogbg-molhiv"))
    parser.add_argument("root")
    args = parser.parse_args(argv)
    root = Path(args.root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)

    if args.dataset.startswith("cifar"):
        try:
            from torchvision.datasets import CIFAR10, CIFAR100
        except ImportError as error:
            raise SystemExit("install requirements-paper.txt first") from error
        dataset = CIFAR10 if args.dataset == "cifar10" else CIFAR100
        dataset(root, train=True, download=True)
        dataset(root, train=False, download=True)
    elif args.dataset == "imagenet":
        for split in ("train", "val"):
            directory = root / split
            if not directory.is_dir() or not any(directory.iterdir()):
                raise SystemExit(f"missing populated ImageNet directory: {directory}")
    elif args.dataset in {"wikitext103", "glue-sst2"}:
        try:
            from datasets import load_dataset
        except ImportError as error:
            raise SystemExit("install requirements-paper.txt first") from error
        if args.dataset == "wikitext103":
            load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", cache_dir=str(root))
        else:
            load_dataset("nyu-mll/glue", "sst2", cache_dir=str(root))
    else:
        try:
            from ogb.graphproppred import PygGraphPropPredDataset
        except ImportError as error:
            raise SystemExit("install requirements-paper.txt first") from error
        PygGraphPropPredDataset(name="ogbg-molhiv", root=str(root))
    print(f"{args.dataset} is ready under {root}")


if __name__ == "__main__":
    main()
