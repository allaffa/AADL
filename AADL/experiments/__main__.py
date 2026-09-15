"""Command-line entry point: ``python -m AADL.experiments``."""

import argparse
import json
from pathlib import Path

from .config import ExperimentConfig
from .registry import list_workloads
from .runner import run_experiment


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run a reproducible AADL experiment")
    parser.add_argument("config", nargs="?", help="JSON experiment configuration")
    parser.add_argument("--list-workloads", action="store_true")
    parser.add_argument("--output", help="override the result directory")
    args = parser.parse_args(argv)
    if args.list_workloads:
        print("\n".join(list_workloads()))
        return 0
    if not args.config:
        parser.error("config is required unless --list-workloads is used")
    config = ExperimentConfig.from_dict(json.loads(Path(args.config).read_text()))
    if args.output:
        config.output.directory = args.output
    record, path = run_experiment(config)
    if path is not None:
        print(json.dumps({"status": record["status"], "result": str(path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
