"""Command-line entry point: ``python -m AADL.experiments``."""

import argparse
import json
from pathlib import Path

from .config import ExperimentConfig
from .registry import list_workloads
from .runner import run_experiment


def _override(data, expression):
    try:
        path, raw_value = expression.split("=", 1)
    except ValueError as error:
        raise ValueError(f"override must have PATH=JSON_VALUE form: {expression}") from error
    target = data
    parts = path.split(".")
    for part in parts[:-1]:
        target = target.setdefault(part, {})
        if not isinstance(target, dict):
            raise ValueError(f"cannot descend into non-object override path: {path}")
    try:
        value = json.loads(raw_value)
    except json.JSONDecodeError:
        value = raw_value
    target[parts[-1]] = value


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run a reproducible AADL experiment")
    parser.add_argument("config", nargs="?", help="JSON experiment configuration")
    parser.add_argument("--list-workloads", action="store_true")
    parser.add_argument("--output", help="override the result directory")
    parser.add_argument(
        "--set", action="append", default=[], metavar="PATH=VALUE",
        help="override a dotted configuration value; VALUE may be JSON",
    )
    args = parser.parse_args(argv)
    if args.list_workloads:
        print("\n".join(list_workloads()))
        return 0
    if not args.config:
        parser.error("config is required unless --list-workloads is used")
    data = json.loads(Path(args.config).read_text())
    for expression in args.set:
        _override(data, expression)
    config = ExperimentConfig.from_dict(data)
    if args.output:
        config.output.directory = args.output
    record, path = run_experiment(config)
    if path is not None:
        print(json.dumps({"status": record["status"], "result": str(path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
