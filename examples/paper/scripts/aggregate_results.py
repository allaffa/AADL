#!/usr/bin/env python3
"""Create statistical tables and figures from one or more result trees."""

import argparse
import json
from pathlib import Path

from AADL.experiments.aggregate import (
    load_records,
    summarize,
    write_csv,
    write_latex,
    write_markdown,
    write_plots,
)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", help="JSON files or result directories")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--group-by", action="append",
        default=["config.family", "config.workload", "config.method.name"],
    )
    parser.add_argument(
        "--metric", action="append", default=[], metavar="NAME=PATH",
        help="metric name and dotted result path; repeat for multiple metrics",
    )
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)
    if not 0 < args.confidence < 1:
        parser.error("--confidence must be in (0, 1)")
    if args.bootstrap_samples < 1:
        parser.error("--bootstrap-samples must be positive")
    metrics = {"wall_time": "total_seconds"}
    for expression in args.metric:
        try:
            name, path = expression.split("=", 1)
        except ValueError:
            parser.error(f"metric must have NAME=PATH form: {expression}")
        metrics[name] = path

    records = load_records(args.inputs)
    if not records:
        parser.error("no completed AADL result records found")
    rows = summarize(
        records, args.group_by, metrics, args.confidence,
        args.bootstrap_samples, args.seed,
    )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    write_csv(rows, output / "summary.csv")
    write_markdown(rows, output / "summary.md")
    write_latex(rows, output / "summary.tex")
    (output / "aggregation.json").write_text(json.dumps({
        "inputs": args.inputs,
        "records": len(records),
        "group_by": args.group_by,
        "metrics": metrics,
        "confidence": args.confidence,
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.seed,
    }, indent=2, sort_keys=True) + "\n")
    if not args.no_plots:
        write_plots(rows, args.group_by, metrics, output)
    print(f"aggregated {len(records)} records into {output}")


if __name__ == "__main__":
    main()
