"""Aggregate AADL JSON records into confidence intervals and paper artifacts."""

import csv
import json
import math
from pathlib import Path

import numpy as np


def dotted_value(record, path):
    value = record
    for part in path.split("."):
        value = value[part]
    return value


def load_records(paths):
    records = []
    for source in paths:
        path = Path(source)
        candidates = path.rglob("*.json") if path.is_dir() else (path,)
        for candidate in candidates:
            try:
                record = json.loads(candidate.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            if (record.get("schema_version") is not None
                    and record.get("status") == "completed"
                    and "config" in record):
                record["_source"] = str(candidate)
                records.append(record)
    return records


def bootstrap_interval(values, confidence=0.95, samples=10_000, seed=0):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        raise ValueError("cannot summarize an empty sample")
    mean = float(values.mean())
    if values.size == 1:
        return mean, mean
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=float)
    # Bound temporary storage for experiment matrices with many repetitions.
    chunk = 1_000
    for start in range(0, samples, chunk):
        count = min(chunk, samples - start)
        draws = rng.choice(values, size=(count, values.size), replace=True)
        means[start:start + count] = draws.mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(means, [tail, 1.0 - tail])
    return float(low), float(high)


def summarize(records, group_by, metrics, confidence=0.95, samples=10_000, seed=0):
    groups = {}
    for record in records:
        try:
            key = tuple(str(dotted_value(record, path)) for path in group_by)
        except (KeyError, TypeError):
            continue
        groups.setdefault(key, []).append(record)

    rows = []
    for group_index, (key, group) in enumerate(sorted(groups.items())):
        row = dict(zip(group_by, key))
        row["runs"] = len(group)
        for metric_name, metric_path in metrics.items():
            values = []
            for record in group:
                try:
                    value = float(dotted_value(record, metric_path))
                except (KeyError, TypeError, ValueError):
                    continue
                if math.isfinite(value):
                    values.append(value)
            if not values:
                continue
            low, high = bootstrap_interval(
                values, confidence, samples, seed + group_index,
            )
            row[f"{metric_name}_n"] = len(values)
            row[f"{metric_name}_mean"] = float(np.mean(values))
            row[f"{metric_name}_std"] = (
                float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            )
            row[f"{metric_name}_ci_low"] = low
            row[f"{metric_name}_ci_high"] = high
        rows.append(row)
    return rows


def _columns(rows):
    return list(dict.fromkeys(key for row in rows for key in row))


def write_csv(rows, path):
    columns = _columns(rows)
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows, path):
    columns = _columns(rows)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_format(row.get(key, "")) for key in columns) + " |")
    Path(path).write_text("\n".join(lines) + "\n")


def _format(value):
    return f"{value:.6g}" if isinstance(value, float) else str(value)


def write_latex(rows, path):
    columns = _columns(rows)
    escaped = lambda value: _format(value).replace("_", r"\_")
    lines = [
        r"\begin{tabular}{" + "l" * len(columns) + "}",
        r"\toprule",
        " & ".join(escaped(value) for value in columns) + r" \\",
        r"\midrule",
    ]
    lines.extend(
        " & ".join(escaped(row.get(key, "")) for key in columns) + r" \\"
        for row in rows
    )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    Path(path).write_text("\n".join(lines) + "\n")


def write_plots(rows, group_by, metrics, directory):
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise ImportError(
            "plot generation requires matplotlib; install requirements-paper.txt"
        ) from error
    labels = [" / ".join(str(row[name]) for name in group_by) for row in rows]
    for metric in metrics:
        selected = [(label, row) for label, row in zip(labels, rows)
                    if f"{metric}_mean" in row]
        if not selected:
            continue
        figure_width = max(6.0, 0.65 * len(selected))
        figure, axis = plt.subplots(figsize=(figure_width, 4.0), constrained_layout=True)
        x = np.arange(len(selected))
        means = np.array([row[f"{metric}_mean"] for _, row in selected])
        lower = means - np.array([row[f"{metric}_ci_low"] for _, row in selected])
        upper = np.array([row[f"{metric}_ci_high"] for _, row in selected]) - means
        axis.errorbar(x, means, yerr=np.vstack([lower, upper]), fmt="o", capsize=4)
        axis.set_xticks(x, [label for label, _ in selected], rotation=35, ha="right")
        axis.set_ylabel(metric)
        axis.grid(axis="y", alpha=0.25)
        for suffix in ("pdf", "png"):
            figure.savefig(Path(directory) / f"{metric}.{suffix}", dpi=300)
        plt.close(figure)
