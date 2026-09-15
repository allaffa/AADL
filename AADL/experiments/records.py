"""Paper-ready, machine-readable result records."""

from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import uuid

import torch


SCHEMA_VERSION = 1


def environment_metadata():
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, check=True,
            text=True, timeout=2,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        commit = None
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "platform": platform.platform(),
        "git_commit": commit,
        "cuda_available": torch.cuda.is_available(),
        "accelerators": [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ],
    }


def new_record(config, context):
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": str(uuid.uuid4()),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "config": config.to_dict(),
        "environment": environment_metadata(),
        "distributed": {
            "enabled": context.enabled,
            "rank": context.rank,
            "world_size": context.world_size,
        },
        "epochs": [],
        "controller": [],
        "status": "running",
    }


def write_record(record, directory):
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    target = path / f"{record['run_id']}.json"
    temporary = target.with_suffix(f".json.tmp-{os.getpid()}")
    def json_safe(value):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        if isinstance(value, dict):
            return {key: json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [json_safe(item) for item in value]
        return value

    temporary.write_text(
        json.dumps(json_safe(record), indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    temporary.replace(target)
    return target
