"""Reusable experiment infrastructure for AADL studies.

The public API deliberately separates workloads, optimization methods, and
execution environments so one scientific example can be run unchanged in
serial or under DDP.
"""

from .config import ExecutionConfig, ExperimentConfig, MethodConfig, OutputConfig
from .registry import Workload, WorkloadInstance, create_workload, list_workloads
from .runner import run_experiment

__all__ = [
    "ExperimentConfig",
    "ExecutionConfig",
    "MethodConfig",
    "OutputConfig",
    "Workload",
    "WorkloadInstance",
    "create_workload",
    "list_workloads",
    "run_experiment",
]
