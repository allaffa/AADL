"""Workload contracts and registry.

Registrations are factories, not constructed datasets, so optional scientific
dependencies and large data files are touched only when a workload is used.
"""

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Protocol

import torch


Batch = Any


@dataclass
class WorkloadInstance:
    model: torch.nn.Module
    train_batches: Callable[[int, int, int], Iterable[Batch]]
    loss: Callable[[torch.nn.Module, Batch], torch.Tensor]
    evaluate: Callable[[torch.nn.Module], dict[str, float]]


class Workload(Protocol):
    family: str
    name: str

    def build(self, device: torch.device, seed: int) -> WorkloadInstance: ...


_WORKLOADS: dict[str, Callable[[dict[str, Any]], Workload]] = {}


def register_workload(name: str):
    def decorate(factory):
        if name in _WORKLOADS:
            raise ValueError(f"workload already registered: {name}")
        _WORKLOADS[name] = factory
        return factory
    return decorate


def _load_builtins():
    # Importing this module performs only registrations; datasets remain lazy.
    from . import workloads  # noqa: F401


def create_workload(name: str, options=None) -> Workload:
    _load_builtins()
    try:
        factory = _WORKLOADS[name]
    except KeyError as error:
        available = ", ".join(sorted(_WORKLOADS))
        raise ValueError(f"unknown workload {name!r}; available: {available}") from error
    return factory(dict(options or {}))


def list_workloads() -> tuple[str, ...]:
    _load_builtins()
    return tuple(sorted(_WORKLOADS))
