"""Serializable configuration shared by all experiment families."""

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping


@dataclass
class MethodConfig:
    name: str = "sgd"
    optimizer: str = "sgd"
    learning_rate: float = 1e-2
    optimizer_options: dict[str, Any] = field(default_factory=dict)
    scheduler: dict[str, Any] = field(default_factory=dict)
    acceleration: dict[str, Any] = field(default_factory=dict)

    def validate(self):
        if self.optimizer not in {"sgd", "adam", "adamw"}:
            raise ValueError(f"unsupported optimizer: {self.optimizer}")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")


@dataclass
class ExecutionConfig:
    device: str = "auto"
    distributed: bool = False
    backend: str | None = None


@dataclass
class OutputConfig:
    directory: str = "results"
    save_trace: bool = True


@dataclass
class ExperimentConfig:
    family: str
    workload: str
    seed: int = 0
    epochs: int = 1
    workload_options: dict[str, Any] = field(default_factory=dict)
    method: MethodConfig = field(default_factory=MethodConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    tags: dict[str, str] = field(default_factory=dict)

    def validate(self):
        if not self.family or not self.workload:
            raise ValueError("family and workload must be non-empty")
        if self.epochs < 1:
            raise ValueError("epochs must be at least one")
        self.method.validate()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]):
        values = dict(data)
        values["method"] = MethodConfig(**values.get("method", {}))
        values["execution"] = ExecutionConfig(**values.get("execution", {}))
        values["output"] = OutputConfig(**values.get("output", {}))
        config = cls(**values)
        config.validate()
        return config
