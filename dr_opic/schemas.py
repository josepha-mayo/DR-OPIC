from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class Task:
    """A bounded coding task with optional verifier metadata."""

    task_id: str
    prompt: str
    domain: str = "python"
    entrypoint: str | None = None
    tests: str | None = None
    split: str = "train"
    verifier_reliability: float = 1.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.task_id.strip():
            raise ValueError("task_id is required")
        if not self.prompt.strip():
            raise ValueError("prompt is required")
        if not 0.0 <= self.verifier_reliability <= 1.0:
            raise ValueError("verifier_reliability must be between 0 and 1")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Candidate:
    """A generated or repaired answer plus verifier observations."""

    task_id: str
    code: str
    source: str = "student"
    passed: bool = False
    reward: float = 0.0
    observation: str = ""
    tokens: int = 0
    latency_s: float = 0.0
    logprob_per_token: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.task_id.strip():
            raise ValueError("task_id is required")
        if self.source not in {"student", "repair", "teacher", "postprocess", "replay"}:
            raise ValueError(f"unknown candidate source: {self.source}")
        if self.tokens < 0:
            raise ValueError("tokens must be non-negative")
        if self.latency_s < 0:
            raise ValueError("latency_s must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RolloutGroup:
    """K attempts for one task."""

    task: Task
    candidates: tuple[Candidate, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task.to_dict(),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "passes": self.passes,
            "samples": self.samples,
        }

    @property
    def passes(self) -> int:
        return sum(1 for c in self.candidates if c.passed)

    @property
    def samples(self) -> int:
        return len(self.candidates)
