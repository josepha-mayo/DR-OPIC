from __future__ import annotations

from dataclasses import dataclass, field
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


@dataclass(frozen=True)
class RolloutGroup:
    """K attempts for one task."""

    task: Task
    candidates: tuple[Candidate, ...]

    @property
    def passes(self) -> int:
        return sum(1 for c in self.candidates if c.passed)

    @property
    def samples(self) -> int:
        return len(self.candidates)
