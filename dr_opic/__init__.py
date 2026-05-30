"""DR-OPIC: verifier-grounded SLM forge utilities."""

from .artifacts import dumps_json, write_json, write_jsonl
from .maths import (
    RewardWeights,
    group_relative_advantages,
    smoothed_pass_rate,
    zpd_weight,
)
from .schemas import Candidate, RolloutGroup, Task
from .verifier import PythonTask, VerificationResult, verify_python
from .routing import RouteDecision, route_task
from .safety import SafetyDecision, classify_coding_safety, selective_accept
from .scheduler import ScheduledTask, SchedulerConfig, schedule_group, schedule_round, training_mix

__all__ = [
    "Candidate",
    "PythonTask",
    "RouteDecision",
    "RewardWeights",
    "RolloutGroup",
    "SafetyDecision",
    "ScheduledTask",
    "SchedulerConfig",
    "Task",
    "VerificationResult",
    "classify_coding_safety",
    "dumps_json",
    "group_relative_advantages",
    "route_task",
    "schedule_group",
    "schedule_round",
    "selective_accept",
    "smoothed_pass_rate",
    "verify_python",
    "write_json",
    "write_jsonl",
    "training_mix",
    "zpd_weight",
]
