"""DR-OPIC: verifier-grounded SLM forge utilities."""

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

__all__ = [
    "Candidate",
    "PythonTask",
    "RouteDecision",
    "RewardWeights",
    "RolloutGroup",
    "SafetyDecision",
    "Task",
    "VerificationResult",
    "classify_coding_safety",
    "group_relative_advantages",
    "route_task",
    "selective_accept",
    "smoothed_pass_rate",
    "verify_python",
    "zpd_weight",
]
