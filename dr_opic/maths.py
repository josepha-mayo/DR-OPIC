from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from statistics import mean, pstdev
from typing import Iterable, Sequence


def smoothed_pass_rate(passes: int, samples: int, alpha: float = 0.5) -> float:
    """Jeffreys-smoothed posterior pass estimate."""

    if samples < 0 or passes < 0 or passes > samples:
        raise ValueError("expected 0 <= passes <= samples")
    return (passes + alpha) / (samples + 2.0 * alpha)


def zpd_weight(passes: int, samples: int, reliability: float = 1.0) -> float:
    """Zone-of-proximal-development weight: 4 p (1-p)."""

    p = smoothed_pass_rate(passes, samples)
    return clamp01(4.0 * p * (1.0 - p) * reliability)


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


@dataclass(frozen=True)
class RewardWeights:
    final_pass: float = 1.0
    public_fraction: float = 0.25
    syntax_ok: float = 0.10
    import_ok: float = 0.05
    repeated_token_penalty: float = 0.05
    invalid_format_penalty: float = 0.05
    length_penalty: float = 0.02
    unsafe_api_penalty: float = 0.05


def verifier_reward(
    *,
    final_pass: bool,
    public_fraction: float = 0.0,
    syntax_ok: bool = False,
    import_ok: bool = False,
    repeated_token: bool = False,
    invalid_format: bool = False,
    normalized_length: float = 0.0,
    unsafe_api: bool = False,
    weights: RewardWeights = RewardWeights(),
) -> float:
    """Scalar reward used for ranking and group-relative advantages."""

    score = 0.0
    score += weights.final_pass * float(final_pass)
    score += weights.public_fraction * clamp01(public_fraction)
    score += weights.syntax_ok * float(syntax_ok)
    score += weights.import_ok * float(import_ok)
    score -= weights.repeated_token_penalty * float(repeated_token)
    score -= weights.invalid_format_penalty * float(invalid_format)
    score -= weights.length_penalty * clamp01(normalized_length)
    score -= weights.unsafe_api_penalty * float(unsafe_api)
    return score


def group_relative_advantages(rewards: Sequence[float], eps: float = 1e-8) -> list[float]:
    """Normalize rewards inside one rollout group."""

    if not rewards:
        return []
    mu = mean(rewards)
    sigma = pstdev(rewards)
    denom = sigma + eps
    return [(r - mu) / denom for r in rewards]


def positive_advantage_weights(rewards: Sequence[float]) -> list[float]:
    return [max(0.0, value) for value in group_relative_advantages(rewards)]


def pass_at_k_upper_bound(pass_rate: float, k: int) -> float:
    """IID intuition only; empirical coverage should be reported instead."""

    if k < 0:
        raise ValueError("k must be non-negative")
    p = clamp01(pass_rate)
    return 1.0 - (1.0 - p) ** k


def empirical_coverage(groups: Iterable[Sequence[bool]]) -> float:
    groups = list(groups)
    if not groups:
        return 0.0
    return sum(any(g) for g in groups) / len(groups)


def selected_rate(selected_passed: Iterable[bool]) -> float:
    values = list(selected_passed)
    if not values:
        return 0.0
    return sum(values) / len(values)


def selector_gap(groups: Iterable[Sequence[bool]], selected_passed: Iterable[bool]) -> float:
    return empirical_coverage(groups) - selected_rate(selected_passed)


def cost_adjusted_gain(quality: float, baseline_quality: float, train_cost: float, infer_cost: float, review_cost: float, rho: float = 1.0, gamma: float = 1.0) -> float:
    denom = train_cost + rho * infer_cost + gamma * review_cost
    if denom <= 0:
        raise ValueError("total cost must be positive")
    return (quality - baseline_quality) / denom


def dense_training_flops(params: float, tokens: float) -> float:
    return 6.0 * params * tokens


def weight_memory_bytes(params: float, bytes_per_param: float) -> float:
    return params * bytes_per_param


def l2_norm(values: Sequence[float]) -> float:
    return sqrt(sum(v * v for v in values))
