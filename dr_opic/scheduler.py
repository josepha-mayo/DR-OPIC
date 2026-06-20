from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from typing import Iterable, Literal

from .maths import clamp01, smoothed_pass_rate, zpd_weight
from .schemas import Candidate, RolloutGroup
from .selectors import error_signature, normalized_edit_distance, select_learnable_winner

Bucket = Literal["eval_only", "mastered", "zpd_train", "repair_train", "decompose", "discard"]


@dataclass(frozen=True)
class SchedulerConfig:
    mastered_threshold: float = 0.85
    repair_threshold: float = 0.20
    min_verifier_reliability: float = 0.50
    max_learnable_edit_ratio: float = 0.65
    min_train_weight: float = 0.05
    repair_bonus: float = 0.20
    novelty_floor: float = 0.25
    failure_balance_cap: float = 2.0


@dataclass(frozen=True)
class ScheduledTask:
    task_id: str
    bucket: Bucket
    p_tilde: float
    zpd_weight: float
    train_weight: float
    passes: int
    samples: int
    selected_passed: bool
    selector_gap: int
    failure_signature: str | None
    nearest_edit_ratio: float | None
    reason: str

    def to_dict(self) -> dict:
        return asdict(self)


def schedule_group(
    group: RolloutGroup,
    repairs: Iterable[Candidate] = (),
    config: SchedulerConfig = SchedulerConfig(),
    failure_frequencies: Counter[str] | None = None,
    novelty: float = 1.0,
) -> ScheduledTask:
    repairs = list(repairs)
    p_tilde = smoothed_pass_rate(group.passes, group.samples)
    base_zpd = zpd_weight(group.passes, group.samples, group.task.verifier_reliability)
    all_candidates = list(group.candidates) + repairs
    failure = _first_failure(group)
    winner = select_learnable_winner(all_candidates, failed_code=failure.code if failure else "")
    selected_passed = bool(winner and winner.passed)
    selector_gap = int(any(c.passed for c in group.candidates)) - int(selected_passed)
    signature = error_signature(failure.observation) if failure else None
    edit_ratio = normalized_edit_distance(failure.code, winner.code) if failure and winner else None
    repair_passed = any(c.passed for c in repairs)

    bucket, reason = _bucket(
        split=group.task.split,
        reliability=group.task.verifier_reliability,
        p_tilde=p_tilde,
        selected_passed=selected_passed,
        repair_passed=repair_passed,
        edit_ratio=edit_ratio,
        config=config,
    )
    train_signature = signature if signature is not None else ""
    train_weight = _train_weight(
        bucket=bucket,
        base_zpd=base_zpd,
        signature=train_signature,
        failure_frequencies=failure_frequencies or Counter(),
        novelty=novelty,
        repair_passed=repair_passed,
        config=config,
    )
    return ScheduledTask(
        task_id=group.task.task_id,
        bucket=bucket,
        p_tilde=p_tilde,
        zpd_weight=base_zpd,
        train_weight=train_weight,
        passes=group.passes,
        samples=group.samples,
        selected_passed=selected_passed,
        selector_gap=selector_gap,
        failure_signature=signature,
        nearest_edit_ratio=edit_ratio,
        reason=reason,
    )


def schedule_round(
    items: Iterable[tuple[RolloutGroup, Iterable[Candidate]]],
    config: SchedulerConfig = SchedulerConfig(),
) -> list[ScheduledTask]:
    materialized = [(group, list(repairs)) for group, repairs in items]
    frequencies: Counter[str] = Counter()
    for group, _ in materialized:
        failure = _first_failure(group)
        if failure:
            sig = error_signature(failure.observation)
            if sig:
                frequencies[sig] += 1
    return [schedule_group(group, repairs, config=config, failure_frequencies=frequencies) for group, repairs in materialized]


def training_mix(scheduled: Iterable[ScheduledTask]) -> dict:
    rows = list(scheduled)
    buckets = Counter(row.bucket for row in rows)
    return {
        "tasks": len(rows),
        "total_train_weight": sum(row.train_weight for row in rows),
        "buckets": dict(buckets),
        "weighted_buckets": {
            bucket: sum(row.train_weight for row in rows if row.bucket == bucket)
            for bucket in sorted(buckets)
        },
        "selected_task_ids": [
            row.task_id for row in sorted(rows, key=lambda r: r.train_weight, reverse=True) if row.train_weight > 0
        ],
    }


def _bucket(
    *,
    split: str,
    reliability: float,
    p_tilde: float,
    selected_passed: bool,
    repair_passed: bool,
    edit_ratio: float | None,
    config: SchedulerConfig,
) -> tuple[Bucket, str]:
    if split != "train":
        return "eval_only", "non_train_split"
    if reliability < config.min_verifier_reliability:
        return "discard", "low_verifier_reliability"
    if p_tilde >= config.mastered_threshold:
        return "mastered", "high_student_pass_rate"
    if repair_passed and edit_ratio is not None and edit_ratio <= config.max_learnable_edit_ratio:
        return "repair_train", "verified_close_repair"
    if selected_passed:
        return "repair_train", "verified_winner_from_failure"
    if config.repair_threshold <= p_tilde < config.mastered_threshold:
        return "zpd_train", "student_in_zpd"
    return "decompose", "too_hard_without_close_fix"


def _train_weight(
    *,
    bucket: Bucket,
    base_zpd: float,
    signature: str | None,
    failure_frequencies: Counter[str],
    novelty: float,
    repair_passed: bool,
    config: SchedulerConfig,
) -> float:
    if bucket in {"eval_only", "discard"}:
        return 0.0
    if bucket == "mastered":
        return config.min_train_weight
    balance = 1.0
    if signature:
        balance = min(config.failure_balance_cap, 1.0 / max(1, failure_frequencies[signature]) * config.failure_balance_cap)
    repair = 1.0 + (config.repair_bonus if repair_passed else 0.0)
    novelty_weight = max(config.novelty_floor, clamp01(novelty))
    if bucket == "decompose":
        return config.min_train_weight * novelty_weight
    return max(config.min_train_weight, base_zpd * balance * repair * novelty_weight)


def _first_failure(group: RolloutGroup) -> Candidate | None:
    return next((candidate for candidate in group.candidates if not candidate.passed), None)
