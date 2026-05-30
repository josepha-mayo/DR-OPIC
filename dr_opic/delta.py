from __future__ import annotations

import difflib
import re
from dataclasses import asdict, dataclass
from typing import Literal


@dataclass(frozen=True)
class DeltaSpan:
    tag: str
    failed_start: int
    failed_end: int
    fixed_start: int
    fixed_end: int
    failed_text: str
    fixed_text: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class DeltaExample:
    task_id: str
    failed: str
    fixed: str
    token_spans: tuple[DeltaSpan, ...]
    line_spans: tuple[DeltaSpan, ...]
    positive_token_indices: tuple[int, ...]
    negative_token_indices: tuple[int, ...]
    shared_token_ratio: float
    edit_token_ratio: float
    objective: str = "counterfactual_delta_subtraction"

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "objective": self.objective,
            "failed": self.failed,
            "fixed": self.fixed,
            "token_spans": [span.to_dict() for span in self.token_spans],
            "line_spans": [span.to_dict() for span in self.line_spans],
            "positive_token_indices": list(self.positive_token_indices),
            "negative_token_indices": list(self.negative_token_indices),
            "shared_token_ratio": self.shared_token_ratio,
            "edit_token_ratio": self.edit_token_ratio,
        }


def code_tokens(text: str) -> list[str]:
    return re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)


def token_delta_spans(failed: str, fixed: str) -> list[DeltaSpan]:
    left = code_tokens(failed)
    right = code_tokens(fixed)
    matcher = difflib.SequenceMatcher(a=left, b=right)
    spans: list[DeltaSpan] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        spans.append(DeltaSpan(tag, i1, i2, j1, j2, " ".join(left[i1:i2]), " ".join(right[j1:j2])))
    return spans


def line_delta_spans(failed: str, fixed: str) -> list[DeltaSpan]:
    left = failed.splitlines()
    right = fixed.splitlines()
    matcher = difflib.SequenceMatcher(a=left, b=right)
    spans: list[DeltaSpan] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        spans.append(DeltaSpan(tag, i1, i2, j1, j2, "\n".join(left[i1:i2]), "\n".join(right[j1:j2])))
    return spans


def delta_masks(failed: str, fixed: str) -> tuple[list[int], list[int]]:
    """Return positive fixed-token indices and negative failed-token indices."""

    left = code_tokens(failed)
    right = code_tokens(fixed)
    matcher = difflib.SequenceMatcher(a=left, b=right)
    positive: list[int] = []
    negative: list[int] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        if tag in {"insert", "replace"}:
            positive.extend(range(j1, j2))
        if tag in {"delete", "replace"}:
            negative.extend(range(i1, i2))
    return positive, negative


def shared_token_ratio(failed: str, fixed: str) -> float:
    left = code_tokens(failed)
    right = code_tokens(fixed)
    if not left and not right:
        return 1.0
    matcher = difflib.SequenceMatcher(a=left, b=right)
    shared = sum(i2 - i1 for tag, i1, i2, _, _ in matcher.get_opcodes() if tag == "equal")
    return shared / max(len(left), len(right), 1)


def edit_token_ratio(failed: str, fixed: str) -> float:
    return 1.0 - shared_token_ratio(failed, fixed)


def build_delta_example(task_id: str, failed: str, fixed: str) -> DeltaExample:
    positive, negative = delta_masks(failed, fixed)
    return DeltaExample(
        task_id=task_id,
        failed=failed,
        fixed=fixed,
        token_spans=tuple(token_delta_spans(failed, fixed)),
        line_spans=tuple(line_delta_spans(failed, fixed)),
        positive_token_indices=tuple(positive),
        negative_token_indices=tuple(negative),
        shared_token_ratio=shared_token_ratio(failed, fixed),
        edit_token_ratio=edit_token_ratio(failed, fixed),
    )


def delta_training_record(task_id: str, failed: str, fixed: str, format: Literal["compact", "full"] = "full") -> dict:
    example = build_delta_example(task_id, failed, fixed)
    if format == "compact":
        return {
            "task_id": task_id,
            "delta_spans": [span.to_dict() for span in example.token_spans],
        }
    return example.to_dict()
