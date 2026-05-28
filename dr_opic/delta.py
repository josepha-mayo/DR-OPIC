from __future__ import annotations

import difflib
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class DeltaSpan:
    tag: str
    failed_start: int
    failed_end: int
    fixed_start: int
    fixed_end: int
    failed_text: str
    fixed_text: str


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


def delta_training_record(task_id: str, failed: str, fixed: str) -> dict:
    return {
        "task_id": task_id,
        "delta_spans": [span.__dict__ for span in token_delta_spans(failed, fixed)],
    }
