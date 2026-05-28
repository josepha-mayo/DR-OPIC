from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable


SFT_FIELDS = {"prompt", "response"}
PREF_REQUIRED = {"prompt", "chosen", "rejected"}
FORBIDDEN_PATTERNS = (
    r"\bTO" r"DO\b",
    r"\bFIX" r"ME\b",
    r"\b(?:fill|insert|write)\s+(?:the\s+)?code\s+here\b",
    r"\bpass\s*(?:#.*)?$",
    r"<\s*think\s*>",
)


def read_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON: {exc}") from exc
    return rows


def audit_rows(rows: Iterable[dict], schema: str = "sft") -> dict:
    issues: list[dict] = []
    prompts: Counter[str] = Counter()
    responses: Counter[str] = Counter()
    for idx, row in enumerate(rows, 1):
        if schema == "sft":
            if set(row) != SFT_FIELDS:
                issues.append({"line": idx, "issue": "schema", "fields": sorted(row)})
            prompt = str(row.get("prompt", ""))
            response = str(row.get("response", ""))
            prompts[prompt] += 1
            responses[response] += 1
            _common_text_issues(idx, prompt, response, issues)
        elif schema == "preference":
            if not PREF_REQUIRED.issubset(row):
                issues.append({"line": idx, "issue": "schema", "missing": sorted(PREF_REQUIRED - set(row))})
            prompt = str(row.get("prompt", ""))
            chosen = str(row.get("chosen", ""))
            rejected = str(row.get("rejected", ""))
            prompts[prompt] += 1
            responses[chosen] += 1
            if chosen.strip() == rejected.strip():
                issues.append({"line": idx, "issue": "chosen_equals_rejected"})
            _common_text_issues(idx, prompt, chosen, issues)
        else:
            raise ValueError(f"unknown schema: {schema}")
    duplicate_prompts = sum(1 for _, n in prompts.items() if n > 1)
    duplicate_responses = sum(1 for _, n in responses.items() if n > 1)
    return {
        "issues": issues,
        "issue_count": len(issues),
        "duplicate_prompts": duplicate_prompts,
        "duplicate_responses": duplicate_responses,
    }


def _common_text_issues(line: int, prompt: str, response: str, issues: list[dict]) -> None:
    if len(prompt.strip()) < 12:
        issues.append({"line": line, "issue": "short_prompt"})
    if len(response.strip()) < 12:
        issues.append({"line": line, "issue": "short_response"})
    for pattern in FORBIDDEN_PATTERNS:
        if re.search(pattern, response, flags=re.I | re.M):
            issues.append({"line": line, "issue": "forbidden_pattern", "pattern": pattern})
