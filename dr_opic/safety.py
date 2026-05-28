from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable


HARMFUL_CODING_PATTERNS = (
    r"\bmalware\b",
    r"\bcredential theft\b",
    r"\bsteal passwords?\b",
    r"\bexfiltrat",
    r"\bbypass\b.*\bsecurity\b",
    r"\bransomware\b",
    r"\bkeylogger\b",
    r"\bphishing\b",
)

BENIGN_DUAL_USE_HINTS = (
    r"\bdefensive\b",
    r"\bdetection\b",
    r"\blog analysis\b",
    r"\bunit test\b",
    r"\btoy\b",
    r"\bsandbox\b",
)


@dataclass(frozen=True)
class SafetyDecision:
    allowed: bool
    label: str
    reason: str


def classify_coding_safety(prompt: str, harmful_patterns: Iterable[str] = HARMFUL_CODING_PATTERNS) -> SafetyDecision:
    text = prompt.lower()
    harmful = [p for p in harmful_patterns if re.search(p, text, flags=re.I)]
    if not harmful:
        return SafetyDecision(True, "benign", "no_harmful_pattern")
    if any(re.search(p, text, flags=re.I) for p in BENIGN_DUAL_USE_HINTS):
        return SafetyDecision(True, "benign_dual_use", "defensive_or_sandbox_context")
    return SafetyDecision(False, "unsafe", harmful[0])


def selective_accept(in_domain: bool, safe: bool, confidence: float, threshold: float = 0.25) -> bool:
    return in_domain and safe and confidence >= threshold
