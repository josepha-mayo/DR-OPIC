from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class RouteDecision:
    domain: str | None
    accepted: bool
    reason: str
    confidence: float


@dataclass(frozen=True)
class DomainRule:
    name: str
    include_patterns: tuple[str, ...]
    exclude_patterns: tuple[str, ...] = ()
    min_hits: int = 1


DEFAULT_DOMAINS = (
    DomainRule("python", (r"\bpython\b", r"\bpytest\b", r"\bdef\s+\w+\(", r"\btraceback\b")),
    DomainRule("typescript", (r"\btypescript\b", r"\btsx?\b", r"\bnode\b", r"\breact\b")),
    DomainRule("sql", (r"\bsql\b", r"\bpostgres\b", r"\bselect\b", r"\bmigration\b")),
    DomainRule("repo_patch", (r"\bdiff\b", r"\bpatch\b", r"\bfile\b", r"\btest\b")),
)

DEFAULT_OOD = (
    r"\bmedical\b",
    r"\blegal\b",
    r"\bfinancial advice\b",
    r"\btherapy\b",
    r"\brelationship\b",
)


def route_task(prompt: str, domains: Iterable[DomainRule] = DEFAULT_DOMAINS, ood_patterns: Iterable[str] = DEFAULT_OOD) -> RouteDecision:
    text = prompt.lower()
    for pattern in ood_patterns:
        if re.search(pattern, text, flags=re.I):
            return RouteDecision(None, False, f"ood:{pattern}", 1.0)

    scored: list[tuple[int, DomainRule]] = []
    for domain in domains:
        if any(re.search(pattern, text, flags=re.I) for pattern in domain.exclude_patterns):
            continue
        hits = sum(bool(re.search(pattern, text, flags=re.I)) for pattern in domain.include_patterns)
        if hits >= domain.min_hits:
            scored.append((hits, domain))

    if not scored:
        return RouteDecision(None, False, "no_validated_domain", 0.0)

    hits, domain = max(scored, key=lambda item: item[0])
    confidence = min(1.0, hits / max(1, len(domain.include_patterns)))
    return RouteDecision(domain.name, True, "matched_domain", confidence)
