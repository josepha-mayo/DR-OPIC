from __future__ import annotations

import ast
import difflib
import re
from dataclasses import dataclass
from typing import Iterable, Sequence

from .schemas import Candidate


@dataclass(frozen=True)
class CandidateScoreConfig:
    verifier_pass: float = 2.0
    fuzz_fraction: float = 0.5
    logprob: float = 0.3
    edit_distance: float = 0.8
    complexity: float = 0.2
    length: float = 0.1
    rare_dependency: float = 0.3


def normalized_edit_distance(left: str, right: str) -> float:
    if not left and not right:
        return 0.0
    ratio = difflib.SequenceMatcher(a=left, b=right).ratio()
    return 1.0 - ratio


def rough_complexity(code: str) -> float:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return 1.0
    branches = sum(isinstance(n, (ast.If, ast.For, ast.While, ast.Try, ast.BoolOp, ast.Match)) for n in ast.walk(tree))
    funcs = sum(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) for n in ast.walk(tree))
    return min(1.0, (branches + 0.5 * funcs) / 20.0)


def rare_dependency_count(code: str, common: Iterable[str] | None = None) -> int:
    common_set = set(common or {"math", "re", "collections", "itertools", "functools", "heapq", "bisect", "typing", "json"})
    mods = set(re.findall(r"^\s*(?:from|import)\s+([A-Za-z_][\w.]*)", code, flags=re.M))
    roots = {m.split(".")[0] for m in mods}
    return len(roots - common_set)


def score_candidate(
    candidate: Candidate,
    failed_code: str = "",
    fuzz_fraction: float = 0.0,
    config: CandidateScoreConfig = CandidateScoreConfig(),
) -> float:
    logp = candidate.logprob_per_token if candidate.logprob_per_token is not None else 0.0
    edit = normalized_edit_distance(failed_code, candidate.code) if failed_code else 0.0
    comp = rough_complexity(candidate.code)
    dep = min(1.0, rare_dependency_count(candidate.code) / 5.0)
    length = min(1.0, len(candidate.code) / 8000.0)
    return (
        config.verifier_pass * float(candidate.passed)
        + config.fuzz_fraction * fuzz_fraction
        + config.logprob * logp
        - config.edit_distance * edit
        - config.complexity * comp
        - config.length * length
        - config.rare_dependency * dep
    )


def select_learnable_winner(candidates: Sequence[Candidate], failed_code: str = "") -> Candidate | None:
    verified = [c for c in candidates if c.passed]
    if not verified:
        return None
    return max(verified, key=lambda c: score_candidate(c, failed_code=failed_code))


def error_signature(observation: str) -> str:
    text = observation.strip().splitlines()
    if not text:
        return "unknown"
    joined = "\n".join(text[-3:])
    joined = re.sub(r"line \d+", "line N", joined)
    joined = re.sub(r"\d+", "N", joined)
    joined = re.sub(r"['\"][^'\"]{20,}['\"]", "'...'", joined)
    return joined[-240:]
