from __future__ import annotations

import ast
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .maths import verifier_reward


UNSAFE_PATTERNS = (
    r"\bos\.system\b",
    r"\bsubprocess\.",
    r"\bshutil\.rmtree\b",
    r"\beval\s*\(",
    r"\bexec\s*\(",
    r"\bopen\s*\([^)]*['\"]w",
    r"\bsocket\.",
)


@dataclass(frozen=True)
class PythonTask:
    prompt: str
    entrypoint: str
    tests: str
    task_id: str = "task"
    timeout_s: float = 5.0
    allow_unsafe: bool = False


@dataclass(frozen=True)
class VerificationResult:
    passed: bool
    reward: float
    observation: str
    syntax_ok: bool
    import_ok: bool
    public_fraction: float = 0.0
    unsafe_api: bool = False
    repeated_token: bool = False
    invalid_format: bool = False


def extract_code(text: str) -> str:
    """Extract code from a markdown response or return the text as-is."""

    match = re.search(r"```(?:python|py)?\s*(.*?)```", text, flags=re.S | re.I)
    if match:
        return match.group(1).strip() + "\n"
    return text.strip() + "\n"


def has_repeated_token_collapse(code: str, threshold: int = 18) -> bool:
    tokens = re.findall(r"\S+", code)
    if not tokens:
        return True
    run = 1
    prev = None
    for tok in tokens:
        if tok == prev:
            run += 1
            if run >= threshold:
                return True
        else:
            run = 1
            prev = tok
    return False


def unsafe_api_detected(code: str) -> bool:
    return any(re.search(pattern, code) for pattern in UNSAFE_PATTERNS)


def static_check_python(code: str, entrypoint: str | None = None, allow_unsafe: bool = False) -> VerificationResult:
    repeated = has_repeated_token_collapse(code)
    unsafe = unsafe_api_detected(code)
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        obs = f"SyntaxError: {exc.msg} at line {exc.lineno}"
        return VerificationResult(False, verifier_reward(final_pass=False, repeated_token=repeated, unsafe_api=unsafe, invalid_format=True), obs, False, False, unsafe_api=unsafe, repeated_token=repeated, invalid_format=True)

    names = {node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
    invalid_format = bool(entrypoint and entrypoint not in names)
    import_ok = True
    if unsafe and not allow_unsafe:
        import_ok = False
    reward = verifier_reward(final_pass=False, syntax_ok=True, import_ok=import_ok, unsafe_api=unsafe, repeated_token=repeated, invalid_format=invalid_format)
    obs = "static_ok" if import_ok and not invalid_format and not repeated else "static_guard_failed"
    return VerificationResult(False, reward, obs, True, import_ok, unsafe_api=unsafe, repeated_token=repeated, invalid_format=invalid_format)


def verify_python(response: str, task: PythonTask, python: str | None = None) -> VerificationResult:
    """Run task tests against a candidate in a temporary process."""

    code = extract_code(response)
    static = static_check_python(code, task.entrypoint, allow_unsafe=task.allow_unsafe)
    if not static.syntax_ok or not static.import_ok or static.invalid_format or static.repeated_token:
        return static

    script = code + "\n\n" + task.tests + "\n"
    exe = python or sys.executable
    with tempfile.TemporaryDirectory(prefix="dr_opic_verify_") as tmp:
        path = Path(tmp) / "candidate.py"
        path.write_text(script, encoding="utf-8")
        try:
            proc = subprocess.run(
                [exe, str(path)],
                cwd=tmp,
                text=True,
                capture_output=True,
                timeout=task.timeout_s,
            )
        except subprocess.TimeoutExpired:
            return VerificationResult(False, verifier_reward(final_pass=False, syntax_ok=True, import_ok=True), "TimeoutExpired", True, True)

    if proc.returncode == 0:
        return VerificationResult(True, verifier_reward(final_pass=True, public_fraction=1.0, syntax_ok=True, import_ok=True), "passed", True, True, public_fraction=1.0)
    obs = (proc.stderr or proc.stdout or "failed").strip()
    return VerificationResult(False, verifier_reward(final_pass=False, public_fraction=0.0, syntax_ok=True, import_ok=True), obs[:4000], True, True)
