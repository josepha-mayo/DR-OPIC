from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .schemas import Candidate, Task
from .verifier import PythonTask, verify_python


@dataclass(frozen=True)
class ReplayReport:
    total: int
    before_passes: int
    after_passes: int
    repaired_passes: int
    rows: list[dict]


def replay_python_candidates(task: Task, candidates: Iterable[Candidate]) -> ReplayReport:
    if not task.entrypoint or not task.tests:
        raise ValueError("python replay requires entrypoint and tests")
    py_task = PythonTask(task.prompt, task.entrypoint, task.tests, task_id=task.task_id)
    rows = []
    before = 0
    after = 0
    repaired = 0
    for cand in candidates:
        result = verify_python(cand.code, py_task)
        before += int(cand.passed)
        after += int(result.passed)
        repaired += int((not cand.passed) and result.passed)
        rows.append({"task_id": task.task_id, "source": cand.source, "before_passed": cand.passed, "after_passed": result.passed, "observation": result.observation})
    return ReplayReport(len(rows), before, after, repaired, rows)
