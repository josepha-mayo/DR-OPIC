from __future__ import annotations

import time
from collections.abc import Callable, Sequence

from .delta import delta_training_record
from .maths import group_relative_advantages, zpd_weight
from .schemas import Candidate, RolloutGroup, Task
from .selectors import error_signature, select_learnable_winner
from .verifier import PythonTask, verify_python

Generator = Callable[[Task, int], str]
Repairer = Callable[[Task, Candidate, int], str]


def repair_prompt(task: Task, failed_code: str, observation: str) -> str:
    tests = task.tests or ""
    return (
        f"Repair the candidate for task {task.task_id}.\n\n"
        f"Prompt:\n{task.prompt}\n\n"
        f"Verifier tests:\n```python\n{tests}\n```\n\n"
        f"Failed code:\n```python\n{failed_code}\n```\n\n"
        f"Observation:\n{observation}\n\n"
        "Return corrected code only."
    )


def rollout_python_task(task: Task, generator: Generator, k: int = 4) -> RolloutGroup:
    if not task.entrypoint or not task.tests:
        raise ValueError("python rollout requires entrypoint and tests")
    py_task = PythonTask(task.prompt, task.entrypoint, task.tests, task_id=task.task_id)
    candidates: list[Candidate] = []
    for i in range(k):
        start = time.perf_counter()
        code = generator(task, i)
        result = verify_python(code, py_task)
        candidates.append(
            Candidate(
                task_id=task.task_id,
                code=code,
                source="student",
                passed=result.passed,
                reward=result.reward,
                observation=result.observation,
                tokens=len(code.split()),
                latency_s=time.perf_counter() - start,
            )
        )
    return RolloutGroup(task, tuple(candidates))


def repair_failures(group: RolloutGroup, repairer: Repairer, rounds: int = 1) -> list[Candidate]:
    repaired: list[Candidate] = []
    if not group.task.entrypoint or not group.task.tests:
        return repaired
    py_task = PythonTask(group.task.prompt, group.task.entrypoint, group.task.tests, task_id=group.task.task_id)
    for failed in group.candidates:
        if failed.passed:
            continue
        for r in range(rounds):
            start = time.perf_counter()
            code = repairer(group.task, failed, r)
            result = verify_python(code, py_task)
            repaired.append(
                Candidate(
                    task_id=group.task.task_id,
                    code=code,
                    source="repair",
                    passed=result.passed,
                    reward=result.reward,
                    observation=result.observation,
                    tokens=len(code.split()),
                    latency_s=time.perf_counter() - start,
                    metadata={"round": r, "failure_signature": error_signature(failed.observation)},
                )
            )
            if result.passed:
                break
    return repaired


def build_round_artifacts(group: RolloutGroup, repaired: Sequence[Candidate] = ()) -> dict:
    all_candidates = list(group.candidates) + list(repaired)
    winner = select_learnable_winner(all_candidates, failed_code=_nearest_failure(group).code if group.candidates else "")
    rewards = [c.reward for c in group.candidates]
    zpd = zpd_weight(group.passes, group.samples, group.task.verifier_reliability)
    artifacts = {
        "task_id": group.task.task_id,
        "passes": group.passes,
        "samples": group.samples,
        "zpd_weight": zpd,
        "advantages": group_relative_advantages(rewards),
        "winner": winner.__dict__ if winner else None,
        "rollouts": [c.__dict__ for c in group.candidates],
        "repairs": [c.__dict__ for c in repaired],
    }
    if winner:
        failed = _nearest_failure(group)
        artifacts["delta"] = delta_training_record(group.task.task_id, failed.code, winner.code)
    return artifacts


def _nearest_failure(group: RolloutGroup) -> Candidate:
    failures = [c for c in group.candidates if not c.passed]
    return failures[0] if failures else group.candidates[0]
