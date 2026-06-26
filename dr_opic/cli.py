from __future__ import annotations

import argparse
import json
from pathlib import Path

from .artifacts import dumps_json, write_json
from .compression import estimate_dense_model
from .datasets import audit_rows, read_jsonl
from .forge import build_round_artifacts, repair_failures, rollout_python_task, save_round_artifacts
from .maths import smoothed_pass_rate, zpd_weight
from .routing import route_task
from .safety import classify_coding_safety, selective_accept
from .scheduler import schedule_group, training_mix
from .schemas import Task
from .verifier import PythonTask, verify_python


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="dr-opic")
    parser.add_argument("--output", help="write JSON output to this file or directory, depending on command")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_zpd = sub.add_parser("zpd", help="compute smoothed pass rate and ZPD weight")
    p_zpd.add_argument("--passes", type=int, required=True)
    p_zpd.add_argument("--samples", type=int, required=True)

    p_audit = sub.add_parser("audit-jsonl", help="audit SFT or preference JSONL")
    p_audit.add_argument("path")
    p_audit.add_argument("--schema", choices=["sft", "preference"], default="sft")

    p_verify = sub.add_parser("verify-python", help="verify candidate code for a JSON task")
    p_verify.add_argument("task_json")
    p_verify.add_argument("--code")
    p_verify.add_argument("--fail-on-error", action="store_true", help="return exit code 1 when verification fails")

    sub.add_parser("forge-demo", help="run a tiny student-first forge loop")

    p_route = sub.add_parser("route", help="route and safety-check one prompt")
    p_route.add_argument("prompt")

    p_estimate = sub.add_parser("estimate-model", help="estimate dense model memory and flops")
    p_estimate.add_argument("--params", type=float, required=True, help="parameter count, e.g. 3.09e9")

    p_delta = sub.add_parser("delta", help="build a counterfactual delta-span record")
    p_delta.add_argument("--task-id", default="delta_task")
    p_delta.add_argument("--failed", required=True, help="path to failed code")
    p_delta.add_argument("--fixed", required=True, help="path to fixed code")

    sub.add_parser("schedule-demo", help="run the verifier-ZPD scheduler on the built-in demo")

    args = parser.parse_args(argv)
    if args.cmd == "zpd":
        return _emit({"p_tilde": smoothed_pass_rate(args.passes, args.samples), "zpd_weight": zpd_weight(args.passes, args.samples)}, args.output)
    if args.cmd == "audit-jsonl":
        def _stream_jsonl(path):
            with open(path, encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        yield json.loads(line)
        rows = _stream_jsonl(args.path)
        return _emit(audit_rows(rows, args.schema), args.output)
    if args.cmd == "verify-python":
        payload = json.loads(Path(args.task_json).read_text(encoding="utf-8"))
        task = _python_task_from_payload(payload)
        code = Path(args.code).read_text(encoding="utf-8") if args.code else payload.get("code", "")
        result = verify_python(code, task)
        _emit(result.__dict__, args.output)
        return 1 if args.fail_on_error and not result.passed else 0
    if args.cmd == "forge-demo":
        _forge_demo(args.output)
        return 0
    if args.cmd == "route":
        route = route_task(args.prompt)
        safety = classify_coding_safety(args.prompt)
        return _emit(
            {
                "route": route.__dict__,
                "safety": safety.__dict__,
                "accepted": selective_accept(route.accepted, safety.allowed, route.confidence),
            },
            args.output,
        )
    if args.cmd == "estimate-model":
        return _emit(estimate_dense_model(args.params).__dict__, args.output)
    if args.cmd == "delta":
        from .delta import delta_training_record

        return _emit(
            delta_training_record(
                args.task_id,
                Path(args.failed).read_text(encoding="utf-8"),
                Path(args.fixed).read_text(encoding="utf-8"),
            ),
            args.output,
        )
    if args.cmd == "schedule-demo":
        artifacts, scheduled = _schedule_demo()
        return _emit({"scheduled": scheduled.to_dict(), "mix": training_mix([scheduled]), "artifacts": artifacts}, args.output)
    return 2


def _forge_demo(output: str | None = None) -> None:
    task = Task(
        task_id="demo_reverse",
        prompt="Implement reverse_words(s) returning words in reverse order.",
        entrypoint="reverse_words",
        tests="assert reverse_words('one two three') == 'three two one'\nassert reverse_words('solo') == 'solo'",
    )

    def student(_: Task, i: int) -> str:
        if i == 0:
            return "def reverse_words(s):\n    return s\n"
        return "def reverse_words(s):\n    return ' '.join(reversed(s.split()))\n"

    def repair(_: Task, failed, __: int) -> str:
        return "def reverse_words(s):\n    return ' '.join(reversed(s.split()))\n"

    group = rollout_python_task(task, student, k=2)
    repairs = repair_failures(group, repair, rounds=1)
    artifacts = build_round_artifacts(group, repairs)
    if output:
        save_round_artifacts(output, artifacts)
        print(str(Path(output)))
    else:
        print(dumps_json(artifacts))


def _schedule_demo():
    task = Task(
        task_id="demo_reverse",
        prompt="Implement reverse_words(s) returning words in reverse order.",
        entrypoint="reverse_words",
        tests="assert reverse_words('one two three') == 'three two one'\nassert reverse_words('solo') == 'solo'",
    )

    def student(_: Task, __: int) -> str:
        return "def reverse_words(s):\n    return s\n"

    def repair(_: Task, failed, __: int) -> str:
        return "def reverse_words(s):\n    return ' '.join(reversed(s.split()))\n"

    group = rollout_python_task(task, student, k=1)
    repairs = repair_failures(group, repair, rounds=1)
    artifacts = build_round_artifacts(group, repairs)
    return artifacts, schedule_group(group, repairs)


def _emit(payload: object, output: str | None = None) -> int:
    if output:
        write_json(output, payload)
    print(dumps_json(payload))
    return 0


def _python_task_from_payload(payload: dict) -> PythonTask:
    allowed = set(PythonTask.__dataclass_fields__)
    return PythonTask(**{k: v for k, v in payload.items() if k in allowed})


if __name__ == "__main__":
    raise SystemExit(main())
