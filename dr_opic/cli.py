from __future__ import annotations

import argparse
import json
from pathlib import Path

from .datasets import audit_rows, read_jsonl
from .forge import build_round_artifacts, repair_failures, rollout_python_task
from .maths import smoothed_pass_rate, zpd_weight
from .schemas import Task
from .verifier import PythonTask, verify_python


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="dr-opic")
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

    sub.add_parser("forge-demo", help="run a tiny student-first forge loop")

    args = parser.parse_args(argv)
    if args.cmd == "zpd":
        print(json.dumps({"p_tilde": smoothed_pass_rate(args.passes, args.samples), "zpd_weight": zpd_weight(args.passes, args.samples)}, indent=2))
        return 0
    if args.cmd == "audit-jsonl":
        rows = read_jsonl(args.path)
        print(json.dumps(audit_rows(rows, args.schema), indent=2))
        return 0
    if args.cmd == "verify-python":
        payload = json.loads(Path(args.task_json).read_text(encoding="utf-8"))
        task = PythonTask(**payload)
        code = Path(args.code).read_text(encoding="utf-8") if args.code else payload.get("code", "")
        result = verify_python(code, task)
        print(json.dumps(result.__dict__, indent=2))
        return 0
    if args.cmd == "forge-demo":
        _forge_demo()
        return 0
    return 2


def _forge_demo() -> None:
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
    print(json.dumps(build_round_artifacts(group, repairs), indent=2, default=str))


if __name__ == "__main__":
    raise SystemExit(main())
