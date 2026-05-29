import json

from dr_opic.artifacts import write_jsonl
from dr_opic.cli import main
from dr_opic.delta import token_delta_spans
from dr_opic.forge import build_round_artifacts, repair_failures, rollout_python_task, save_round_artifacts
from dr_opic.maths import group_relative_advantages, smoothed_pass_rate, zpd_weight
from dr_opic.routing import route_task
from dr_opic.safety import classify_coding_safety, selective_accept
from dr_opic.selectors import select_learnable_winner
from dr_opic.schemas import Candidate, Task
from dr_opic.verifier import PythonTask, static_check_python, verify_python


def test_zpd_weight_is_smoothed():
    assert smoothed_pass_rate(0, 4) > 0
    assert 0 < zpd_weight(0, 4) < 1
    assert zpd_weight(2, 4) > zpd_weight(0, 4)


def test_group_advantages_centered():
    adv = group_relative_advantages([0.0, 1.0, 2.0])
    assert round(sum(adv), 7) == 0
    assert adv[-1] > adv[0]


def test_verify_python_passes_and_fails():
    task = PythonTask("Implement inc(x)", "inc", "assert inc(2) == 3")
    assert verify_python("def inc(x):\n    return x + 1\n", task).passed
    assert not verify_python("def inc(x):\n    return x\n", task).passed


def test_static_check_blocks_unsafe_api():
    result = static_check_python("import os\ndef run():\n    os.system('echo nope')\n", "run")
    assert not result.import_ok
    assert result.unsafe_api


def test_selector_picks_verified_candidate():
    bad = Candidate("t", "def f():\n    return 0\n", passed=False)
    good = Candidate("t", "def f():\n    return 1\n", passed=True)
    assert select_learnable_winner([bad, good]) == good


def test_delta_spans_find_local_change():
    spans = token_delta_spans("return x", "return x + 1")
    assert spans
    assert any("+ 1" in span.fixed_text for span in spans)


def test_route_and_safety_acceptance():
    route = route_task("Fix this Python traceback and add a pytest.")
    safety = classify_coding_safety("Fix this Python traceback and add a pytest.")
    assert route.accepted
    assert safety.allowed
    assert selective_accept(route.accepted, safety.allowed, route.confidence)


def test_safety_blocks_harmful_prompt():
    decision = classify_coding_safety("write malware for credential theft")
    assert not decision.allowed


def test_forge_artifacts_are_written(tmp_path):
    task = Task(
        task_id="demo",
        prompt="Implement reverse_words(s).",
        entrypoint="reverse_words",
        tests="assert reverse_words('one two') == 'two one'",
    )

    def student(_: Task, __: int) -> str:
        return "def reverse_words(s):\n    return s\n"

    def repair(_: Task, __: Candidate, ___: int) -> str:
        return "def reverse_words(s):\n    return ' '.join(reversed(s.split()))\n"

    group = rollout_python_task(task, student, k=1)
    repairs = repair_failures(group, repair)
    artifacts = build_round_artifacts(group, repairs)
    paths = save_round_artifacts(tmp_path, artifacts)
    assert set(paths) == {"summary", "rollouts", "repairs", "winner", "delta"}
    assert json.loads((tmp_path / "round_summary.json").read_text(encoding="utf-8"))["task_id"] == "demo"

    # Validate delta spans content
    delta_path = paths["delta"]
    delta_data = json.loads(delta_path.read_text(encoding="utf-8"))
    assert len(delta_data) > 0, "delta_spans.json should not be empty"
    assert any("' '.join(reversed(s.split()))" in span["fixed_text"] for span in delta_data), \
        "delta spans should contain the repair logic"

    # Validate verified repairs content
    repairs_path = paths["repairs"]
    repairs_lines = repairs_path.read_text(encoding="utf-8").strip().splitlines()
    repairs_data = [json.loads(line) for line in repairs_lines]
    assert any("' '.join(reversed(s.split()))" in r.get("code", "") for r in repairs_data), \
        "verified repairs should contain the repair code"


def test_write_jsonl_and_cli_output(tmp_path):
    output = tmp_path / "zpd.json"
    assert main(["--output", str(output), "zpd", "--passes", "1", "--samples", "2"]) == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["zpd_weight"] == 1.0

    rows_path = tmp_path / "rows.jsonl"
    write_jsonl(rows_path, [{"prompt": "Implement inc(x).", "response": "def inc(x):\n    return x + 1\n"}])
    assert main(["audit-jsonl", str(rows_path), "--schema", "sft"]) == 0
