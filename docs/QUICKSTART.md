# Quickstart

This is the fastest way to run DR-OPIC locally.

## 1. Install

```bash
python -m pip install -e ".[dev]"
```

## 2. Run Tests

```bash
python -m pytest -q
```

## 3. Run The Forge Demo

```bash
python -m dr_opic.cli forge-demo
```

The output contains:

- `passes` and `samples`
- `zpd_weight`
- group-relative `advantages`
- raw `rollouts`
- verified `repairs`
- selected `winner`
- local `delta` spans from failed code to fixed code

Write the same output to artifact files:

```bash
python -m dr_opic.cli --output outputs\demo forge-demo
```

## 4. Verify A Candidate

```bash
python -m dr_opic.cli verify-python examples/python_task.json --code examples/reverse_words_good.py
```

Try the failing version:

```bash
python -m dr_opic.cli verify-python examples/python_task.json --code examples/reverse_words_bad.py
```

Return a failing process exit code when verification fails:

```bash
python -m dr_opic.cli verify-python examples/python_task.json --code examples/reverse_words_bad.py --fail-on-error
```

## 5. Audit External Rows

This repo does not include training rows. Keep private datasets outside the repo.

Audit SFT rows:

```bash
python -m dr_opic.cli audit-jsonl C:\datasets\slm\sft.jsonl --schema sft
```

Audit preference rows:

```bash
python -m dr_opic.cli audit-jsonl C:\datasets\slm\preferences.jsonl --schema preference
```

## 6. Programmatic Use

```python
from dr_opic.schemas import Task
from dr_opic.forge import rollout_python_task, repair_failures, build_round_artifacts

task = Task(
    task_id="reverse_words",
    prompt="Implement reverse_words(s).",
    entrypoint="reverse_words",
    tests="assert reverse_words('one two') == 'two one'",
)

def student(task, i):
    return "def reverse_words(s):\n    return s\n"

def repair(task, failed, round_index):
    return "def reverse_words(s):\n    return ' '.join(reversed(s.split()))\n"

group = rollout_python_task(task, student, k=1)
repairs = repair_failures(group, repair, rounds=1)
artifacts = build_round_artifacts(group, repairs)
print(artifacts["winner"])
```
