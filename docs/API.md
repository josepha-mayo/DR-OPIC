# API Reference

## Verify Python

```python
from dr_opic.verifier import PythonTask, verify_python

task = PythonTask(
    prompt="Implement inc(x).",
    entrypoint="inc",
    tests="assert inc(2) == 3",
)

result = verify_python("def inc(x):\n    return x + 1\n", task)
assert result.passed
```

## Run A Forge Round

```python
from dr_opic.forge import build_round_artifacts, repair_failures, rollout_python_task
from dr_opic.schemas import Task

task = Task(
    task_id="reverse_words",
    prompt="Implement reverse_words(s).",
    entrypoint="reverse_words",
    tests="assert reverse_words('one two') == 'two one'",
)

def student(task, index):
    return "def reverse_words(s):\n    return s\n"

def repair(task, failed, round_index):
    return "def reverse_words(s):\n    return ' '.join(reversed(s.split()))\n"

group = rollout_python_task(task, student, k=1)
repairs = repair_failures(group, repair)
artifacts = build_round_artifacts(group, repairs)
```

## Save Artifacts

```python
from dr_opic.forge import save_round_artifacts

save_round_artifacts("outputs/round_001", artifacts)
```

## Counterfactual Delta Spans

```python
from dr_opic.delta import build_delta_example

failed = "def f(x):\n    return x\n"
fixed = "def f(x):\n    return x + 1\n"
example = build_delta_example("inc", failed, fixed)
print(example.positive_token_indices)
print(example.negative_token_indices)
```

`positive_token_indices` are fixed-code tokens to increase. `negative_token_indices`
are failed-code tokens that can be subtracted relative to a reference model.

## Verifier-ZPD Scheduling

```python
from dr_opic.scheduler import schedule_group, training_mix

scheduled = schedule_group(group, repairs)
print(scheduled.bucket, scheduled.train_weight)
print(training_mix([scheduled]))
```

Buckets are `mastered`, `zpd_train`, `repair_train`, `decompose`, `eval_only`,
and `discard`.

## Route And Safety Check

```python
from dr_opic.routing import route_task
from dr_opic.safety import classify_coding_safety, selective_accept

prompt = "Fix this Python traceback and add pytest coverage."
route = route_task(prompt)
safety = classify_coding_safety(prompt)
accepted = selective_accept(route.accepted, safety.allowed, route.confidence)
```

## Model Estimate

```python
from dr_opic.compression import estimate_dense_model

estimate = estimate_dense_model(3.09e9)
print(estimate.fp16_gb, estimate.q4_gb)
```
