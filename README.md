# DR-OPIC

DR-OPIC is a small-language-model framework for coding specialists:

**Domain-Routed On-Policy Iterative Correction** with verifier-grounded rewards,
learnable repair selection, delta-span subtraction, preference pairs, replay
certification, and optional directional weight ablation utilities.

The repository turns the local SLM forge notebooks into a reusable Python package.
It intentionally does not include private datasets, model weights, Kaggle output
folders, or manuscript PDFs.

## Core Idea

Static SFT asks a small model to imitate broad teacher answers. DR-OPIC makes the
student attempt first:

1. Route a task to a bounded coding domain or abstain.
2. Sample the current student on the task.
3. Run executable verifiers and collect failure observations.
4. Estimate task learnability with ZPD weighting.
5. Ask domain teachers or repair policies to fix the student's actual failures.
6. Keep only verified candidates.
7. Select the verified correction closest to the student's reachable behavior.
8. Train self-success, repair, delta-span, preference, and RLVR objectives.
9. Promote only when greedy, coverage, selected, repair, safety, and abstention
   gates pass.
10. Compress after behavior improves, then run retention/recovery gates.

## Install

```bash
python -m pip install -e .
python -m pip install -e ".[dev]"
```

Optional loss helpers support PyTorch:

```bash
python -m pip install -e ".[torch]"
```

## Quick Demo

```bash
dr-opic forge-demo
dr-opic zpd --passes 2 --samples 5
dr-opic audit-jsonl examples/sft_rows.jsonl --schema sft
dr-opic verify-python examples/python_task.json
```

## Package Map

- `dr_opic.maths`: rewards, ZPD weights, group advantages, coverage metrics.
- `dr_opic.verifier`: static code checks and subprocess Python test verifier.
- `dr_opic.forge`: student-first rollout and repair data construction.
- `dr_opic.selectors`: learnable winner scoring and selector-gap accounting.
- `dr_opic.delta`: token and line delta spans for local correction training.
- `dr_opic.preference`: verified DPO/ORPO-style preference helpers.
- `dr_opic.datasets`: JSONL schema validation and quality audits.
- `dr_opic.replay`: deterministic replay certification for stored candidates.
- `dr_opic.routing`: simple domain router and abstention decisions.
- `dr_opic.safety`: coding-safety classifier and selective acceptance helper.
- `dr_opic.compression`: memory/compute estimates and retention gates.
- `dr_opic.losses`: optional PyTorch losses for SFT, delta, DPO, and RLVR.
- `dr_opic.ablation`: refusal-direction and projection/repulsion math helpers.
- `dr_opic.cli`: runnable command line tools.

## Safety and Scope

This framework is for bounded coding SLM research. Verifiers execute code in a
temporary subprocess with timeouts, but they are not a security sandbox. Do not
run untrusted code outside an isolated container or VM. Production deployments
should use least-privilege tools, secret scanning, OOD abstention, and human
review gates.

## Minimal Python API

```python
from dr_opic.maths import zpd_weight
from dr_opic.verifier import PythonTask, verify_python
from dr_opic.selectors import CandidateScoreConfig, select_learnable_winner

task = PythonTask(
    prompt="Implement add_one(xs).",
    entrypoint="add_one",
    tests="assert add_one([1, 2]) == [2, 3]",
)

result = verify_python("def add_one(xs):\n    return [x + 1 for x in xs]\n", task)
print(result.passed, zpd_weight(passes=1, samples=3))
```

## Release Gate

A run should not be promoted only because one metric moved. Minimum release
artifacts should include:

- `student_rollouts_round_*.jsonl`
- `verified_candidates_round_*.jsonl`
- `zpd_weights_round_*.jsonl`
- `learnable_winners.jsonl`
- `delta_spans.jsonl`
- `selector_gap_report.json`
- base vs adapter metrics for greedy@1, coverage@K, selected@K, repair@1
- abstention and red-team safety summary
- contamination and dedupe report
