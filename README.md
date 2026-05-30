# DR-OPIC

DR-OPIC is a runnable Python framework for coding SLM experiments:

**Domain-Routed On-Policy Iterative Correction**.

It does three concrete things:

1. Runs student coding attempts against executable tests.
2. Builds verified repair/delta/preference training records from real failures.
3. Reports the metrics that matter for a coding SLM: `greedy@1`,
   `coverage@K`, `selected@K`, `selector_gap`, and `repair@1`.

No private datasets, PDFs, model weights, or Kaggle outputs are included in this
repo.

## Install

From the repo root:

```bash
python -m pip install -e ".[dev]"
```

Run the test suite:

```bash
python -m pytest -q
```

Expected result:

```text
7 passed
```

## Run The Built-In Demo

The demo uses one toy Python task. The first student answer fails, the second
passes, a repair candidate is verified, and DR-OPIC emits rollout, repair, ZPD,
advantage, winner, and delta-span records.

```bash
python -m dr_opic.cli forge-demo
```

You can also use the installed console command:

```bash
dr-opic forge-demo
```

Write a stable artifact bundle:

```bash
python -m dr_opic.cli --output outputs\demo forge-demo
```

This creates:

```text
round_summary.json
student_rollouts.jsonl
verified_repairs.jsonl
learnable_winner.json
delta_spans.json
```

## Verify A Python Candidate

Create a task JSON:

```json
{
  "prompt": "Implement reverse_words(s) returning words in reverse order.",
  "entrypoint": "reverse_words",
  "tests": "assert reverse_words('one two three') == 'three two one'\nassert reverse_words('solo') == 'solo'",
  "task_id": "reverse_words_demo"
}
```

Run verification against code embedded in the JSON or from a file:

```bash
python -m dr_opic.cli verify-python examples/python_task.json --code examples/reverse_words_good.py
```

Expected output includes:

```json
{
  "passed": true,
  "observation": "passed"
}
```

## Compute ZPD Weight

```bash
python -m dr_opic.cli zpd --passes 2 --samples 5
```

This prints Jeffreys-smoothed pass rate and the ZPD weight:

```text
p_tilde = (passes + 0.5) / (samples + 1)
w_zpd = 4 * p_tilde * (1 - p_tilde)
```

## Route And Estimate

Route a prompt through the domain and safety checks:

```bash
python -m dr_opic.cli route "Fix this Python traceback and add pytest coverage"
```

Estimate dense model memory and per-token compute:

```bash
python -m dr_opic.cli estimate-model --params 3.09e9
```

## Audit A JSONL Training File

The repo does not ship training data. To audit an SFT JSONL stored outside this
repo:

```bash
python -m dr_opic.cli audit-jsonl C:\datasets\slm\sft.jsonl --schema sft
```

Expected SFT schema:

```json
{"prompt": "...", "response": "..."}
```

For preference rows:

```bash
python -m dr_opic.cli audit-jsonl C:\datasets\slm\preferences.jsonl --schema preference
```

Required preference fields:

```json
{"prompt": "...", "chosen": "...", "rejected": "..."}
```

## Modules

- `dr_opic.maths`: ZPD, rewards, advantages, coverage metrics, cost estimates.
- `dr_opic.verifier`: Python code extraction, static checks, test execution.
- `dr_opic.forge`: student-first rollout, repair, and artifact construction.
- `dr_opic.selectors`: verified learnable-winner selection.
- `dr_opic.delta`: token/line delta spans between failed and fixed code.
- `dr_opic.preference`: scalar helpers for verified DPO/ORPO-style pairs.
- `dr_opic.datasets`: JSONL schema and quality audit helpers.
- `dr_opic.replay`: deterministic replay certification.
- `dr_opic.routing`: domain routing and abstention helper.
- `dr_opic.safety`: simple coding-safety acceptance helper.
- `dr_opic.compression`: memory/compute estimates and retention gates.
- `dr_opic.losses`: optional PyTorch losses for SFT, delta, DPO, and RLVR.

More detail:

- [Quickstart](docs/QUICKSTART.md)
- [Architecture](docs/ARCHITECTURE.md)
- [API Reference](docs/API.md)
- [Math Notes](docs/MATH.md)
- [Release Protocol](docs/RELEASE_PROTOCOL.md)
- [Notebook Tutorial](notebooks/DR_OPIC_Tutorial.ipynb)

## Safety Scope

`verify-python` executes candidate code in a temporary subprocess with a timeout.
That is useful for local research, but it is not a security sandbox. Run untrusted
model code inside a container or VM.
