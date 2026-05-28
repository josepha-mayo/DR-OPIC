# Architecture

DR-OPIC is intentionally small. The package provides the glue needed to run a
student-first coding SLM loop without bundling model weights or datasets.

## Data Flow

```text
Task
  -> route and safety check
  -> student rollout K times
  -> Python verifier
  -> ZPD and advantage calculation
  -> repair failed attempts
  -> verify repairs
  -> select learnable winner
  -> emit JSON/JSONL artifacts
  -> train with SFT, delta, preference, or RLVR losses
```

## Modules

`schemas.py`

Typed dataclasses for tasks, candidates, and rollout groups. These validate
required fields and serialize to JSON-safe dictionaries.

`verifier.py`

Extracts Python code, runs static guards, executes tests in a temporary
subprocess with a timeout, and returns a structured verification result.

`forge.py`

Runs the student-first loop: rollout, repair, artifact construction, and artifact
writing.

`selectors.py`

Scores verified candidates by pass status, edit distance to the failed attempt,
rough complexity, dependency burden, length, and optional model logprob.

`delta.py`

Builds token-level and line-level spans between failed code and fixed code.

`losses.py`

Optional PyTorch tensor losses for advantage-weighted SFT, delta-span training,
verified preference, and clipped RLVR.

`cli.py`

Thin command-line interface for verification, routing, ZPD, model estimates, and
the built-in forge demo.

## Artifact Contract

`python -m dr_opic.cli --output outputs/demo forge-demo` writes:

- `round_summary.json`
- `student_rollouts.jsonl`
- `verified_repairs.jsonl`
- `learnable_winner.json`
- `delta_spans.json`

These file names are stable and can be consumed by training notebooks or release
scripts.

## Security Boundary

The verifier is a research helper, not a sandbox. It uses a subprocess and
timeouts to isolate normal failures, but malicious code should be executed only
inside a container, VM, or hosted sandbox with network and filesystem controls.
