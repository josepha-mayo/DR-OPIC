# Local Frameworks Consolidated

The local `model_ablation` folder contained several custom systems. This repo
consolidates the reusable parts into a smaller public framework.

## DR-OPIC / SLM Forge

Student-first coding SLM training:

- rollout current student before teacher repair
- execute tests and collect observations
- weight tasks by ZPD learnability
- cluster failures by error signature
- request minimal repairs from a domain teacher or policy
- verify repaired candidates
- train on self-success, repair, and delta spans

Implemented in `dr_opic.forge`, `dr_opic.maths`, `dr_opic.selectors`, and
`dr_opic.delta`.

## Verifier-Gated Agentic Harness

The Kaggle notebooks repeatedly used:

- code extraction from markdown
- static shape checks
- public test execution
- candidate sampling
- selected@K and coverage@K tracking
- one or more repair rounds
- CSV/JSON proof artifacts

Implemented in `dr_opic.verifier`, `dr_opic.forge`, and `dr_opic.replay`.

## ORPO / Verified Preference Lane

The preference runs showed that generic pairs are risky for code. The framework
keeps preference helpers small and expects:

- passing candidate beats failing candidate
- length-normalized log probabilities
- base-relative margins
- reward-based margins
- no preference training before repair probes are stable

Implemented in `dr_opic.preference`.

## Replay Certifier

Replay certification re-runs stored candidates through deterministic verifiers
and records whether postprocessing or repair truly improves pass counts.

Implemented in `dr_opic.replay`.

## Dataset Auditors

The dataset specs emphasized strict JSONL schemas, no chain-of-thought tags, no
placeholder code, prompt diversity, verifier artifacts, and train/eval separation.

Implemented in `dr_opic.datasets`.

## Directional Ablation / Repulsion

The ablation work identified refusal or behavior directions from contrastive
activations and projected them out of activations or weights.

Implemented as math-only helpers in `dr_opic.ablation`. This repo does not ship
weights or refusal-removal datasets.

## Release Discipline

A release needs proof artifacts, not just a checkpoint:

- base vs adapter metrics
- coverage@K, selected@K, selector gap, repair@1
- error signature counts
- delta spans
- dataset audit
- contamination report
- safety/abstention report
- compression retention report
