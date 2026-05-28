# Contributing

DR-OPIC is kept intentionally small. Contributions should make the verifier loop,
artifact contract, training-objective helpers, or documentation more runnable.

## Rules

- Do not commit private datasets, model weights, PDFs, notebooks with embedded
  secrets, Kaggle outputs, or generated checkpoints.
- Do not add unrelated model-ablation research to this repo.
- Keep examples tiny, deterministic, and safe to execute.
- Add tests for new public functions and CLI commands.
- Preserve stable artifact names unless the release protocol is updated.

## Local Check

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
python -m dr_opic.cli forge-demo
```

## Pull Request Checklist

- tests pass
- docs explain how to run the changed feature
- no private data or binary model artifacts are tracked
- CLI output is JSON when practical
- verifier behavior is deterministic under timeout
