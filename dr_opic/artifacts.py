from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable


def to_plain(value: Any) -> Any:
    """Convert framework objects into JSON-safe containers."""

    if is_dataclass(value):
        return to_plain(asdict(value))
    if isinstance(value, dict):
        return {str(k): to_plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_plain(v) for v in value]
    return value


def dumps_json(value: Any, *, indent: int | None = 2) -> str:
    return json.dumps(to_plain(value), indent=indent, ensure_ascii=False)


def write_json(path: str | Path, value: Any) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(dumps_json(value) + "\n", encoding="utf-8")
    return target


def write_jsonl(path: str | Path, rows: Iterable[Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(to_plain(row), ensure_ascii=False) + "\n")
    return target
