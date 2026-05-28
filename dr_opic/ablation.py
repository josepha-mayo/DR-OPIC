from __future__ import annotations

from math import sqrt
from typing import Sequence


Vector = Sequence[float]
Matrix = Sequence[Sequence[float]]


def normalize(vector: Vector, eps: float = 1e-12) -> list[float]:
    norm = sqrt(sum(x * x for x in vector))
    if norm <= eps:
        raise ValueError("cannot normalize zero vector")
    return [x / norm for x in vector]


def refusal_direction(harmful_mean: Vector, harmless_mean: Vector, invert: bool = False) -> list[float]:
    if len(harmful_mean) != len(harmless_mean):
        raise ValueError("direction vectors must have the same dimension")
    raw = [h - g for h, g in zip(harmful_mean, harmless_mean)]
    if invert:
        raw = [-x for x in raw]
    return normalize(raw)


def project_out_vector(activation: Vector, direction: Vector, strength: float = 1.0) -> list[float]:
    v = normalize(direction)
    dot = sum(a * b for a, b in zip(activation, v))
    return [a - strength * dot * b for a, b in zip(activation, v)]


def column_repulsion(weight: Matrix, direction: Vector, strength: float = 1.0) -> list[list[float]]:
    """Apply W' = W - strength * outer(v, v^T W) for row-major W."""

    v = normalize(direction)
    if not weight:
        return []
    cols = len(weight[0])
    if len(v) != len(weight):
        raise ValueError("direction dimension must match weight rows")
    projection = [sum(v[i] * weight[i][j] for i in range(len(v))) for j in range(cols)]
    return [
        [weight[i][j] - strength * v[i] * projection[j] for j in range(cols)]
        for i in range(len(weight))
    ]


def torch_column_repulsion(weight, direction, strength: float = 1.0):
    """Torch implementation for real model weights."""

    import torch

    v = direction / torch.clamp(torch.linalg.vector_norm(direction), min=1e-12)
    if weight.shape[0] != v.shape[0]:
        raise ValueError("direction dimension must match weight rows")
    projection = v.reshape(1, -1) @ weight
    return weight - strength * (v.reshape(-1, 1) @ projection)
