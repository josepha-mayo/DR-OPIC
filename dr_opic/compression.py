from __future__ import annotations

from dataclasses import dataclass

from .maths import weight_memory_bytes


@dataclass(frozen=True)
class CompressionEstimate:
    params: float
    fp16_gb: float
    q8_gb: float
    q4_gb: float
    train_flops_per_token: float
    infer_flops_per_token: float


def estimate_dense_model(params: float) -> CompressionEstimate:
    gb = 1024 ** 3
    return CompressionEstimate(
        params=params,
        fp16_gb=weight_memory_bytes(params, 2.0) / gb,
        q8_gb=weight_memory_bytes(params, 1.0) / gb,
        q4_gb=weight_memory_bytes(params, 0.5) / gb,
        train_flops_per_token=6.0 * params,
        infer_flops_per_token=2.0 * params,
    )


def retention_gate(before: float, after: float, max_drop: float = 0.01) -> bool:
    return after + max_drop >= before
