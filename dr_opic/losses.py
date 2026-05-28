from __future__ import annotations


def require_torch():
    try:
        import torch
        import torch.nn.functional as F
    except ImportError as exc:
        raise ImportError("install dr-opic[torch] to use tensor losses") from exc
    return torch, F


def advantage_weighted_sft_loss(token_logprobs, advantages, mask=None):
    """-sum(max(A,0) * log p) over response tokens."""

    torch, _ = require_torch()
    weights = torch.clamp(advantages, min=0.0)
    while weights.ndim < token_logprobs.ndim:
        weights = weights.unsqueeze(-1)
    values = -weights * token_logprobs
    if mask is not None:
        values = values * mask
        return values.sum() / mask.sum().clamp_min(1.0)
    return values.mean()


def delta_span_loss(fixed_logprobs, failed_logprobs, ref_failed_logprobs, pos_mask, neg_mask, margin: float = 0.0, lambda_neg: float = 0.3):
    """Reward fixed spans and subtract failed spans that exceed reference margin."""

    torch, _ = require_torch()
    pos = -(fixed_logprobs * pos_mask).sum() / pos_mask.sum().clamp_min(1.0)
    excess = failed_logprobs - ref_failed_logprobs - margin
    neg = (torch.relu(excess) * neg_mask).sum() / neg_mask.sum().clamp_min(1.0)
    return pos + lambda_neg * neg


def dpo_loss(chosen_logp, rejected_logp, ref_chosen_logp, ref_rejected_logp, margin=0.0, beta: float = 0.1):
    _, F = require_torch()
    logits = (chosen_logp - ref_chosen_logp) - (rejected_logp - ref_rejected_logp) - margin
    return -F.logsigmoid(beta * logits).mean()


def clipped_rlvr_loss(new_logp, old_logp, advantages, clip_eps: float = 0.2):
    torch, _ = require_torch()
    ratio = torch.exp(new_logp - old_logp)
    clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
    return -torch.minimum(ratio * advantages, clipped * advantages).mean()
