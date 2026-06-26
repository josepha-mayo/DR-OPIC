from __future__ import annotations

import math


def length_normalized_logprob(logprob: float, tokens: int) -> float:
    return logprob / max(1, tokens)


def verified_preference_margin(chosen_reward: float, rejected_reward: float, scale: float = 1.0) -> float:
    return max(0.0, chosen_reward - rejected_reward) * scale


def dpo_logit(
    chosen_logp: float,
    rejected_logp: float,
    ref_chosen_logp: float,
    ref_rejected_logp: float,
    margin: float = 0.0,
) -> float:
    return (chosen_logp - ref_chosen_logp) - (rejected_logp - ref_rejected_logp) - margin


def dpo_loss_scalar(logit: float, beta: float = 0.1) -> float:
    x = beta * logit
    if x >= 0:
        return math.log1p(math.exp(-x))
    return -x + math.log1p(math.exp(x))


def orpo_odds_ratio(chosen_logp: float, rejected_logp: float, eps: float = 1e-6) -> float:
    pc = min(1.0 - eps, max(eps, math.exp(min(chosen_logp, 88.0))))
    pr = min(1.0 - eps, max(eps, math.exp(min(rejected_logp, 88.0))))
    return math.log(pc / (1.0 - pc)) - math.log(pr / (1.0 - pr))
