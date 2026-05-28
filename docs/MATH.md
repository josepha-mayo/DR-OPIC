# DR-OPIC Math Notes

## Domain Routing

Let `D = {D_1, ..., D_K}` be bounded coding domains. A router maps a task to a
specialist or abstains:

```text
r_phi(x) in {1, ..., K, abstain}
pi_k(y | x, r_phi(x) = k)
```

The target is cost-adjusted reliable competence:

```text
eta_k = (Q_k - Q_base,k) / (C_train,k + rho C_infer,k + gamma C_review,k)
```

Promotion also requires safety and abstention constraints:

```text
unsafe_accept_k <= epsilon_k
ood_accept_k <= alpha_k
selective_risk_k(theta, tau) <= delta_k
```

## Verifier Reward

For a candidate `y` on task `x`:

```text
r(y, x) =
  1.00 final_pass
+ 0.25 public_test_fraction
+ 0.10 syntax_ok
+ 0.05 import_ok
- 0.05 repeated_token_penalty
- 0.05 invalid_format_penalty
- 0.02 normalized_length_penalty
- 0.05 unsafe_api_penalty
```

Final pass dominates. Partial rewards rank failures for repair and RL; they do
not replace held-out tests.

## ZPD Weighting

For `s` passes in `K` samples:

```text
p_tilde = (s + 0.5) / (K + 1)
w_zpd = 4 p_tilde (1 - p_tilde)
```

This peaks near tasks the model sometimes solves and sometimes fails. It stays
nonzero for small-K all-fail groups, which keeps near-impossible tasks available
for decomposition and repair instead of deleting them.

## Learnable Winner

For a verified candidate `c` and failed student attempt `s`:

```text
Score(c; s) =
  lambda_v 1[verifier(c) = pass]
+ lambda_f fuzz_pass_fraction(c)
+ lambda_l log pi_student(c | x) / |c|
- lambda_e normalized_edit_distance(c, s)
- lambda_c complexity(c)
- lambda_d rare_dependency_count(c)
```

The target is not the prettiest teacher answer. It is the passing answer closest
to the student's reachable failure state.

## Composite Objective

```text
L_DR-OPIC =
  L_self
+ lambda_r L_repair
+ lambda_delta L_delta
+ lambda_ood L_ood
+ lambda_cal L_cal
+ lambda_pref L_pref
+ lambda_rl L_RLVR
+ lambda_comp R_comp
```

Practical early runs should start with:

```text
L = L_self + L_repair + 0.3 L_delta
```

Add verified preference or RLVR only after the repair probe improves.

## Self Training

For a rollout group with rewards `{r_i}`:

```text
A_i = (r_i - mean(r)) / (std(r) + eps)
L_self = - w_task sum_i max(A_i, 0) log pi_theta(y_i | x)
```

This is advantage-weighted behavior cloning over the student's own distribution.

## Repair Training

```text
c_repair = format(task=x, failed_code=y_fail, observation=o_verifier)
L_repair = - w_task log pi_theta(y_fix | c_repair)
```

Loss should be applied only to the assistant correction, not to the task, failed
code, or verifier observation.

## Delta-Span Subtraction

Align failed code `y-` and fixed code `y+`.

```text
D+ = added/replaced tokens in the verified fix
D- = removed/replaced tokens in the failed answer

L_delta =
  - w_task sum_{t in D+} log pi_theta(y+_t | prefix+_t)
  + lambda_neg w_task sum_{t in D-} relu(
      log pi_theta(y-_t | prefix-_t)
    - log pi_ref(y-_t | prefix-_t)
    - margin
    )
```

This increases corrected spans and subtracts only the wrong local spans, avoiding
whole-program punishment when most code is shared and correct.

## Verified Preference

Use preference only when pairs are execution-grounded:

```text
ell_theta(y | c) = log pi_theta(y | c) / max(1, tokens(y))
Delta =
  [ell_theta(y+) - ell_ref(y+)]
- [ell_theta(y-) - ell_ref(y-)]
- margin(r+, r-)

L_vDPO = - log sigmoid(beta Delta)
```

## RLVR

For verifiable rewards and an old policy:

```text
rho_i = pi_theta(y_i | x) / pi_old(y_i | x)
L_RLVR = - E_i min(rho_i A_i, clip(rho_i, 1-eps, 1+eps) A_i)
```

The reward must come from execution tests, fuzzing, static checks, type checks,
linters, safety gates, and abstention checks.

## Test-Time Scaling

Report empirical metrics:

```text
coverage@K = tasks with at least one passing sample / tasks
selected@K = tasks where selector chose a passing sample / tasks
selector_gap = coverage@K - selected@K
repair@1 = tasks fixed by one repair after K failures / tasks
```

Do not rely on the IID `1 - (1 - p)^K` formula except as intuition.

## Repulsion Orthogonalization

For directional behavior editing, with normalized direction `v` and weight `W`:

```text
W' = W - strength * outer(v, v^T W)
```

This is a column-space projection/repulsion helper. It should be treated as a
research utility and evaluated with safety and retention gates.
