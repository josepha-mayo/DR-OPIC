# Release Protocol

## 1. Define the Domain

- exact task distribution
- allowed languages and libraries
- verifier type and reliability
- OOD and unsafe request policy
- escalation/abstention behavior

## 2. Build Splits

- train/dev/eval split by dedupe key
- benchmark task IDs excluded from training
- source/license metadata
- secret and unsafe-code scan
- prompt/response near-duplicate audit

## 3. Run Student First

For each task:

- generate `K` student attempts
- run static checks and tests
- store observation, reward, latency, tokens, and collapse markers
- compute ZPD weight

## 4. Repair and Select

- route failures by domain and error signature
- repair from failed code plus verifier observation
- verify every repair
- select learnable winners by pass, edit distance, complexity, dependencies, and length
- store delta spans between failed and fixed code

## 5. Train in Phases

Phase A:

```text
L_self + L_repair
```

Phase B:

```text
L_self + L_repair + 0.3 L_delta
```

Phase C:

```text
add verified preference only if probe improves
```

Phase D:

```text
add RLVR only where verifiers are robust
```

## 6. Gate Promotion

Required metrics:

- greedy@1
- coverage@K
- selected@K
- selector gap
- repair@1
- hard-subset pass rate
- malformed output rate
- repeated-token collapse rate
- OOD false accept rate
- unsafe compliance rate
- latency and memory

Promote only if selected@K or repair@1 improves without meaningful regression in
greedy, safety, abstention, or formatting.

## 7. Compress After Behavior Improves

- merge adapter
- quantize
- replay retention suite
- run compression recovery rows only if needed
- publish proof card with exact base, adapter, quantization, and metrics
