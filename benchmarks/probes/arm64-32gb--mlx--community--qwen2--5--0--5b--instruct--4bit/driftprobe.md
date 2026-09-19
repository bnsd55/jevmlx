# Batched drift probe (W5c-4)

model: `mlx-community/Qwen2.5-0.5B-Instruct-4bit` — schema `code_security` (28 rows), cache dtype mlx.core.float16

## (a) Repeatability

5 runs at 112 rows (decide_many slots shape): max run-to-run diff **0.000000** — deterministic

## (b/c/d) Width matrix

| slot_mode | rows | d_raw med/p99/max | d_gap med/p99/max | d_logp med/p99/max | winners≠ | 2-group d_gap | 4-group d_gap | perm d_gap |
|---|---|---|---|---|---|---|---|---|
| broadcast | 1 | 0.0000/0.0000/0.0000 | 0.0000/0.0000/0.0000 | 0.0000/0.0000/0.0000 | 0 | - | - | - |
| slots | 1 | 0.0000/0.0000/0.0000 | 0.0000/0.0000/0.0000 | 0.0000/0.0000/0.0000 | 0 | - | - | - |
| broadcast | 2 | 0.0292/0.0558/0.0558 | 0.0177/0.0344/0.0344 | 0.0178/0.0350/0.0350 | 0 | - | - | 0.0344 |
| slots | 2 | 0.0292/0.0558/0.0558 | 0.0177/0.0344/0.0344 | 0.0178/0.0350/0.0350 | 0 | - | - | 0.0344 |
| broadcast | 4 | 0.0230/0.0558/0.0558 | 0.0155/0.0344/0.0344 | 0.0204/0.0350/0.0350 | 0 | - | - | 0.0344 |
| slots | 4 | 0.0230/0.0558/0.0558 | 0.0155/0.0344/0.0344 | 0.0204/0.0350/0.0350 | 0 | - | - | 0.0344 |
| broadcast | 8 | 0.0253/0.0683/0.0683 | 0.0140/0.0364/0.0364 | 0.0117/0.0421/0.0421 | 0 | - | - | 0.0364 |
| slots | 8 | 0.0253/0.0683/0.0683 | 0.0140/0.0364/0.0364 | 0.0117/0.0421/0.0421 | 0 | - | - | 0.0364 |
| broadcast | 28 | 0.0203/0.1242/0.1242 | 0.0073/0.0600/0.0600 | 0.0106/0.0425/0.0425 | 0 | - | - | 0.0600 |
| slots | 28 | 0.0203/0.1242/0.1242 | 0.0073/0.0600/0.0600 | 0.0106/0.0425/0.0425 | 0 | - | - | 0.0600 |
| broadcast | 56 | 0.0130/0.0992/0.0992 | 0.0097/0.0481/0.0481 | 0.0075/0.0342/0.0342 | 0 | - | - | 0.0481 |
| slots | 56 | 0.0130/0.0992/0.0992 | 0.0097/0.0481/0.0481 | 0.0075/0.0342/0.0342 | 0 | - | - | 0.0481 |
| broadcast | 112 | 0.0162/0.1219/0.1219 | 0.0093/0.0636/0.0636 | 0.0085/0.0450/0.0450 | 0 | 0.0636 | 0.0636 | 0.0636 |
| slots | 112 | 0.0162/0.1219/0.1219 | 0.0093/0.0636/0.0636 | 0.0085/0.0450/0.0450 | 0 | 0.0636 | 0.0636 | 0.0636 |

## (e) Mask/cache-object control

at 16 rows: d_raw max 0.1242, d_gap max 0.0600, winners≠ 0
> mlx_lm derives the mask from the cache object; the decide_many path differs from batch=1 ONLY by the BatchKVCache(left_padding) object. This control measures the drift of that exact shape difference at a small width (the full curve is in the width matrix).

## (f) Cache control

prefill deterministic: True
merged batched cache per-slot values identical: True
broadcast slots: d_raw max 0.1242, d_gap max 0.0600, winners≠ 0
independent slots: d_raw max 0.1242, d_gap max 0.0600, winners≠ 0
> cache VALUES were compared EXACTLY (elementwise, shapes, offsets) before scoring — and per-slot inside the merged batched cache: identical => the drift comes from the forward shape, not the cache contents.

## Reference: check_batched_parity (the 0.125 source)

max_raw_row_drift=0.125, max_final_drift=0.03820767966379712, winners_identical=True

## (g) Near-tie corpus

3 rows with batch-1 margin < 0.3 (min margin 0.0523)
| rows | d_raw max | d_gap max | d_logp max | winners≠ |
|---|---|---|---|---|
| 16 | 0.0214 | 0.0101 | 0.0055 | 0 |
| 56 | 0.0176 | 0.0127 | 0.0069 | 0 |
| 112 | 0.0226 | 0.0117 | 0.0064 | 0 |
