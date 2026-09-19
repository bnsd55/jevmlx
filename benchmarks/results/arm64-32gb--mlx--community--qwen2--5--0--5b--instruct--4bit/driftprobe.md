# Batched drift probe (W5c-4)

model: `mlx-community/Qwen2.5-0.5B-Instruct-4bit` — schema `code_security` (28 rows), cache dtype mlx.core.float16

## (a) Repeatability

5 runs at 112 rows (decide_many slots shape): max run-to-run diff **0.000000** — deterministic

## (b/c/d) Width matrix

| slot_mode | rows | d_raw med/p99/max | d_gap med/p99/max | d_logp med/p99/max | winners≠ | 2-group d_gap | 4-group d_gap | perm d_gap |
|---|---|---|---|---|---|---|---|---|
| broadcast | 1 | 0.0000/0.0000/0.0000 | 0.0000/0.0000/0.0000 | 0.0000/0.0000/0.0000 | 0 | - | - | - |
| slots | 1 | 0.0000/0.0000/0.0000 | 0.0000/0.0000/0.0000 | 0.0000/0.0000/0.0000 | 0 | - | - | - |
| broadcast | 2 | 0.0352/0.0547/0.0547 | 0.0273/0.0391/0.0391 | 0.0231/0.0381/0.0381 | 0 | - | - | 0.0391 |
| slots | 2 | 0.0352/0.0547/0.0547 | 0.0273/0.0391/0.0391 | 0.0231/0.0381/0.0381 | 0 | - | - | 0.0391 |
| broadcast | 4 | 0.0312/0.0547/0.0547 | 0.0234/0.0391/0.0391 | 0.0268/0.0423/0.0423 | 0 | - | - | 0.0391 |
| slots | 4 | 0.0312/0.0547/0.0547 | 0.0234/0.0391/0.0391 | 0.0268/0.0423/0.0423 | 0 | - | - | 0.0391 |
| broadcast | 8 | 0.0312/0.0781/0.0781 | 0.0156/0.0469/0.0469 | 0.0128/0.0333/0.0333 | 0 | - | - | 0.0469 |
| slots | 8 | 0.0312/0.0781/0.0781 | 0.0156/0.0469/0.0469 | 0.0128/0.0333/0.0333 | 0 | - | - | 0.0469 |
| broadcast | 16 | 0.0195/0.1250/0.1250 | 0.0117/0.0625/0.0625 | 0.0118/0.0443/0.0443 | 0 | - | - | 0.0625 |
| slots | 16 | 0.0195/0.1250/0.1250 | 0.0117/0.0625/0.0625 | 0.0118/0.0443/0.0443 | 0 | - | - | 0.0625 |
| broadcast | 28 | 0.0195/0.1250/0.1250 | 0.0117/0.0625/0.0625 | 0.0122/0.0443/0.0443 | 0 | - | - | 0.0625 |
| slots | 28 | 0.0195/0.1250/0.1250 | 0.0117/0.0625/0.0625 | 0.0122/0.0443/0.0443 | 0 | - | - | 0.0625 |
| broadcast | 32 | 0.0156/0.1250/0.1250 | 0.0117/0.0625/0.0625 | 0.0129/0.0443/0.0443 | 0 | - | - | 0.0625 |
| slots | 32 | 0.0156/0.1250/0.1250 | 0.0117/0.0625/0.0625 | 0.0129/0.0443/0.0443 | 0 | - | - | 0.0625 |
| broadcast | 56 | 0.0156/0.1094/0.1094 | 0.0156/0.0625/0.0625 | 0.0089/0.0443/0.0443 | 0 | - | - | 0.0625 |
| slots | 56 | 0.0156/0.1094/0.1094 | 0.0156/0.0625/0.0625 | 0.0089/0.0443/0.0443 | 0 | - | - | 0.0625 |
| broadcast | 64 | 0.0156/0.1094/0.1094 | 0.0156/0.0625/0.0625 | 0.0092/0.0443/0.0443 | 0 | - | - | 0.0625 |
| slots | 64 | 0.0156/0.1094/0.1094 | 0.0156/0.0625/0.0625 | 0.0092/0.0443/0.0443 | 0 | - | - | 0.0625 |
| broadcast | 84 | 0.0156/0.1250/0.1250 | 0.0117/0.0625/0.0625 | 0.0117/0.0443/0.0443 | 0 | - | - | 0.0625 |
| slots | 84 | 0.0156/0.1250/0.1250 | 0.0117/0.0625/0.0625 | 0.0117/0.0443/0.0443 | 0 | - | - | 0.0625 |
| broadcast | 112 | 0.0156/0.1250/0.1250 | 0.0117/0.0625/0.0625 | 0.0117/0.0443/0.0443 | 0 | 0.0625 | 0.0625 | 0.0625 |
| slots | 112 | 0.0156/0.1250/0.1250 | 0.0117/0.0625/0.0625 | 0.0117/0.0443/0.0443 | 0 | 0.0625 | 0.0625 | 0.0625 |
| broadcast | 128 | 0.0156/0.1250/0.1250 | 0.0117/0.0625/0.0625 | 0.0117/0.0443/0.0443 | 0 | - | - | 0.0625 |
| slots | 128 | 0.0156/0.1250/0.1250 | 0.0117/0.0625/0.0625 | 0.0117/0.0443/0.0443 | 0 | - | - | 0.0625 |

## (e) Mask/cache-object control

at 16 rows: d_raw max 0.1250, d_gap max 0.0625, winners≠ 0
> mlx_lm derives the mask from the cache object; the decide_many path differs from batch=1 ONLY by the BatchKVCache(left_padding) object. This control measures the drift of that exact shape difference at a small width (the full curve is in the width matrix).

## (f) Cache control

prefill deterministic: True
merged batched cache per-slot values identical: True
broadcast slots: d_raw max 0.1250, d_gap max 0.0625, winners≠ 0
independent slots: d_raw max 0.1250, d_gap max 0.0625, winners≠ 0
> cache VALUES were compared EXACTLY (elementwise, shapes, offsets) before scoring — and per-slot inside the merged batched cache: identical => the drift comes from the forward shape, not the cache contents.

## Reference: check_batched_parity (the 0.125 source)

max_raw_row_drift=0.125, max_final_drift=0.04577926327619686, winners_identical=True

## (g) Near-tie corpus

3 rows with batch-1 margin < 0.3 (min margin 0.0625)
| rows | d_raw max | d_gap max | d_logp max | winners≠ |
|---|---|---|---|---|
| 16 | 0.0312 | 0.0156 | 0.0086 | 0 |
| 56 | 0.0312 | 0.0156 | 0.0086 | 0 |
| 112 | 0.0312 | 0.0156 | 0.0086 | 0 |
