# W5c-10: Per-Layer Bisect of the Batched Gap Drift

## Summary

The batched gap drift (0.0625 nats at M≥16, PR #66 driftprobe) was bisected
layer-by-layer on the Qwen2.5-0.5B-Instruct-4bit model to localize its source.
The fp32 decision-token head (PR #70) did not remove it, proving the drift
enters before the LM head. This probe isolates WHERE in the transformer body.

## Reproduce

```bash
# From the repo root:
.venv/bin/python benchmarks/layer_bisect.py
# Override model:
.venv/bin/python benchmarks/layer_bisect.py --model <mlx-model-id>
```

Writes `benchmarks/probes/<arch>--<model>/layer_bisect.json` and prints seven
markdown tables to stdout.

## Facts

1. **Deterministic.** Repeated runs of the same N-row batch produce
   identical logits (driftprobe §a repeatability).

2. **fp16 body (native quantized):** gap drift 0.0625 nats plateau from 16
   rows onward, exact fp16 steps. The drift appears at layer 0 attention
   output, grows smoothly through the stack, and reaches final_norm_out
   0.625 (M=16) / 0.531 (M=112).

3. **fp32 body (fully dequantized, ~2 GB):** the plateau does NOT collapse.
   At M=16 the final norm is clean (0.0013), but at M=112 it reaches 0.498.
   The width curve shows discrete **steps** at M=32 (final 0.06), M=84
   (0.32), M=112 (0.50) — kernel-tiling-boundary jumps, not a ramp.

4. **Cache, grouping, permutation invariance.** Cache tensors are identical
   per slot; broadcast vs per-context cache produces identical results;
   grouping and row permutation are invariant (driftprobe §b/c/d).

5. **fp32 decision head does not remove it** (PR #70): d_gap 0.064 at M=112
   with the fp32 head vs 0.0625 quantized — the drift originates upstream
   of the LM head.

6. **Attention block and MLP are shape-invariant when recomputed on
   identical inputs** (1e-6 relative at M=16/32/112 for all four sub-steps:
   q/k/v projections, RoPE, SDPA, o_proj). In-situ == recomputed at the
   first diverging op (layer 2 mlp_out, diff 0.000000). The divergence vs
   the M=1 reference appears at layer 2 MLP: input relative diff 1e-6,
   output relative diff 1e-3 — amplification inside the MLP of
   Qwen2.5-0.5B where activations reach 3.6×10².

7. **Not a cache, mask, position, or indexing bug in jevmlx.** Decision-token
   id per row matches M=1 vs M=112 (True). KV cache dtype is fp16 (prefill
   ran on the fp16 model before dequantization); SDPA delegates to
   `mx.fast.scaled_dot_product_attention`; mask shape is (M,1,9,1453) with
   uniform cache offset 1453 across all rows.

## Conclusion

Shape-dependent numerics of the MLX forward at ≥32 rows per pass, amplified
by the model's activation outliers; outside jevmlx's control; mitigated by
the fail-closed parity gate and the measured drift envelope on the near-tie
band; a per-pass row cap for probability-sensitive use is decided by the 7B
parity run on the M5.

## Tables

Seven tables are printed by the probe and stored in `layer_bisect.json`:

- **Table 1:** drift by layer (fp16 body) — per-layer max abs diff at
  decision positions, M=16 and M=112 vs M=1 reference.
- **Table 2:** drift by layer (fp32 body) — same, with the fully
  dequantized fp32 model.
- **Table 3:** fp32 width curve — final_norm_out + L03 mlp_out at
  M=16,24,32,48,56,64,84,112,128.
- **Table 4:** fp32 relative diff — max_abs_diff / max|activation| per
  layer for M=16 and M=112.
- **Table 5:** GEMM isolation — full-batch vs row-by-row MLP at L03,
  jump M=32 (diff 1.9e-5, relative 2.5e-7).
- **Table 6:** attention isolation — full-batch vs row-by-row attention
  at L03, 4-way split (qkv proj, RoPE, SDPA, o_proj), M=16/32/112.
- **Table 7:** cache dtype + mask + SDPA path info at M=16/32/112.
- **Table 8:** in-situ vs recomputed op isolation (L00-L05) — first
  diverging op, recomputed-vs-in-situ comparison, decision-token match.
