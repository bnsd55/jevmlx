"""W5c-10: per-layer bisect of the batched gap drift.

The fp32 decision head (W5c-8, PR #70) did NOT collapse the plateau
(d_gap 0.064 at M=112 vs 0.0625 quantized). So the drift enters BEFORE
the LM head: the final hidden state itself differs by batch shape. This
bisect localizes WHERE in the transformer body the drift first appears
and how it grows.

For M = 1, 16, 112 on the SAME context/rows (the driftprobe schema),
this captures at each transformer layer's decision positions:

- residual_in   : x  (input to the block)
- norm_out      : input_layernorm(x)
- attn_out      : self_attn(input_layernorm(x), mask, cache)  (pre-residual)
- mlp_out       : mlp(post_attention_layernorm(h))            (pre-residual)
- final_norm_out: the inner model's final RMSNorm output       (last step)

Per-layer metric: max abs diff at decision positions vs the M=1 reference.

TWO body precisions:
(1) fp16 body — the native quantized model (the production path).
(2) fp32 body — every quantized linear dequantized to fp32
    (mx.dequantize -> float32 copy of the whole model, ~2 GB on the 0.5B).

If the fp32 body collapses the plateau, the cause is quantized-kernel
shape rounding in the body; if not, it is attention/mask/position
dependent on M.

Also records: which layer first shows a non-zero diff, and whether the
drift grows smoothly or jumps (per-layer delta).

Run (model-loading — MUST go through the shared slot lock):

    ~/git/jev-on-a-laptop/.agent-mail/slowtest.sh .venv/bin/python benchmarks/layer_bisect.py

Writes layer_bisect.json + prints two markdown tables. No conclusions.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import mlx.core as mx

from jevmlx.engine import Engine, _build_schema_rows, _prefill, load_engine
from jevmlx.parity import _case_context, _make_schema, bundled_preset_specs
from jevmlx.timing import Ledger

WIDTHS = (1, 16, 112)


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------


def _set_submodule(root, dotted_name: str, new_mod) -> None:
    """Set a submodule by dotted path (e.g. 'model.layers.0.mlp.up_proj').

    Numeric path segments index into lists (``layers`` is a list of
    TransformerBlock). An empty name means root itself (skip — can't
    replace root)."""
    if not dotted_name:
        return
    parts = dotted_name.split(".")
    parent = root
    for p in parts[:-1]:
        parent = _get_child(parent, p)
    last_parent = parent
    last_key = parts[-1]
    if isinstance(last_parent, list):
        last_parent[int(last_key)] = new_mod
    else:
        setattr(last_parent, last_key, new_mod)


def _get_child(parent, key: str):
    """Get a child by dotted-key segment: int index for lists, attr otherwise."""
    if isinstance(parent, list):
        return parent[int(key)]
    return getattr(parent, key)


def _dequantize_model_fp32(model) -> None:
    """Dequantize EVERY quantized linear/embedding in the model to a true
    fp32 copy.

    Mutates the model in place: each QuantizedLinear / QuantizedEmbedding's
    weight/scales/biases are replaced by a single dequantized float32
    weight, so the forward runs in genuine fp32 (not an astype(float32)
    wrapper around the quantized kernel). ~2 GB on the 0.5B.
    """
    from mlx.nn import Embedding, Linear
    from mlx.nn.layers.quantized import QuantizedEmbedding, QuantizedLinear

    for name, mod in list(model.named_modules()):
        if isinstance(mod, QuantizedLinear):
            deq = mx.dequantize(
                mod.weight,
                scales=mod.scales,
                biases=mod.biases,
                group_size=mod.group_size,
                bits=mod.bits,
            ).astype(mx.float32)
            lin = Linear(
                input_dims=deq.shape[1],
                output_dims=deq.shape[0],
                bias=False,
            )
            lin.weight = deq
            _set_submodule(model, name, lin)
        elif isinstance(mod, QuantizedEmbedding):
            deq = mx.dequantize(
                mod.weight,
                scales=mod.scales,
                biases=mod.biases,
                group_size=getattr(mod, "group_size", 64),
                bits=getattr(mod, "bits", 4),
            ).astype(mx.float32)
            new_emb = Embedding(num_embeddings=deq.shape[0], dims=deq.shape[1])
            new_emb.weight = deq
            _set_submodule(model, name, new_emb)


def _rows_for_width(built: dict, n: int) -> tuple[list, list]:
    """The schema's rows replicated/truncated to EXACTLY n rows (driftprobe
    convention: row j -> schema row j % n_schema_rows)."""
    k = math.ceil(n / max(1, len(built["rows"])))
    return (built["rows"] * k)[:n], (built["row_decision"] * k)[:n]


# ---------------------------------------------------------------------------
# The per-layer forward with capture
# ---------------------------------------------------------------------------


def _create_attention_mask(h, cache):
    """mlx_lm's create_attention_mask, resolved at call time (deferred import
    keeps the module importable without mlx on Linux CI)."""
    from mlx_lm.models.base import create_attention_mask

    return create_attention_mask(h, cache)


def _run_layer_captures(
    model,
    padded: mx.array,
    cache: list,
    positions: mx.array,
) -> dict[str, list]:
    """Run the transformer BODY manually, layer by layer, capturing
    intermediates at the decision positions.

    Returns a dict keyed by capture name; each value is a list of per-layer
    arrays (or one for final_norm_out). The arrays are the decision-position
    hidden vectors: shape (chunk_len, hidden_dim).

    Captures per layer:
      residual_in   = x
      norm_out      = input_layernorm(x)
      attn_out      = self_attn(norm_out, mask, cache_l)   (pre-residual)
      mlp_out       = mlp(post_attention_layernorm(h))     (pre-residual)
    Plus:
      final_norm_out = model.norm(h)  (after all layers)
    """
    m = model.model
    h = m.embed_tokens(padded)
    if cache is None:
        cache = [None] * len(m.layers)
    mask = _create_attention_mask(h, cache[0])

    chunk_len = padded.shape[0]
    pos = positions  # (chunk_len,) int positions into the width axis

    caps: dict[str, list] = {
        "residual_in": [],
        "norm_out": [],
        "attn_out": [],
        "mlp_out": [],
    }
    for _li, (layer, c) in enumerate(zip(m.layers, cache, strict=True)):
        x = h
        caps["residual_in"].append(x[mx.arange(chunk_len), pos])
        norm_out = layer.input_layernorm(x)
        caps["norm_out"].append(norm_out[mx.arange(chunk_len), pos])
        r = layer.self_attn(norm_out, mask, c)
        caps["attn_out"].append(r[mx.arange(chunk_len), pos])
        h = x + r
        mlp_out = layer.mlp(layer.post_attention_layernorm(h))
        caps["mlp_out"].append(mlp_out[mx.arange(chunk_len), pos])
        h = h + mlp_out

    caps["final_norm_out"] = [m.norm(h)[mx.arange(chunk_len), pos]]
    # Materialize everything (we held lazy graphs through the whole forward).
    for k in caps:
        caps[k] = [mx.eval(a) or a for a in caps[k]]
    return caps


def _max_abs_diff(ref: mx.array, got: mx.array) -> float:
    """Max abs diff over the decision-position hidden vectors (flattened)."""
    return float(mx.max(mx.abs(ref.astype(mx.float32) - got.astype(mx.float32))).item())


def _capture_reference_batch1(
    model,
    built: dict,
    pf_cache_list: list,
    pad_id: int,
) -> tuple[dict[str, list], list[int]]:
    """The canonical batch=1 reference: ALL schema rows, ONE ROW PER FORWARD.

    Captures per-layer intermediates for each schema row individually (the
    exact shape the near-tie rescore trusts). Returns (caps, schema_keys)
    where caps[cap][schema_row] is a (1, hidden) array and schema_keys is
    [0, 1, ..., n_schema_rows-1]."""
    from jevmlx.engine import _broadcast_cache, _eval_cache_state

    n_schema = len(built["rows"])
    # Accumulate per-schema-row captures into lists of (1, hidden) arrays.
    per_row_caps: list[dict[str, list]] = []
    for ridx in range(n_schema):
        row = built["rows"][ridx]
        dec = built["row_decision"][ridx]
        padded = mx.array([row], dtype=mx.int32)  # (1, width)
        positions = mx.array([dec[0]], dtype=mx.int32)  # (1,)
        b_cache = _broadcast_cache(pf_cache_list, 1)
        _eval_cache_state(b_cache)
        caps = _run_layer_captures(model, padded, b_cache, positions)
        per_row_caps.append(caps)
    # Transpose: caps[cap][layer] -> list over schema rows of (1, hidden).
    names = ["residual_in", "norm_out", "attn_out", "mlp_out", "final_norm_out"]
    merged: dict[str, list] = {n: [] for n in names}
    for name in names:
        n_caps = len(per_row_caps[0][name])
        for li in range(n_caps):
            merged[name].append(mx.concatenate([prc[name][li] for prc in per_row_caps], axis=0))
    return merged, list(range(n_schema))


def _capture_at_width(
    model,
    built: dict,
    pf_cache_list: list,
    n: int,
    pad_id: int,
) -> tuple[dict[str, list], list[int]]:
    """Build the n-row padded batch (broadcast single cache) and capture
    per-layer intermediates at decision positions. Returns (caps, schema_keys)
    where schema_keys[j] = j % n_schema_rows for mapping back to the reference."""
    rows_n, decisions_n = _rows_for_width(built, n)
    widths = [len(r) for r in rows_n]
    width = max(widths)
    padded = mx.array([r + [pad_id] * (width - len(r)) for r in rows_n], dtype=mx.int32)
    positions = mx.array([d[0] for d in decisions_n], dtype=mx.int32)

    from jevmlx.engine import _broadcast_cache, _eval_cache_state

    b_cache = _broadcast_cache(pf_cache_list, n)
    max_padding = max(width - w for w in widths) if widths else 0
    if max_padding > 0:
        for c in b_cache:
            if hasattr(c, "prepare"):
                c.prepare(lengths=widths, right_padding=[width - w for w in widths])
    _eval_cache_state(b_cache)
    caps = _run_layer_captures(model, padded, b_cache, positions)
    schema_keys = [j % len(built["rows"]) for j in range(n)]
    return caps, schema_keys


# ---------------------------------------------------------------------------
# The bisect
# ---------------------------------------------------------------------------


def _per_layer_diffs(
    ref_caps: dict[str, list],
    got_caps: dict[str, list],
    ref_keys: list[int],
    got_keys: list[int],
) -> dict[str, list[float]]:
    """For each capture name, per-layer max abs diff at decision positions,
    mapping got rows back to their schema-row reference."""
    names = ["residual_in", "norm_out", "attn_out", "mlp_out", "final_norm_out"]
    out: dict[str, list[float]] = {n: [] for n in names}
    for name in names:
        ref_arrs = ref_caps[name]
        got_arrs = got_caps[name]
        n_layers = len(got_arrs)
        for li in range(n_layers):
            # got_arrs[li] is (n_got, hidden); ref_arrs[li] is (n_ref, hidden).
            # Map each got row to its schema-row reference and take max abs diff.
            g = got_arrs[li].astype(mx.float32)
            r = ref_arrs[li].astype(mx.float32)
            diffs = []
            for j, sk in enumerate(got_keys):
                # find the reference row index for this schema key
                ri = ref_keys.index(sk)
                diffs.append(mx.max(mx.abs(g[j] - r[ri])).item())
            out[name].append(max(diffs))
    return out


def run_bisect(
    engine: Engine,
    built: dict,
    pf_cache_list: list,
    pad_id: int,
    *,
    label: str,
) -> dict[str, Any]:
    """Run the per-layer bisect at WIDTHS. Returns the per-layer diff table."""
    print(f"  [{label}] capturing M=1 reference (all schema rows, batch=1)...", flush=True)
    ref_caps, ref_keys = _capture_reference_batch1(engine.model, built, pf_cache_list, pad_id)

    tables = {}
    for n in WIDTHS:
        if n == 1:
            continue
        print(f"  [{label}] capturing M={n}...", flush=True)
        got_caps, got_keys = _capture_at_width(engine.model, built, pf_cache_list, n, pad_id)
        diffs = _per_layer_diffs(ref_caps, got_caps, ref_keys, got_keys)
        tables[f"M={n}"] = diffs

    return {"reference": "M=1", "tables": tables}


# ---------------------------------------------------------------------------
# W5c-10 follow-ups (fp32 body): width curve, relative diff, GEMM isolation
# ---------------------------------------------------------------------------

WIDTH_CURVE = (16, 24, 32, 48, 56, 64, 84, 112, 128)


def _capture_with_magnitude(
    model,
    built: dict,
    pf_cache_list: list,
    n: int,
    pad_id: int,
) -> tuple[dict[str, list], dict[str, list], list[int]]:
    """Like _capture_at_width but also returns max |activation| per layer.

    Returns (caps, magnitudes, schema_keys) where magnitudes[cap][li] is the
    max abs value over the decision-position hidden vectors for that layer.
    """
    from jevmlx.engine import _broadcast_cache, _eval_cache_state

    rows_n, decisions_n = _rows_for_width(built, n)
    widths = [len(r) for r in rows_n]
    width = max(widths)
    padded = mx.array([r + [pad_id] * (width - len(r)) for r in rows_n], dtype=mx.int32)
    positions = mx.array([d[0] for d in decisions_n], dtype=mx.int32)
    b_cache = _broadcast_cache(pf_cache_list, n)
    max_padding = max(width - w for w in widths) if widths else 0
    if max_padding > 0:
        for c in b_cache:
            if hasattr(c, "prepare"):
                c.prepare(lengths=widths, right_padding=[width - w for w in widths])
    _eval_cache_state(b_cache)
    caps = _run_layer_captures(model, padded, b_cache, positions)
    names = ["residual_in", "norm_out", "attn_out", "mlp_out", "final_norm_out"]
    mags: dict[str, list] = {n: [] for n in names}
    for name in names:
        for arr in caps[name]:
            mags[name].append(float(mx.max(mx.abs(arr.astype(mx.float32))).item()))
    schema_keys = [j % len(built["rows"]) for j in range(n)]
    return caps, mags, schema_keys


def followup_width_curve(
    model,
    built: dict,
    pf_cache_list: list,
    pad_id: int,
    ref_caps: dict[str, list],
    ref_keys: list[int],
) -> dict[str, Any]:
    """(1) Width curve on the fp32 body: final_norm_out + L03 mlp_out diff
    at M = 16,24,32,48,56,64,84,112,128 — find the exact jump M and whether
    it is a step or a ramp."""
    rows: list[dict] = []
    for n in WIDTH_CURVE:
        caps, _mags, got_keys = _capture_with_magnitude(model, built, pf_cache_list, n, pad_id)
        diffs = _per_layer_diffs(ref_caps, caps, ref_keys, got_keys)
        rows.append(
            {
                "M": n,
                "final_norm_out": diffs["final_norm_out"][0],
                "L03_mlp_out": diffs["mlp_out"][3],
            }
        )
        print(
            f"  M={n:3d}: final_norm_out={rows[-1]['final_norm_out']:.6f}  "
            f"L03_mlp_out={rows[-1]['L03_mlp_out']:.6f}",
            flush=True,
        )
    return {"widths": list(WIDTH_CURVE), "rows": rows}


def followup_relative_diff(
    model,
    built: dict,
    pf_cache_list: list,
    pad_id: int,
    ref_caps: dict[str, list],
    ref_keys: list[int],
) -> dict[str, Any]:
    """(2) Relative diff = max_abs_diff / max|activation| per layer for
    M=16 and M=112. Tests the outlier-magnitude hypothesis (Qwen2 has
    activation outliers of 1e3-1e4)."""
    out = {}
    for n in (16, 112):
        caps, mags, got_keys = _capture_with_magnitude(model, built, pf_cache_list, n, pad_id)
        diffs = _per_layer_diffs(ref_caps, caps, ref_keys, got_keys)
        names = ["residual_in", "norm_out", "attn_out", "mlp_out", "final_norm_out"]
        rel: dict[str, list[float]] = {nm: [] for nm in names}
        abs_mags: dict[str, list[float]] = {nm: [] for nm in names}
        abs_diffs: dict[str, list[float]] = {nm: [] for nm in names}
        for nm in names:
            for li in range(len(diffs[nm])):
                d = diffs[nm][li]
                m = mags[nm][li]
                abs_diffs[nm].append(d)
                abs_mags[nm].append(m)
                rel[nm].append(d / m if m > 0 else 0.0)
        out[f"M={n}"] = {
            "abs_diff": abs_diffs,
            "max_abs_activation": abs_mags,
            "relative_diff": rel,
        }
    return out


def followup_gemm_isolation(
    model,
    built: dict,
    pf_cache_list: list,
    pad_id: int,
    jump_m: int,
) -> dict[str, Any]:
    """(3) At the jump M, isolate the MLP op: compare
    mlp(post_attention_layernorm(h)) computed (a) on the full batch vs
    (b) row by row on the same h, in fp32. If (a) != (b) the GEMM path
    changes with M; if equal, look at the attention output feeding h.

    We re-run the forward to L03 capturing h (the residual after attention,
    i.e. the input to the MLP block) for the full batch, then recompute the
    MLP both ways on that SAME h."""
    from jevmlx.engine import _broadcast_cache, _eval_cache_state

    m = model.model
    rows_n, decisions_n = _rows_for_width(built, jump_m)
    widths = [len(r) for r in rows_n]
    width = max(widths)
    padded = mx.array([r + [pad_id] * (width - len(r)) for r in rows_n], dtype=mx.int32)
    positions = mx.array([d[0] for d in decisions_n], dtype=mx.int32)
    b_cache = _broadcast_cache(pf_cache_list, jump_m)
    max_padding = max(width - w for w in widths) if widths else 0
    if max_padding > 0:
        for c in b_cache:
            if hasattr(c, "prepare"):
                c.prepare(lengths=widths, right_padding=[width - w for w in widths])
    _eval_cache_state(b_cache)

    # Forward to the START of L03's MLP: embed + layers 0,1,2 (full attention).
    h = m.embed_tokens(padded)
    mask = _create_attention_mask(h, b_cache[0])
    for li in range(3):
        layer = m.layers[li]
        x = h
        r = layer.self_attn(layer.input_layernorm(x), mask, b_cache[li])
        h = x + r
        # (we do NOT run the MLP of layers 0-2 here; we need the residual
        #  h AFTER attention but BEFORE MLP — but the block applies both.
        #  Actually the block is h = x + attn(x); then h2 = h + mlp(h).
        #  We need the h that feeds L03's MLP, which is the block OUTPUT of
        #  L02, i.e. after L02's MLP too. So run the full block for 0,1,2.)
        h = h + layer.mlp(layer.post_attention_layernorm(h))

    # h is now the input to L03 (residual_in for L03). Compute L03's MLP
    # input norm.
    layer3 = m.layers[3]
    mlp_in = layer3.post_attention_layernorm(h)  # (M, width, D)
    # Decision-position vectors.
    chunk_len = padded.shape[0]

    # (a) full-batch MLP: run the MLP on the full (M, width, D) tensor.
    mlp_full = layer3.mlp(mlp_in)  # (M, width, D)
    mlp_full_pos = mlp_full[mx.arange(chunk_len), positions]  # (M, D)
    mx.eval(mlp_full_pos)

    # (b) row-by-row MLP on the SAME h: run the MLP one row at a time.
    mlp_row_pos = []
    for i in range(chunk_len):
        one = mlp_in[i : i + 1]  # (1, width, D)
        out = layer3.mlp(one)  # (1, width, D)
        mlp_row_pos.append(out[0, positions[i].item()])  # (D,)
    mlp_row_pos = mx.stack(mlp_row_pos, axis=0)  # (M, D)
    mx.eval(mlp_row_pos)

    diff = mx.max(mx.abs(mlp_full_pos.astype(mx.float32) - mlp_row_pos.astype(mx.float32))).item()
    mag = float(mx.max(mx.abs(mlp_full_pos.astype(mx.float32))).item())
    print(
        f"  GEMM isolation @ M={jump_m}: full-vs-row max abs diff = {diff:.6f}, "
        f"max |activation| = {mag:.6f}, relative = {diff / mag if mag > 0 else 0:.6f}",
        flush=True,
    )
    return {
        "jump_M": jump_m,
        "full_vs_row_max_abs_diff": diff,
        "max_abs_activation": mag,
        "relative_diff": diff / mag if mag > 0 else 0.0,
        "gemm_path_changes_with_M": diff > 1e-6,
    }


def _forward_to_layer_input(model, padded, cache, n_layers: int):
    """Run embed + full blocks for layers [0, n_layers) and return the
    residual input TO layer n_layers (the h feeding its attention)."""
    m = model.model
    h = m.embed_tokens(padded)
    mask = _create_attention_mask(h, cache[0])
    for li in range(n_layers):
        layer = m.layers[li]
        x = h
        r = layer.self_attn(layer.input_layernorm(x), mask, cache[li])
        h = x + r
        h = h + layer.mlp(layer.post_attention_layernorm(h))
    return h, mask


def _run_attention_substeps(layer, x_norm, mask, cache_l):
    """Run the attention block's sub-steps individually, returning the
    intermediate tensors at each stage:
      qkv    : (q, k, v) after projection GEMMs (pre-RoPE, pre-cache)
      qk_rope: (q, k) after RoPE (pre-cache-fetch)
      sdpa   : the scaled_dot_product_attention output
      out    : the o_proj output (full attention result)
    """
    from mlx_lm.models.base import scaled_dot_product_attention as _sdpa

    B, L, D = x_norm.shape
    queries = layer.q_proj(x_norm)
    keys = layer.k_proj(x_norm)
    values = layer.v_proj(x_norm)
    q_pre = queries.reshape(B, L, layer.n_heads, -1).transpose(0, 2, 1, 3)
    k_pre = keys.reshape(B, L, layer.n_kv_heads, -1).transpose(0, 2, 1, 3)
    v_pre = values.reshape(B, L, layer.n_kv_heads, -1).transpose(0, 2, 1, 3)

    if cache_l is not None:
        q_rope = layer.rope(q_pre, offset=cache_l.offset)
        k_rope = layer.rope(k_pre, offset=cache_l.offset)
        k_fetch, v_fetch = cache_l.update_and_fetch(k_rope, v_pre)
    else:
        q_rope = layer.rope(q_pre)
        k_rope = layer.rope(k_pre)
        k_fetch, v_fetch = k_rope, v_pre

    sdpa_out = _sdpa(q_rope, k_fetch, v_fetch, cache=cache_l, scale=layer.scale, mask=mask)
    sdpa_out = sdpa_out.transpose(0, 2, 1, 3).reshape(B, L, -1)
    o_out = layer.o_proj(sdpa_out)
    return {
        "qkv": (q_pre, k_pre, v_pre),
        "qk_rope": (q_rope, k_rope),
        "sdpa": sdpa_out,
        "out": o_out,
    }


def followup_attention_isolation(
    model,
    built: dict,
    pf_cache_list: list,
    pad_id: int,
    widths: tuple[int, ...],
) -> dict[str, Any]:
    """(1)+(2) Attention isolation at L03: full-batch vs row-by-row, split
    into qkv proj / RoPE qk / sdpa / o_proj. Which sub-step first differs.

    For each M in widths: forward to L03's input (h), then run L03's
    attention (a) on the full batch and (b) row by row on the SAME h,
    comparing each sub-step's output at decision positions."""
    from jevmlx.engine import _broadcast_cache, _eval_cache_state

    m = model.model
    layer3 = m.layers[3]
    results = {}
    for n in widths:
        rows_n, decisions_n = _rows_for_width(built, n)
        widths_r = [len(r) for r in rows_n]
        width = max(widths_r)
        padded = mx.array([r + [pad_id] * (width - len(r)) for r in rows_n], dtype=mx.int32)
        positions = mx.array([d[0] for d in decisions_n], dtype=mx.int32)
        b_cache = _broadcast_cache(pf_cache_list, n)
        max_padding = max(width - w for w in widths_r) if widths_r else 0
        if max_padding > 0:
            for c in b_cache:
                if hasattr(c, "prepare"):
                    c.prepare(lengths=widths_r, right_padding=[width - w for w in widths_r])
        _eval_cache_state(b_cache)

        # Forward to L03 input (h) — full batch.
        h_full, mask_full = _forward_to_layer_input(model, padded, b_cache, 3)
        x_norm_full = layer3.input_layernorm(h_full)
        # (a) full-batch attention sub-steps.
        full_steps = _run_attention_substeps(layer3.self_attn, x_norm_full, mask_full, b_cache[3])
        mx.eval(full_steps["out"])

        # (b) row-by-row attention on the SAME h (one row at a time).
        # Use h_full[i:i+1] (the identical batched hidden state sliced to
        # one row) with a fresh batch=1 cache broadcast so cache contents
        # match, but run the attention op at batch=1 — isolating whether
        # the attention GEMM/SDPA path changes with M.
        chunk_len = padded.shape[0]
        row_steps = {k: [] for k in ("qkv", "qk_rope", "sdpa", "out")}
        for i in range(chunk_len):
            h_one = h_full[i : i + 1]  # (1, width, D) — SAME h, one row
            x_norm_one = layer3.input_layernorm(h_one)
            # Fresh cache broadcast for this single row (same KV contents).
            rc = _broadcast_cache(pf_cache_list, 1)
            _eval_cache_state(rc)
            mask_one = _create_attention_mask(h_one, rc[0])
            steps = _run_attention_substeps(layer3.self_attn, x_norm_one, mask_one, rc[3])
            pos_i = positions[i].item()
            for k in row_steps:
                # qkv/qk_rope are (B, H, L, head_dim) after transpose;
                # sdpa/out are (B, L, D) after reshape. Index the L axis.
                if k == "qkv":
                    q, kk, v = steps[k]
                    row_steps[k].append((q[0, :, pos_i, :], kk[0, :, pos_i, :], v[0, :, pos_i, :]))
                elif k == "qk_rope":
                    q, kk = steps[k]
                    row_steps[k].append((q[0, :, pos_i, :], kk[0, :, pos_i, :]))
                else:
                    # sdpa/out are (B, L, D)
                    row_steps[k].append(steps[k][0, pos_i, :])

        # Compare full vs row at decision positions.
        # qkv/qk_rope are (B, H, L, head_dim); sdpa/out are (B, L, D).
        cl = chunk_len
        pos_list = positions.tolist()

        def _gather_pos(t, _cl=cl, _pos=pos_list):
            parts = []
            for b in range(_cl):
                if t.ndim == 4:
                    parts.append(t[b, :, _pos[b], :])  # (H, head_dim)
                else:
                    parts.append(t[b, _pos[b], :])  # (D,)
            return mx.stack(parts, axis=0)

        steps_diff = {}
        for k in ("qkv", "qk_rope", "sdpa", "out"):
            if k == "qkv":
                # 3 tensors (q, k, v)
                diffs = []
                mags = []
                for ti in range(3):
                    full_t = _gather_pos(full_steps[k][ti])
                    row_t = mx.stack([r[ti] for r in row_steps[k]], axis=0)
                    d = float(
                        mx.max(mx.abs(full_t.astype(mx.float32) - row_t.astype(mx.float32))).item()
                    )
                    mg = float(mx.max(mx.abs(full_t.astype(mx.float32))).item())
                    diffs.append(d)
                    mags.append(mg)
                steps_diff[k] = {
                    "sub": ["q_proj", "k_proj", "v_proj"],
                    "max_abs_diff": diffs,
                    "max_abs_activation": mags,
                    "relative_diff": [
                        d / m if m > 0 else 0.0 for d, m in zip(diffs, mags, strict=True)
                    ],
                }
            elif k == "qk_rope":
                diffs = []
                mags = []
                for ti in range(2):
                    full_t = _gather_pos(full_steps[k][ti])
                    row_t = mx.stack([r[ti] for r in row_steps[k]], axis=0)
                    d = float(
                        mx.max(mx.abs(full_t.astype(mx.float32) - row_t.astype(mx.float32))).item()
                    )
                    mg = float(mx.max(mx.abs(full_t.astype(mx.float32))).item())
                    diffs.append(d)
                    mags.append(mg)
                steps_diff[k] = {
                    "sub": ["q_rope", "k_rope"],
                    "max_abs_diff": diffs,
                    "max_abs_activation": mags,
                    "relative_diff": [
                        d / m if m > 0 else 0.0 for d, m in zip(diffs, mags, strict=True)
                    ],
                }
            else:
                full_t = _gather_pos(full_steps[k])
                row_t = mx.stack(row_steps[k], axis=0)
                d = float(
                    mx.max(mx.abs(full_t.astype(mx.float32) - row_t.astype(mx.float32))).item()
                )
                mg = float(mx.max(mx.abs(full_t.astype(mx.float32))).item())
                steps_diff[k] = {
                    "max_abs_diff": d,
                    "max_abs_activation": mg,
                    "relative_diff": d / mg if mg > 0 else 0.0,
                }
        results[f"M={n}"] = steps_diff
        print(f"  attn isolation M={n}:", flush=True)
        for k, v in steps_diff.items():
            if "sub" in v:
                for si, sub in enumerate(v["sub"]):
                    print(
                        f"    {k}/{sub}: abs={v['max_abs_diff'][si]:.6f} "
                        f"mag={v['max_abs_activation'][si]:.6f} "
                        f"rel={v['relative_diff'][si]:.6f}",
                        flush=True,
                    )
            else:
                print(
                    f"    {k}: abs={v['max_abs_diff']:.6f} "
                    f"mag={v['max_abs_activation']:.6f} rel={v['relative_diff']:.6f}",
                    flush=True,
                )
    return results


def followup_cache_and_mask_info(
    model,
    built: dict,
    pf_cache_list: list,
    pad_id: int,
    widths: tuple[int, ...],
) -> dict[str, Any]:
    """(3)+(4) KV cache dtype + mask object type/shape + which mlx function
    is called for attention + argument shapes, at each M."""

    from jevmlx.engine import _broadcast_cache, _eval_cache_state

    m = model.model
    # Cache dtype from the prefill cache.
    cache_dtype = str(pf_cache_list[0].keys.dtype)
    cache_type = type(pf_cache_list[0]).__name__
    has_bits = hasattr(pf_cache_list[0], "bits")
    # Which function sdpa delegates to.
    sdpa_func = (
        "quantized_scaled_dot_product_attention"
        if has_bits
        else "mx.fast.scaled_dot_product_attention"
    )

    out = {
        "cache_type": cache_type,
        "cache_dtype": cache_dtype,
        "cache_has_bits": has_bits,
        "sdpa_function": sdpa_func,
    }
    per_m = {}
    for n in widths:
        rows_n, decisions_n = _rows_for_width(built, n)
        widths_r = [len(r) for r in rows_n]
        width = max(widths_r)
        padded = mx.array([r + [pad_id] * (width - len(r)) for r in rows_n], dtype=mx.int32)
        b_cache = _broadcast_cache(pf_cache_list, n)
        max_padding = max(width - w for w in widths_r) if widths_r else 0
        if max_padding > 0:
            for c in b_cache:
                if hasattr(c, "prepare"):
                    c.prepare(lengths=widths_r, right_padding=[width - w for w in widths_r])
        _eval_cache_state(b_cache)
        # Forward to L03 input to get the mask.
        h, mask = _forward_to_layer_input(model, padded, b_cache, 3)
        mask_type = type(mask).__name__ if mask is not None else "None"
        mask_shape = tuple(int(s) for s in mask.shape) if mask is not None else None
        # The sdpa argument shapes: q, k, v after reshape+rope+fetch.
        attn3 = m.layers[3].self_attn
        x_norm = m.layers[3].input_layernorm(h)
        B, L, D = x_norm.shape
        q = attn3.q_proj(x_norm).reshape(B, L, attn3.n_heads, -1).transpose(0, 2, 1, 3)
        k = attn3.k_proj(x_norm).reshape(B, L, attn3.n_kv_heads, -1).transpose(0, 2, 1, 3)
        v = attn3.v_proj(x_norm).reshape(B, L, attn3.n_kv_heads, -1).transpose(0, 2, 1, 3)
        q = attn3.rope(q, offset=b_cache[3].offset)
        k = attn3.rope(k, offset=b_cache[3].offset)
        k_fetch, v_fetch = b_cache[3].update_and_fetch(k, v)
        per_m[f"M={n}"] = {
            "mask_type": mask_type,
            "mask_shape": mask_shape,
            "q_shape": tuple(int(s) for s in q.shape),
            "k_shape": tuple(int(s) for s in k_fetch.shape),
            "v_shape": tuple(int(s) for s in v_fetch.shape),
            "cache_offset": [int(x) for x in b_cache[3].offset.tolist()]
            if hasattr(b_cache[3].offset, "tolist")
            else int(b_cache[3].offset),
        }
    out["per_M"] = per_m
    return out


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _fmt_table(tables: dict[str, Any], n_layers: int) -> str:
    """Markdown table: rows = layer/capture, columns = M values."""
    ms = sorted(tables.keys(), key=lambda s: int(s.split("=")[1]))
    header = "| layer / capture | " + " | ".join(ms) + " |"
    sep = "|---|" + "|".join(["---"] * len(ms)) + "|"
    lines = [header, sep]
    # Determine layer count from the first capture.
    caps = ["residual_in", "norm_out", "attn_out", "mlp_out", "final_norm_out"]
    # residual_in/norm_out/attn_out/mlp_out have n_layers entries; final_norm_out has 1.
    for li in range(n_layers):
        for cap in caps:
            if cap == "final_norm_out" and li != n_layers - 1:
                continue
            row = f"| L{li:02d} {cap} |"
            for m in ms:
                vals = tables[m][cap]
                # final_norm_out has a single entry (index 0); the per-layer
                # captures have n_layers entries.
                idx = 0 if cap == "final_norm_out" else li
                v = vals[idx] if idx < len(vals) else None
                row += f" {v:.6f} |" if v is not None else " - |"
            lines.append(row)
    # first-nonzero summary
    lines.append("")
    lines.append("**first non-zero diff per capture:**")
    for cap in caps:
        for m in ms:
            vals = tables[m][cap]
            first = next((i for i, v in enumerate(vals) if v > 1e-8), None)
            lines.append(f"- {cap} @ {m}: layer {first}")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="mlx-community/Qwen2.5-0.5B-Instruct-4bit")
    ap.add_argument("--out", default=None, help="output dir override")
    args = ap.parse_args()

    engine = load_engine(args.model)
    tokenizer = engine.tokenizer

    specs = bundled_preset_specs()
    case_id, preset = max(specs, key=lambda kv: len(kv[1].get("schema", {}).get("fields", {})))
    schema = _make_schema(case_id, preset["schema"])
    context = _case_context(case_id, preset)
    built = _build_schema_rows(schema, tokenizer, "slots")
    n_rows = len(built["rows"])
    if n_rows == 0:
        print("probe schema has no rows; aborting", file=sys.stderr)
        return 2
    pf = _prefill(engine.model, tokenizer, context, schema, Ledger(), "slots", engine.profile)
    pf_cache_list = pf.cache
    pad_id = built["pad_id"]
    n_layers = len(engine.model.model.layers)

    print(f"bisecting {args.model}: schema={case_id} rows={n_rows} layers={n_layers}")

    # (1) fp16 body — the native quantized model.
    print("=== (1) fp16 body (native quantized) ===", flush=True)
    fp16_result = run_bisect(engine, built, pf_cache_list, pad_id, label="fp16")

    # (2) fp32 body — dequantize every quantized linear into fp32.
    # NOTE: this mutates the loaded engine's model in place (~2 GB). The
    # prefill cache was built on the fp16 model; the cache VALUES are the
    # same tensors (KV states), and we re-broadcast fresh copies per width,
    # so the fp32 forward reuses the same cached KV (the cache is not
    # re-quantized — it holds the fp16 KV states from prefill).
    print("=== (2) fp32 body (fully dequantized) ===", flush=True)
    _dequantize_model_fp32(engine.model)
    # Re-warm so the fp32 path is compiled.
    _w = engine.model.model(mx.array([[1, 2, 3]], dtype=mx.int32))
    mx.eval(_w)
    fp32_result = run_bisect(engine, built, pf_cache_list, pad_id, label="fp32")

    # The fp32 batch=1 reference (for the follow-ups).
    print("=== (3) fp32 follow-ups ===", flush=True)
    fp32_ref_caps, fp32_ref_keys = _capture_reference_batch1(
        engine.model, built, pf_cache_list, pad_id
    )

    print("  (3.1) width curve (final_norm_out + L03 mlp_out)...", flush=True)
    width_curve = followup_width_curve(
        engine.model, built, pf_cache_list, pad_id, fp32_ref_caps, fp32_ref_keys
    )

    print("  (3.2) relative diff (max_abs_diff / max|activation|)...", flush=True)
    rel_diff = followup_relative_diff(
        engine.model, built, pf_cache_list, pad_id, fp32_ref_caps, fp32_ref_keys
    )

    # Find the jump M: the first M where L03_mlp_out > 0.01.
    jump_m = next(
        (r["M"] for r in width_curve["rows"] if r["L03_mlp_out"] > 0.01),
        width_curve["rows"][-1]["M"] if width_curve["rows"] else 112,
    )
    print(f"  (3.3) GEMM isolation at jump M={jump_m}...", flush=True)
    gemm_iso = followup_gemm_isolation(engine.model, built, pf_cache_list, pad_id, jump_m)

    print("  (3.4) attention isolation (full vs row, 4-way split)...", flush=True)
    attn_iso = followup_attention_isolation(
        engine.model, built, pf_cache_list, pad_id, (16, 32, 112)
    )

    print("  (3.5) cache dtype + mask + sdpa path info...", flush=True)
    cache_info = followup_cache_and_mask_info(
        engine.model, built, pf_cache_list, pad_id, (16, 32, 112)
    )

    print()
    print("## Table 1: drift by layer (fp16 body)")
    print()
    print(_fmt_table(fp16_result["tables"], n_layers))
    print()
    print("## Table 2: drift by layer (fp32 body)")
    print()
    print(_fmt_table(fp32_result["tables"], n_layers))

    # --- Follow-up tables (fp32 body) ---
    print()
    print("## Table 3: fp32 width curve (final_norm_out + L03 mlp_out)")
    print()
    print("| M | final_norm_out | L03_mlp_out |")
    print("|---|---|---|")
    for r in width_curve["rows"]:
        print(f"| {r['M']} | {r['final_norm_out']:.6f} | {r['L03_mlp_out']:.6f} |")

    print()
    print("## Table 4: fp32 relative diff (max_abs_diff / max|activation|)")
    print()
    for mkey in ("M=16", "M=112"):
        rd = rel_diff[mkey]
        print(f"### {mkey}")
        print()
        print("| layer / capture | abs_diff | max_abs_activation | relative_diff |")
        print("|---|---|---|---|")
        names = ["residual_in", "norm_out", "attn_out", "mlp_out", "final_norm_out"]
        n_caps = len(rd["abs_diff"]["residual_in"])
        for li in range(n_caps):
            for nm in names:
                if nm == "final_norm_out" and li != n_caps - 1:
                    continue
                idx = 0 if nm == "final_norm_out" else li
                print(
                    f"| L{li:02d} {nm} | {rd['abs_diff'][nm][idx]:.6f} | "
                    f"{rd['max_abs_activation'][nm][idx]:.6f} | "
                    f"{rd['relative_diff'][nm][idx]:.6f} |"
                )
        print()

    print("## Table 5: GEMM isolation (full-batch vs row-by-row MLP at L03)")
    print()
    print(
        "| jump_M | full_vs_row_max_abs_diff | max_abs_activation |"
        " relative_diff | gemm_path_changes |"
    )
    print("|---|---|---|---|---|")
    print(
        f"| {gemm_iso['jump_M']} | {gemm_iso['full_vs_row_max_abs_diff']:.6f} | "
        f"{gemm_iso['max_abs_activation']:.6f} | {gemm_iso['relative_diff']:.6f} | "
        f"{gemm_iso['gemm_path_changes_with_M']} |"
    )

    # --- Attention isolation table ---
    print()
    print("## Table 6: attention isolation at L03 (full-batch vs row-by-row)")
    print()
    print("| M | step | sub | max_abs_diff | max_abs_activation | relative_diff |")
    print("|---|---|---|---|---|---|")
    for mkey in ("M=16", "M=32", "M=112"):
        steps = attn_iso[mkey]
        for step in ("qkv", "qk_rope", "sdpa", "out"):
            s = steps[step]
            if "sub" in s:
                for si, sub in enumerate(s["sub"]):
                    print(
                        f"| {mkey} | {step} | {sub} | "
                        f"{s['max_abs_diff'][si]:.6f} | "
                        f"{s['max_abs_activation'][si]:.6f} | "
                        f"{s['relative_diff'][si]:.6f} |"
                    )
            else:
                print(
                    f"| {mkey} | {step} | - | "
                    f"{s['max_abs_diff']:.6f} | "
                    f"{s['max_abs_activation']:.6f} | "
                    f"{s['relative_diff']:.6f} |"
                )

    # --- Cache/mask/sdpa info table ---
    print()
    print("## Table 7: cache dtype + mask + sdpa path info")
    print()
    print(
        f"cache_type={cache_info['cache_type']}  cache_dtype={cache_info['cache_dtype']}  "
        f"cache_has_bits={cache_info['cache_has_bits']}  "
        f"sdpa_function={cache_info['sdpa_function']}"
    )
    print()
    print("| M | mask_type | mask_shape | q_shape | k_shape | v_shape | cache_offset |")
    print("|---|---|---|---|---|---|---|")
    for mkey in ("M=16", "M=32", "M=112"):
        pm = cache_info["per_M"][mkey]
        print(
            f"| {mkey} | {pm['mask_type']} | {pm['mask_shape']} | "
            f"{pm['q_shape']} | {pm['k_shape']} | {pm['v_shape']} | "
            f"{pm['cache_offset']} |"
        )

    # Write JSON.
    out_dir = Path(args.out) if args.out else Path("benchmarks/probes") / _probe_dir(args.model)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "layer_bisect.json").write_text(
        json.dumps(
            {
                "model": args.model,
                "case": case_id,
                "schema_rows": n_rows,
                "layers": n_layers,
                "widths": list(WIDTHS),
                "fp16_body": fp16_result,
                "fp32_body": fp32_result,
                "fp32_width_curve": width_curve,
                "fp32_relative_diff": rel_diff,
                "fp32_gemm_isolation": gemm_iso,
                "fp32_attention_isolation": attn_iso,
                "fp32_cache_mask_info": cache_info,
            },
            indent=2,
        )
    )
    print(f"\nwrote {out_dir / 'layer_bisect.json'}")
    return 0


def _probe_dir(model_id: str) -> str:
    """Mirror driftprobe's directory naming."""
    import platform

    arch = platform.machine()
    slug = model_id.replace("/", "--").replace(".", "-")
    return f"{arch}-32gb--{slug}"


if __name__ == "__main__":
    sys.exit(main())
