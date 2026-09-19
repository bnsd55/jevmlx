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
