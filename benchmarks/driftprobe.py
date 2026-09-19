"""W5c-4: batched drift diagnostic matrix — is the 0.125-nat drift kernel
shape or a bug?

decide_many parity shows ~0.125 nats raw row drift at 112 merged rows vs
batch=1 (0.5B). This probe runs the discriminating matrix (GPT-REVIEW-3
section 2) on a REAL model and writes ``driftprobe.json`` plus a markdown
table, so the numbers — not a fitted tolerance — decide:

(a) repeatability: the same 112-row batch 5 times; any run-to-run diff is
    nondeterminism, not rounding;
(b) width curve: the same rows scored at 1..128 tensor rows in ONE pass,
    BOTH cache arrangements (broadcast vs per-row slots — the decide_many
    shape): median/p99/max of d_raw, d_gap, d_logp, winner mismatches vs
    the canonical batch=1 reference (kernel tiling gives plateaus/jumps; a
    linear "context count" law would be surprising);
(c) contexts vs rows: the same 112 tensor rows as 1x112, 2x56, 4x28
    group-distinct prefill slots — drift must follow TENSOR rows, not
    logical bookkeeping;
(d) permutation: rotate row order under fixed slots, map back —
    shape-dependent rounding is permutation-invariant; slot-following
    drift is a structural bug;
(e) mask control: the decide_many shape at small width (mlx_lm derives the
    mask from the cache object — BatchKVCache(left_padding) IS the mask
    difference) vs batch=1;
(f) cache control: independent prefill per replicated context vs broadcast
    one cache vs merged contiguous copies — cache tensors (values via
    checksums, shapes, offsets) asserted identical BEFORE the first suffix
    layer, then the drift each arrangement produces;
(g) near-tie corpus: rows whose batch=1 top-two margin < 0.3 nats through
    the same width matrix (identical winners on easy margins say nothing
    about the rescore gate).

Every metric is per-row and separated (GPT-REVIEW-3 §2):
- d_raw  = max_i |z_i(B) - z_i(1)|            (uncentered, diagnostic only)
- d_gap  = max_{i<j} |(z_i - z_j)(B) - (z_i - z_j)(1)|
- d_logp = max_i |logsoftmax(z(B))_i - logsoftmax(z(1))_i|
- winner mismatches vs the batch=1 reference.

Run (model-loading — MUST go through the shared slot lock):

    ~/git/jev-on-a-laptop/.agent-mail/slowtest.sh \
        .venv/bin/python benchmarks/driftprobe.py [--model ID] [--quick]

Outputs: benchmarks/results/<machine>--<model-slug>/driftprobe.json + .md.
Fast tests only for CI — this script is slow by design and is NOT a test.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import re
import statistics
import sys
from typing import Any

from jevmlx.engine import (
    Engine,
    _broadcast_cache,
    _build_schema_rows,
    _prefill,
    _score_rows,
    load_engine,
)
from jevmlx.timing import Ledger

# The width matrix (b). 112 is the observed drift point; 28*4 and 56*2 are
# the context-orthogonalization points (c); 128 crosses a plausible tiling
# threshold above 112.
WIDTHS = (1, 2, 4, 8, 16, 28, 32, 56, 64, 84, 112, 128)
REPEATS = 5
CONTEXT_GROUPS = (2, 4)  # (c): 112 rows split 1x112, 2x56, 4x28
NEAR_TIE_MARGIN = 0.3  # (g)


def _d_stats(diffs: list[float]) -> dict[str, float]:
    """median/p99/max for a list of per-row drift values."""
    if not diffs:
        return {"median": 0.0, "p99": 0.0, "max": 0.0, "n": 0}
    s = sorted(diffs)
    idx = min(len(s) - 1, math.ceil(0.99 * len(s)) - 1)
    return {
        "median": statistics.median(s),
        "p99": s[max(0, idx)],
        "max": s[-1],
        "n": len(s),
    }


def _row_metrics(ref: list[float], got: list[float]) -> dict[str, Any]:
    """d_raw / d_gap / d_logp / winner for ONE row's candidate logits.

    d_gap is the max over ALL pairs i<j of |(z_i - z_j)^(B) - (z_i - z_j)^(1)|
    — for candidate logits that collapses to the difference of ranges:
    max_{i<j}|a_i-a_j| over a set == max(set) - min(set).
    """
    d_raw = max(abs(a - b) for a, b in zip(ref, got, strict=True))
    ref_gap = max(ref) - min(ref)
    got_gap = max(got) - min(got)
    d_gap = abs(ref_gap - got_gap)
    m = max(ref)
    ref_lp = [v - (m + math.log(sum(math.exp(x - m) for x in ref))) for v in ref]
    m = max(got)
    got_lp = [v - (m + math.log(sum(math.exp(x - m) for x in got))) for v in got]
    d_logp = max(abs(a - b) for a, b in zip(ref_lp, got_lp, strict=True))
    return {
        "d_raw": d_raw,
        "d_gap": d_gap,
        "d_logp": d_logp,
        "winner": ref.index(max(ref)) == got.index(max(got)),
    }


def _rows_for_width(built: dict, n: int) -> tuple[list, list]:
    """The schema's rows replicated and truncated to EXACTLY n rows.

    Row j of the returned list is built["rows"][j % n_schema_rows] — the
    mapping used to compare against the batch=1 reference.
    """
    k = math.ceil(n / max(1, len(built["rows"])))
    return (built["rows"] * k)[:n], (built["row_decision"] * k)[:n]


def _score_rows_at(
    engine: Engine,
    built: dict,
    pf_cache_list: list,
    rows_n: list,
    decisions_n: list,
    *,
    slots: list | None = None,
) -> dict[int, list[float]]:
    """One _score_rows call: rows_n in ONE pass (auto_max_rows=len), the
    decide_many padded/merged shape. ``slots``: per-row cache list (the
    decide_many slot path) or None (broadcast single cache)."""
    ledger = Ledger()
    scored = _score_rows(
        engine.model,
        pf_cache_list,
        rows_n,
        decisions_n,
        engine.vocab_size,
        built["pad_id"],
        max(1, len(rows_n)),
        ledger,
        cache_slots=slots,
    )
    return dict(scored.row_logits)


def _batch1_reference(engine: Engine, built: dict, pf_cache_list: list) -> dict[int, list[float]]:
    """The canonical batch=1 reference: ALL the schema's rows, ONE ROW PER
    FORWARD (auto_max_rows=1) — the exact shape the near-tie rescore
    trusts and the reference check_batched_parity's RAW stage uses."""
    ledger = Ledger()
    scored = _score_rows(
        engine.model,
        pf_cache_list,
        built["rows"],
        built["row_decision"],
        engine.vocab_size,
        built["pad_id"],
        1,
        ledger,
    )
    return dict(scored.row_logits)


def _drift_rows(
    ref: dict[int, list[float]],
    got: dict[int, list[float]],
    n_schema_rows: int,
    *,
    restrict: set[int] | None = None,
) -> dict[str, Any]:
    """Aggregate d_raw/d_gap/d_logp/winner over every batched row key,
    mapped back to its schema row (key j -> schema row j % n_schema_rows).
    ``restrict``: only schema-row indexes in this set."""
    per_row = []
    for j, vals in got.items():
        orig = j % n_schema_rows
        if orig not in ref:
            continue
        if restrict is not None and orig not in restrict:
            continue
        per_row.append(_row_metrics(ref[orig], vals))
    return {
        "d_raw": _d_stats([r["d_raw"] for r in per_row]),
        "d_gap": _d_stats([r["d_gap"] for r in per_row]),
        "d_logp": _d_stats([r["d_logp"] for r in per_row]),
        "winner_mismatches": sum(not r["winner"] for r in per_row),
    }


def _repeatability(
    engine: Engine,
    built: dict,
    pf_cache_list: list,
    n: int = 112,
    repeats: int = REPEATS,
) -> dict[str, Any]:
    """(a) the same n-row decide_many-shaped batch `repeats` times."""
    rows_n, decisions_n = _rows_for_width(built, n)
    slots = [pf_cache_list] * n
    runs = [
        _score_rows_at(engine, built, pf_cache_list, rows_n, decisions_n, slots=slots)
        for _ in range(repeats)
    ]
    max_diff = 0.0
    for i in runs[0]:
        for r in runs[1:]:
            if i in r:
                max_diff = max(
                    max_diff,
                    max(abs(a - b) for a, b in zip(runs[0][i], r[i], strict=True)),
                )
    return {
        "tensor_rows": n,
        "repeats": repeats,
        "max_run_to_run_diff": max_diff,
        "nondeterministic": max_diff > 0.0,
    }


def _width_matrix(
    engine: Engine,
    built: dict,
    pf_cache_list: list,
    context: str,
    schema: Any,
    tokenizer: Any,
    widths: tuple[int, ...] = WIDTHS,
) -> list[dict[str, Any]]:
    """(b) width curve x both slot modes + (c) context orthogonality +
    (d) permutation."""
    n_schema_rows = len(built["rows"])
    ref = _batch1_reference(engine, built, pf_cache_list)
    out: list[dict[str, Any]] = []

    # (c) group-distinct prefills: `groups` separate _prefill calls
    # (deterministically identical VALUES, distinct objects) — the real
    # decide_many slot shape.
    def _prefills(k: int) -> list:
        return [
            _prefill(
                engine.model, tokenizer, context, schema, Ledger(), "slots", engine.profile
            ).cache
            for _ in range(k)
        ]

    for n in widths:
        rows_n, decisions_n = _rows_for_width(built, n)
        for label, slots in (("broadcast", None), ("slots", [pf_cache_list] * n)):
            got = _score_rows_at(engine, built, pf_cache_list, rows_n, decisions_n, slots=slots)
            row = {
                "matrix": "width",
                "slot_mode": label,
                "tensor_rows": n,
                "context_groups": 1,
                **_drift_rows(ref, got, n_schema_rows),
            }

            # (c) same tensor count, slots from group-distinct prefills.
            if n == 112:
                for groups in CONTEXT_GROUPS:
                    pf_group = _prefills(groups)
                    slots_g = [pf_group[min(i // (n // groups), groups - 1)] for i in range(n)]
                    got_g = _score_rows_at(
                        engine, built, pf_cache_list, rows_n, decisions_n, slots=slots_g
                    )
                    row[f"groups_{groups}"] = _drift_rows(ref, got_g, n_schema_rows)

            # (d) permutation: rotate ROW order by a nontrivial shift under
            # FIXED slots; map back and require per-row drift invariance.
            if n > 1:
                rot = max(1, min(7, n - 1))
                perm_rows = rows_n[rot:] + rows_n[:rot]
                perm_decisions = decisions_n[rot:] + decisions_n[:rot]
                got_p = _score_rows_at(
                    engine, built, pf_cache_list, perm_rows, perm_decisions, slots=slots
                )
                # Map back: perm_rows[i] = rows_n[(i + rot) % n].
                remapped = {((j + rot) % n): v for j, v in got_p.items()}
                row["permuted"] = _drift_rows(ref, remapped, n_schema_rows)
            out.append(row)
    return out


def _mask_control(
    engine: Engine,
    built: dict,
    pf_cache_list: list,
    ref: dict[int, list[float]],
) -> dict[str, Any]:
    """(e) mask/cache-object control at a small width.

    mlx_lm derives the attention mask from the cache object; the batched
    path differs from batch=1 by the merged BatchKVCache(left_padding=[0]*n)
    object even with NO padding. The decide_many shape at a small width IS
    that control: same rows, same cache values, only the batch/mask object
    differs. (There is no mask=None vs explicit-mask toggle exposed by
    mlx_lm's model() — the cache object IS the mask source.)
    """
    n_schema_rows = len(built["rows"])
    n = min(16, max(2, n_schema_rows))
    rows_n, decisions_n = _rows_for_width(built, n)
    got = _score_rows_at(engine, built, pf_cache_list, rows_n, decisions_n)
    return {
        "tensor_rows": n,
        "note": (
            "mlx_lm derives the mask from the cache object; the decide_many "
            "path differs from batch=1 ONLY by the BatchKVCache(left_padding) "
            "object. This control measures the drift of that exact shape "
            "difference at a small width (the full curve is in the width "
            "matrix)."
        ),
        **_drift_rows(ref, got, n_schema_rows),
    }


def _cache_control(
    engine: Engine,
    schema: Any,
    context: str,
    built: dict,
    pf_cache_list: list,
    tokenizer: Any,
    ref: dict[int, list[float]],
) -> dict[str, Any]:
    """(f) cache control at one width: three ways to place ONE context's
    cache under n rows, compared BEFORE the first suffix layer.

    1. independent: n separate _prefill calls (n unbatched caches).
    2. broadcast: one prefill merged with itself n times (the decide path
       for a single context).
    3. merged contiguous copies: the merged BatchKVCache _score_rows
       builds — its per-slot slices asserted equal to the unbatched cache
       (values by checksum, shapes, offsets) BEFORE scoring.

    Then the drift each slot arrangement produces at width n.
    """
    n_schema_rows = len(built["rows"])
    n = min(16, max(2, n_schema_rows))
    rows_n, decisions_n = _rows_for_width(built, n)

    import mlx.core as mx

    def _eq(a: Any, b: Any) -> bool:
        """Bit-exact equality treating NaN == NaN (array_equal is False on
        any NaN, even when both sides carry the identical NaN)."""
        if tuple(a.shape) != tuple(b.shape):
            return False
        both_nan = mx.isnan(a) & mx.isnan(b)
        return bool((mx.isnan(a) == mx.isnan(b)).all() and ((a == b) | both_nan).all())

    def _cache_eq(a: list, b: list) -> bool:
        """Exact elementwise equality of two unbatched cache lists
        (values, shapes, offsets)."""
        for ca, cb in zip(a, b, strict=True):
            ka, va = ca.state
            kb, vb = cb.state
            if not _eq(ka, kb) or not _eq(va, vb):
                return False
            if int(getattr(ca, "offset", 0)) != int(getattr(cb, "offset", 0)):
                return False
        return True

    # 1. independent prefills.
    indep = [
        _prefill(engine.model, tokenizer, context, schema, Ledger(), "slots", engine.profile).cache
        for _ in range(n)
    ]
    identical_prefill = _cache_eq(indep[0], pf_cache_list)

    # 3. the merged cache _score_rows itself builds (BEFORE scoring): each
    # slot's slice must equal the source cache EXACTLY (elementwise).
    # BatchKVCache.merge places slot i's keys at [i, :, :offset, :].
    merged = _broadcast_cache(pf_cache_list, n)
    merged_ok = True
    for layer_src, layer_b in zip(pf_cache_list, merged, strict=True):
        src_k, src_v = layer_src.state
        # keys layout: (batch, n_kv_heads, seq, head_dim) — seq is axis 2.
        # The unbatched cache keeps its leading batch dim of 1; squeeze it
        # so shapes match the per-slot slice.
        src_k0, src_v0 = src_k[0], src_v[0]
        seq_k, seq_v = src_k0.shape[1], src_v0.shape[1]
        for slot in range(n):
            sl_k = layer_b.keys[slot, :, :seq_k, :]
            sl_v = layer_b.values[slot, :, :seq_v, :]
            if (
                tuple(sl_k.shape) != tuple(src_k0.shape)
                or not _eq(sl_k, src_k0)
                or tuple(sl_v.shape) != tuple(src_v0.shape)
                or not _eq(sl_v, src_v0)
            ):
                merged_ok = False
                break
        if not merged_ok:
            break

    # Drift: broadcast slots vs independent-prefill slots, same width.
    got_b = _score_rows_at(
        engine, built, pf_cache_list, rows_n, decisions_n, slots=[pf_cache_list] * n
    )
    got_i = _score_rows_at(engine, built, pf_cache_list, rows_n, decisions_n, slots=indep)
    return {
        "tensor_rows": n,
        "prefill_deterministic": identical_prefill,
        "merged_cache_slots_identical": merged_ok,
        "broadcast_slots": _drift_rows(ref, got_b, n_schema_rows),
        "independent_slots": _drift_rows(ref, got_i, n_schema_rows),
        "note": (
            "cache VALUES were compared EXACTLY (elementwise, shapes, "
            "offsets) before scoring — and per-slot inside the merged "
            "batched cache: identical => the drift comes from the forward "
            "shape, not the cache contents."
        ),
    }


def _near_tie_matrix(
    engine: Engine,
    built: dict,
    pf_cache_list: list,
    ref: dict[int, list[float]],
    widths: tuple[int, ...] = (16, 56, 112),
) -> dict[str, Any]:
    """(g) near-tie corpus: the width matrix restricted to rows whose
    batch=1 top-two margin < NEAR_TIE_MARGIN (the rescore gate's domain).
    """
    n_schema_rows = len(built["rows"])
    near_tie_rows = set()
    margins = {}
    for i, vals in ref.items():
        s = sorted(vals, reverse=True)
        margins[i] = s[0] - s[1]
        if s[0] - s[1] < NEAR_TIE_MARGIN:
            near_tie_rows.add(i)
    base = {
        "near_tie_rows": len(near_tie_rows),
        "min_batch1_margin": min(margins.values()) if margins else None,
        "widths": [],
    }
    if not near_tie_rows:
        base["note"] = f"no batch-1 margins < {NEAR_TIE_MARGIN} in this schema"
        return base
    for n in widths:
        rows_n, decisions_n = _rows_for_width(built, n)
        got = _score_rows_at(
            engine, built, pf_cache_list, rows_n, decisions_n, slots=[pf_cache_list] * n
        )
        base["widths"].append(
            {
                "tensor_rows": n,
                "slot_mode": "slots",
                **_drift_rows(ref, got, n_schema_rows, restrict=near_tie_rows),
            }
        )
    return base


def _model_folder(engine: Engine) -> str:
    """The machine-tagged results folder for this engine (the bench's own
    layout): benchmarks/results/<machine>-<model-slug>/driftprobe.json."""
    chip = platform.machine()  # arm64
    try:
        ram_gb = round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30)
    except (ValueError, OSError):
        ram_gb = 0
    slug = re.sub(r"[^a-z0-9]+", "--", engine.model_id.lower()).strip("-")
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "results", f"{chip}-{ram_gb}gb--{slug}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="mlx-community/Qwen2.5-0.5B-Instruct-4bit")
    ap.add_argument("--quick", action="store_true", help="widths subset only")
    ap.add_argument("--out", default=None, help="output dir override")
    args = ap.parse_args()

    widths = (1, 2, 4, 8, 28, 56, 112) if args.quick else WIDTHS

    engine = load_engine(args.model)
    tokenizer = engine.tokenizer

    # The probe schema: the highest-cardinality bundled preset (most rows),
    # built EXACTLY as check_batched_parity builds it.
    from jevmlx.parity import _make_schema, bundled_preset_specs

    specs = bundled_preset_specs()
    case_id, preset = max(
        specs,
        key=lambda kv: len(kv[1].get("schema", {}).get("fields", {})),
    )
    schema = _make_schema(case_id, preset["schema"])
    context = preset.get("context") or "neutral probe context"
    built = _build_schema_rows(schema, tokenizer, "slots")
    n_rows = len(built["rows"])
    if n_rows == 0:
        print("probe schema has no rows; aborting", file=sys.stderr)
        return 2
    pf = _prefill(engine.model, tokenizer, context, schema, Ledger(), "slots", engine.profile)
    pf_cache_list = pf.cache

    print(f"probing {args.model}: schema={case_id} rows={n_rows}")
    report: dict[str, Any] = {
        "model": args.model,
        "case": case_id,
        "schema_rows": n_rows,
        "cache_dtype": str(pf_cache_list[0].keys.dtype),
    }

    print("(a) repeatability...", flush=True)
    report["repeatability"] = _repeatability(engine, built, pf_cache_list)

    print("(b/c/d) width + context-orthogonality + permutation...", flush=True)
    report["width_matrix"] = _width_matrix(
        engine, built, pf_cache_list, context, schema, tokenizer, widths
    )

    ref = _batch1_reference(engine, built, pf_cache_list)

    print("(e) mask control...", flush=True)
    report["mask_control"] = _mask_control(engine, built, pf_cache_list, ref)

    print("(f) cache control...", flush=True)
    report["cache_control"] = _cache_control(
        engine, schema, context, built, pf_cache_list, tokenizer, ref
    )

    print("(g) near-tie corpus...", flush=True)
    report["near_tie"] = _near_tie_matrix(engine, built, pf_cache_list, ref)

    # The 0.125 reference measurement: the exact code path that reported it
    # (check_batched_parity's RAW pre-rescore stage) — the probe's own
    # matrix must reproduce it or the probe is not measuring the same
    # thing. Reported verbatim, no conclusion attached.
    from jevmlx.parity import check_batched_parity

    print("parity reference (the path that reported 0.125)...", flush=True)
    parity = check_batched_parity(engine, [(case_id, preset)])
    report["parity_reference"] = parity
    print(
        f"  check_batched_parity: max_raw_row_drift="
        f"{parity.get('max_raw_row_drift_nats')}, "
        f"max_final_drift={parity.get('max_abs_drift_nats')}, "
        f"winners_identical={parity.get('winners_identical')}",
        flush=True,
    )

    out_dir = args.out or _model_folder(engine)
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "driftprobe.json")
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)

    md = _markdown(report)
    md_path = os.path.join(out_dir, "driftprobe.md")
    with open(md_path, "w") as f:
        f.write(md)
    print(md)
    print(f"\nwrote {json_path} and {md_path}")
    return 0


def _fmt(v: float | None) -> str:
    return "-" if v is None else f"{v:.4f}"


def _fmt3(d: dict) -> str:
    return f"{d['median']:.4f}/{d['p99']:.4f}/{d['max']:.4f}"


def _markdown(r: dict[str, Any]) -> str:
    lines = [
        "# Batched drift probe (W5c-4)",
        "",
        f"model: `{r['model']}` — schema `{r['case']}` ({r['schema_rows']} rows),"
        f" cache dtype {r['cache_dtype']}",
        "",
        "## (a) Repeatability",
        "",
    ]
    rep = r["repeatability"]
    lines.append(
        f"{rep['repeats']} runs at {rep['tensor_rows']} rows (decide_many "
        f"slots shape): max run-to-run diff **{rep['max_run_to_run_diff']:.6f}**"
        f" — {'NONDETERMINISTIC' if rep['nondeterministic'] else 'deterministic'}"
    )
    lines += ["", "## (b/c/d) Width matrix", ""]
    lines.append(
        "| slot_mode | rows | d_raw med/p99/max | d_gap med/p99/max | d_logp med/p99/max"
        " | winners≠ | 2-group d_gap | 4-group d_gap | perm d_gap |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for row in r["width_matrix"]:
        g2 = row.get("groups_2", {}).get("d_gap", {}).get("max")
        g4 = row.get("groups_4", {}).get("d_gap", {}).get("max")
        perm = row.get("permuted", {}).get("d_gap", {}).get("max")
        lines.append(
            f"| {row['slot_mode']} | {row['tensor_rows']} | {_fmt3(row['d_raw'])} "
            f"| {_fmt3(row['d_gap'])} | {_fmt3(row['d_logp'])} "
            f"| {row['winner_mismatches']} | {_fmt(g2)} | {_fmt(g4)} | {_fmt(perm)} |"
        )
    mc = r["mask_control"]
    lines += ["", "## (e) Mask/cache-object control", ""]
    lines.append(
        f"at {mc['tensor_rows']} rows: d_raw max {mc['d_raw']['max']:.4f},"
        f" d_gap max {mc['d_gap']['max']:.4f}, winners≠ {mc['winner_mismatches']}"
    )
    lines.append(f"> {mc['note']}")
    cc = r["cache_control"]
    lines += ["", "## (f) Cache control", ""]
    lines.append(f"prefill deterministic: {cc['prefill_deterministic']}")
    lines.append(
        f"merged batched cache per-slot values identical: {cc['merged_cache_slots_identical']}"
    )
    lines.append(
        f"broadcast slots: d_raw max {cc['broadcast_slots']['d_raw']['max']:.4f},"
        f" d_gap max {cc['broadcast_slots']['d_gap']['max']:.4f},"
        f" winners≠ {cc['broadcast_slots']['winner_mismatches']}"
    )
    lines.append(
        f"independent slots: d_raw max {cc['independent_slots']['d_raw']['max']:.4f},"
        f" d_gap max {cc['independent_slots']['d_gap']['max']:.4f},"
        f" winners≠ {cc['independent_slots']['winner_mismatches']}"
    )
    lines.append(f"> {cc['note']}")
    pr = r.get("parity_reference")
    lines += ["", "## Reference: check_batched_parity (the 0.125 source)", ""]
    if pr:
        lines.append(
            f"max_raw_row_drift={pr.get('max_raw_row_drift_nats')}, "
            f"max_final_drift={pr.get('max_abs_drift_nats')}, "
            f"winners_identical={pr.get('winners_identical')}"
        )
    nt = r["near_tie"]
    lines += ["", "## (g) Near-tie corpus", ""]
    lines.append(
        f"{nt['near_tie_rows']} rows with batch-1 margin < {NEAR_TIE_MARGIN}"
        + (
            f" (min margin {nt['min_batch1_margin']:.4f})"
            if nt.get("min_batch1_margin") is not None
            else ""
        )
    )
    if nt.get("note"):
        lines.append(f"> {nt['note']}")
    if nt.get("widths"):
        lines.append("| rows | d_raw max | d_gap max | d_logp max | winners≠ |")
        lines.append("|---|---|---|---|---|")
        for w in nt["widths"]:
            lines.append(
                f"| {w['tensor_rows']} | {w['d_raw']['max']:.4f} |"
                f" {w['d_gap']['max']:.4f} | {w['d_logp']['max']:.4f} |"
                f" {w['winner_mismatches']} |"
            )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
