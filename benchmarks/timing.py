#!/usr/bin/env python3
"""Timing report for the parallel constrained decision engine.

GPT-REVIEW 'PR 3: Report these separately': one tool that runs decide() on a
model for the bundled presets, N repetitions, and reports the engine's
timing split — prior_ms, prefill_ms, plan_compile_ms, cache_broadcast_ms,
suffix_eval_ms, lm_head_gather_ms, second_pass_ms, total_ms,
peak_active_bytes — plus rows, padded token positions, forward passes,
rescored fields and the second-pass rerun rate. Median and p95 over the
repetitions; full per-rep raw numbers written alongside.

Usage:
    .venv/bin/python -m benchmarks.timing --model mlx-community/Qwen2.5-0.5B-Instruct-4bit
    .venv/bin/python -m benchmarks.timing --model <id> --reps 5 --out DIR
"""

from __future__ import annotations

import argparse
import json
import os
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

DEFAULT_PRESETS = [
    "fintech_fraud.json",
    "support_triage.json",
    "high_cardinality_255.json",
]

# The engine's timing split (bug 9) — exactly these keys, median + p95.
TIMING_KEYS = [
    "prior_ms",
    "prefill_ms",
    "plan_compile_ms",
    "cache_broadcast_ms",
    "suffix_eval_ms",
    "lm_head_gather_ms",
    "second_pass_ms",
    "total_ms",
]

COUNT_KEYS = [
    "peak_active_bytes",
    "rows",
    "padded_token_positions",
    "sequential_forward_passes",
    "rescored_fields_count",
    "num_fields",
]


def extract_telemetry(result: dict) -> dict:
    """Pull the timing split + counters out of one engine result."""
    field_telemetry = result.get("field_telemetry", {})
    rescored = result.get("rescored_fields", [])
    rerun_fields = result.get("rerun_fields", [])
    num_fields = result.get("num_fields", len(field_telemetry))
    return {
        **{k: result.get(k, 0.0) for k in TIMING_KEYS},
        "peak_active_bytes": result.get("peak_active_bytes", 0),
        "rows": sum(t.get("rows", 0) for t in field_telemetry.values()),
        "padded_token_positions": result.get("padded_token_positions", 0),
        "sequential_forward_passes": result.get("sequential_forward_passes", 0),
        "rescored_fields_count": len(rescored),
        "num_fields": num_fields,
        # Rerun rate: second-pass fields / fields (0.0 when nothing reruns).
        "rerun_rate": round(len(rerun_fields) / num_fields, 4) if num_fields else 0.0,
    }


def _p95(values: list[float]) -> float:
    """Nearest-rank p95 (ceil(0.95*n)-th of the sorted values)."""
    if not values:
        return 0.0
    ordered = sorted(values)
    import math

    idx = min(len(ordered) - 1, max(0, math.ceil(0.95 * len(ordered)) - 1))
    return ordered[idx]


def aggregate(reps: list[dict]) -> dict:
    """Median + p95 over per-rep telemetry dicts (one per preset)."""
    agg: dict[str, dict[str, float]] = {}
    keys = list(reps[0].keys()) if reps else []
    for key in keys:
        values = [rep[key] for rep in reps]
        if not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values):
            # Non-numeric keys (e.g. a preset tag) pass through untouched.
            agg[key] = {"value": values[0]}
            continue
        agg[key] = {
            "median": round(statistics.median(values), 4),
            "p95": round(_p95(values), 4),
            "min": round(min(values), 4),
            "max": round(max(values), 4),
        }
    return agg


def run_timing(
    model_id: str,
    presets: list[str] | None = None,
    reps: int = 3,
    prior_correction: bool = False,
    load_engine_fn=None,
    run_fn=None,
    load_preset_fn=None,
) -> dict:
    """Run decide() on each preset `reps` times; returns the report dict.

    ``load_engine_fn``/``run_fn``/``load_preset_fn`` are injectable for the
    fake-model test; defaults load a real mlx-lm model.
    """
    if load_engine_fn is None:
        from jevmlx.engine import load_engine as load_engine_fn
    if run_fn is None:
        from jevmlx.engine import run_parallel_generation as run_fn
    if load_preset_fn is None:
        from jevmlx.cli import load_preset as load_preset_fn

    presets = presets or DEFAULT_PRESETS
    report: dict = {
        "model": model_id,
        "reps": reps,
        "prior_correction": prior_correction,
        "presets": {},
    }

    model, tokenizer = load_engine_fn(model_id)
    for rel in presets:
        preset = load_preset_fn(rel)
        from jevmlx.schema import StructuredSchema

        schema = StructuredSchema(preset["schema"])
        rep_rows: list[dict] = []
        for _rep in range(reps):
            result = run_fn(
                model,
                tokenizer,
                preset["context"],
                schema,
                temperature=1.0,
                scoring="slots",
                prior_correction=prior_correction,
            )
            row = extract_telemetry(result)
            row["preset"] = preset["id"]
            rep_rows.append(row)
        report["presets"][preset["id"]] = {
            "title": preset["title"],
            "num_fields": len(preset["schema"]),
            "aggregate": aggregate(rep_rows),
            "raw": rep_rows,
        }
    return report


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model", required=True, help="Hugging Face model id (mlx-lm compatible)")
    ap.add_argument("--reps", type=int, default=3, help="decide() repetitions per preset")
    ap.add_argument("--presets", nargs="+", default=DEFAULT_PRESETS)
    ap.add_argument(
        "--prior-correction", action="store_true", help="include the neutral-context prior pass"
    )
    ap.add_argument("--out", default=os.path.join(ROOT, "results"), help="output directory")
    ap.add_argument("--tag", default=None, help="output name (default: timing-<model slug>)")
    args = ap.parse_args()

    report = run_timing(args.model, args.presets, args.reps, args.prior_correction)

    tag = args.tag or f"timing-{args.model.replace('/', '_')}"
    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, f"{tag}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Human-readable print: one block per preset, the split with median/p95.
    print(f"model: {args.model} | reps: {args.reps} | prior_correction: {args.prior_correction}")
    for preset_id, block in report["presets"].items():
        print(f"\n== {preset_id} ({block['num_fields']} fields) ==")
        agg = block["aggregate"]
        header = f"{'metric':<28}{'median':>12}{'p95':>12}"
        print(header)
        for key in [
            *TIMING_KEYS,
            "rows",
            "padded_token_positions",
            "sequential_forward_passes",
            "rescored_fields_count",
            "rerun_rate",
            "peak_active_bytes",
        ]:
            if key in agg:
                print(f"{key:<28}{agg[key]['median']:>12.2f}{agg[key]['p95']:>12.2f}")
    print(f"\nsaved {out_path}", flush=True)


if __name__ == "__main__":
    main()
