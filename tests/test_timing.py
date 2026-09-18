"""W3-R timing report: aggregation logic only (fake model).

The real run (mlx-lm model, Metal) is for the M5 machine; these tests pin
the extract/aggregate/median/p95 contract that the report and the CLI print.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from conftest import make_engine_result

from benchmarks.timing import TIMING_KEYS, aggregate, extract_telemetry, run_timing


def test_extract_telemetry_pulls_split_and_counters():
    result = {
        "prior_ms": 12.0,
        "prefill_ms": 40.0,
        "plan_compile_ms": 1.5,
        "cache_broadcast_ms": 3.0,
        "suffix_eval_ms": 90.0,
        "lm_head_gather_ms": 4.0,
        "second_pass_ms": 8.0,
        "total_ms": 150.0,
        "peak_active_bytes": 12345,
        "padded_token_positions": 777,
        "sequential_forward_passes": 3,
        "rescored_fields": ["pick"],
        "rerun_fields": ["parent"],
        "num_fields": 4,
        "field_telemetry": {
            "a": {"rows": 2},
            "b": {"rows": 3},
        },
    }
    row = extract_telemetry(result)
    for k in TIMING_KEYS:
        assert k in row
    assert row["rows"] == 5  # summed from field telemetry
    assert row["padded_token_positions"] == 777
    assert row["sequential_forward_passes"] == 3
    assert row["rescored_fields_count"] == 1
    assert row["rerun_rate"] == 0.25  # 1 rerun field / 4 fields
    assert row["peak_active_bytes"] == 12345
    assert row["num_fields"] == 4


def test_extract_telemetry_zero_rerun_rate_when_no_fields():
    row = extract_telemetry({"num_fields": 0, "field_telemetry": {}})
    assert row["rerun_rate"] == 0.0


def test_aggregate_median_p95_min_max():
    reps = [
        {"total_ms": 100.0, "prior_ms": 10.0, "rows": 4},
        {"total_ms": 200.0, "prior_ms": 20.0, "rows": 4},
        {"total_ms": 300.0, "prior_ms": 30.0, "rows": 4},
        {"total_ms": 400.0, "prior_ms": 40.0, "rows": 4},
    ]
    agg = aggregate(reps)
    assert agg["total_ms"]["median"] == 250.0  # (200+300)/2
    # p95 nearest-rank over 4 values: ceil(3.8)=4th -> 400.
    assert agg["total_ms"]["p95"] == 400.0
    assert agg["total_ms"]["min"] == 100.0
    assert agg["total_ms"]["max"] == 400.0
    assert agg["rows"]["median"] == 4


def test_run_timing_fake_model_three_presets_one_call():
    """run_timing with injected fakes: each preset decides `reps` times; the
    aggregate carries median + p95; no real model is loaded."""
    calls = {"n": 0}
    preset_call = {"n": 0}

    def fake_load_engine(_model_id):
        return ("fake-model", "fake-tok")

    def fake_run(
        model, tokenizer, context, schema, temperature=1.0, scoring="slots", prior_correction=False
    ):
        # The preset's title changes per preset; reset the local counter so
        # each preset sees reps 1..N (rescored odd reps only -> median 0).
        if schema is not getattr(fake_run, "_last_schema", None):
            fake_run._last_schema = schema
            preset_call["n"] = 0
        preset_call["n"] += 1
        calls["n"] += 1
        n = preset_call["n"]
        # Drifting total_ms so median/p95 have something to bite on.
        return make_engine_result(
            elapsed_ms=10.0 + calls["n"],
            prefill_ms=5.0,
            suffix_eval_ms=4.0,
            total_ms=10.0 + n,
            peak_active_bytes=1000 + n,
            padded_token_positions=12 * (n % 3 + 1),
            sequential_forward_passes=2,
            rescored_fields=["pick"] if n % 2 else [],
            num_fields=2,
            field_telemetry={"pick": {"rows": 2}},
            parsed_json={},
        )

    def fake_load_preset(rel):
        return {
            "id": rel.replace(".json", ""),
            "title": rel,
            "schema": {"pick": {"type": "enum", "description": "d", "choices": ["ALPHA", "BETA"]}},
            "context": "ctx",
        }

    report = run_timing(
        "fake/model",
        presets=["one.json", "two.json", "three.json"],
        reps=5,
        load_engine_fn=fake_load_engine,
        run_fn=fake_run,
        load_preset_fn=fake_load_preset,
    )
    # 3 presets x 5 reps = 15 decide calls total.
    assert calls["n"] == 15
    assert set(report["presets"]) == {"one", "two", "three"}
    for block in report["presets"].values():
        agg = block["aggregate"]
        assert "total_ms" in agg and "median" in agg["total_ms"] and "p95" in agg["total_ms"]
        assert len(block["raw"]) == 5
        # prior_ms was 0 every rep -> median 0.
        assert agg["prior_ms"]["median"] == 0.0
        # rescored on odd reps of 5 -> [1,0,1,0,1], median 1 (reps restart
        # per preset, so no cross-preset counting contamination).
        assert agg["rescored_fields_count"]["median"] == 1
        assert agg["rescored_fields_count"]["max"] == 1
