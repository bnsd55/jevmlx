"""Fix: perturbation_flip_rate was None on perturbed combos.

Root cause: run_eval carried the ``perturbation`` key to prediction lines
only when ``carry_perturbation=True``, but no caller set it (the CLI and
bench._run_one both called run_eval without it). So even on perturbed
datasets (where cases carry ``meta.perturbation``), the prediction lines
lacked the key, and perturbation_flip_rate returned None (no pairs).

Fix at the OWNING layer: run_eval auto-detects perturbation metadata
(any case with a non-None meta.perturbation) and carries the key itself.
The carry_perturbation parameter is deleted — no caller needs to remember
a flag. Non-perturbed datasets are unaffected (no key added).

Tests:
- end-to-end fake-engine eval through run_eval + compute_metrics
- all-flip / partial-flip / no-flip rates
- non-perturbed dataset: no key added (contract unchanged)
- bench._run_one: report.json has perturbation_flip_rate (the M5 path)
- detection logic (perturbed / plain / empty-meta)
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from jevmlx import evalrun
from jevmlx.evalmetrics import compute_metrics, perturbation_flip_rate


def _perturbed_cases(tmp_path: Path) -> str:
    """A small perturbed jsonl: 1 original + 3 variants (ws, preamble, numfmt).

    The variants share the original's group_id; meta.perturbation names the
    kind. The original has no meta.perturbation (None).
    """
    cases = [
        {
            "id": "case-001",
            "group_id": "case-001",
            "schema": {"risk": {"type": "enum", "choices": ["LOW", "HIGH"], "description": "d"}},
            "context": "The applicant pays late.",
            "labels": {"risk": "LOW"},
            "split": "test",
            "meta": {},
        },
    ]
    for k, kind in enumerate(["ws", "preamble", "numfmt"], start=1):
        cases.append(
            {
                "id": f"case-001#p{k}",
                "group_id": "case-001",
                "schema": {
                    "risk": {"type": "enum", "choices": ["LOW", "HIGH"], "description": "d"}
                },
                "context": "The applicant pays late.",
                "labels": {"risk": "LOW"},
                "split": "test",
                "meta": {"perturbation": kind},
            }
        )
    path = tmp_path / "perturbed.jsonl"
    path.write_text("\n".join(json.dumps(c) for c in cases) + "\n", encoding="utf-8")
    return str(path)


def _fake_decide_fn_all_flip():
    """A decide_fn that flips ALL variants (original=LOW, variants=HIGH).

    Uses a per-call counter: call 1 is the original, calls 2+ are variants.
    """

    call_count = [0]

    def decide(schema_dict, context, constraints=None, oracle_overrides=None):
        call_count[0] += 1
        is_variant = call_count[0] > 1
        pred = "HIGH" if is_variant else "LOW"
        return {"risk": {"prediction": pred, "probability": 0.9, "valid": True}}

    return decide


def _fake_decide_fn_partial_flip():
    """A decide_fn that flips only 2 of 3 variants (call 3 stays LOW)."""

    call_count = [0]

    def decide(schema_dict, context, constraints=None, oracle_overrides=None):
        call_count[0] += 1
        # Calls 2,4 flip (HIGH); call 3 stays (LOW).
        is_variant = call_count[0] in (2, 4)
        pred = "HIGH" if is_variant else "LOW"
        return {"risk": {"prediction": pred, "probability": 0.9, "valid": True}}

    return decide


def test_perturbation_flip_rate_is_float(tmp_path):
    """End-to-end: perturbed jsonl -> run_eval (auto-carry) ->
    compute_metrics -> perturbation_flip_rate is a float, not None."""
    cases_path = _perturbed_cases(tmp_path)
    cases = evalrun.load_cases(cases_path)

    # All 3 variants flip (original=LOW, all variants=HIGH -> rate=1.0).
    decide_fn = _fake_decide_fn_all_flip()
    evalrun.run_eval(
        cases,
        decide_fn,
        track="parallel",
        model="fake",
        out_dir=str(tmp_path / "out"),
    )

    # Load the predictions and compute metrics.
    preds_path = tmp_path / "out" / "predictions.jsonl"
    records = [json.loads(line) for line in preds_path.read_text().splitlines() if line.strip()]

    # Every variant line carries the perturbation key (auto-detected).
    variant_lines = [r for r in records if r.get("perturbation") is not None]
    assert len(variant_lines) == 3
    # The original line has perturbation = None.
    original_lines = [r for r in records if r.get("perturbation") is None]
    assert len(original_lines) == 1

    # The metric finds pairs and returns a float.
    rate = perturbation_flip_rate(records)
    assert rate is not None
    assert isinstance(rate, float)
    # All 3 variants flipped (HIGH vs original LOW).
    assert rate == pytest.approx(1.0)

    # compute_metrics includes it in the result dict.
    metrics = compute_metrics(records)
    assert "perturbation_flip_rate" in metrics
    assert metrics["perturbation_flip_rate"] == pytest.approx(1.0)


def test_perturbation_flip_rate_partial_flip(tmp_path):
    """When only 2 of 3 variants flip, the rate is 2/3."""
    cases_path = _perturbed_cases(tmp_path)
    cases = evalrun.load_cases(cases_path)

    decide_fn = _fake_decide_fn_partial_flip()
    evalrun.run_eval(
        cases,
        decide_fn,
        track="parallel",
        model="fake",
        out_dir=str(tmp_path / "out_partial"),
    )

    preds_path = tmp_path / "out_partial" / "predictions.jsonl"
    records = [json.loads(line) for line in preds_path.read_text().splitlines() if line.strip()]

    rate = perturbation_flip_rate(records)
    assert rate == pytest.approx(2 / 3)


def test_perturbation_flip_rate_zero_when_no_flips(tmp_path):
    """When no variant flips, the rate is 0.0 (not None)."""
    cases_path = _perturbed_cases(tmp_path)
    cases = evalrun.load_cases(cases_path)

    # All predictions are LOW (no flips).
    def no_flip_decide(schema_dict, context, constraints=None, oracle_overrides=None):
        return {"risk": {"prediction": "LOW", "probability": 0.9, "valid": True}}

    decide_fn = no_flip_decide
    evalrun.run_eval(
        cases,
        decide_fn,
        track="parallel",
        model="fake",
        out_dir=str(tmp_path / "out_noflip"),
    )

    preds_path = tmp_path / "out_noflip" / "predictions.jsonl"
    records = [json.loads(line) for line in preds_path.read_text().splitlines() if line.strip()]

    rate = perturbation_flip_rate(records)
    assert rate == 0.0


def test_non_perturbed_dataset_no_key_added(tmp_path):
    """A non-perturbed dataset does NOT get the perturbation key (contract
    unchanged). This is the byte-identical-to-main guarantee."""
    cases = [
        {
            "id": "c1",
            "group_id": "c1",
            "schema": {"risk": {"type": "enum", "choices": ["LOW", "HIGH"], "description": "d"}},
            "context": "ctx",
            "labels": {"risk": "LOW"},
            "split": "test",
            "meta": {},
        }
    ]
    cases_path = tmp_path / "plain.jsonl"
    cases_path.write_text(json.dumps(cases[0]) + "\n", encoding="utf-8")

    def decide(schema_dict, context, constraints=None, oracle_overrides=None):
        return {"risk": {"prediction": "LOW", "probability": 0.9, "valid": True}}

    evalrun.run_eval(
        evalrun.load_cases(str(cases_path)),
        decide,
        track="parallel",
        model="fake",
        out_dir=str(tmp_path / "out_plain"),
    )

    preds_path = tmp_path / "out_plain" / "predictions.jsonl"
    records = [json.loads(line) for line in preds_path.read_text().splitlines() if line.strip()]

    # No line carries the perturbation key.
    assert all("perturbation" not in r for r in records)
    # The metric returns None (no pairs).
    assert perturbation_flip_rate(records) is None


def test_bench_run_one_report_has_perturbation_flip_rate(tmp_path):
    """The M5 path: bench._run_one on a perturbed jsonl produces a report.json
    with perturbation_flip_rate (not None / not missing).

    This is the regression test for the original M5 finding: bench._run_one
    calls run_eval directly, and before the fix it never carried the
    perturbation key, so the report had a dash.
    """
    from jevmlx.bench import _run_one

    cases_path = _perturbed_cases(tmp_path)
    combo_dir = tmp_path / "combo"

    # Mock the engine load + decide_fn so we don't need a real MLX model.
    # _run_one calls load_engine(model) then parallel_decide_fn(engine, ...).
    # We patch both so the fake decide_fn (all-flip) is used.
    fake_engine = type(
        "FakeEngine", (), {"tokenizer": type("FakeTok", (), {"chat_template": None})()}
    )()
    fake_decide = _fake_decide_fn_all_flip()

    with (
        patch("jevmlx.engine.load_engine", lambda model: fake_engine),
        patch("jevmlx.bench.parallel_decide_fn", lambda engine, scoring="slots", **kw: fake_decide),
    ):
        _run_one(
            model="fake",
            track="parallel",
            scorer="slots",
            jsonl=Path(cases_path),
            combo_dir=combo_dir,
        )

    # The report.json has perturbation_flip_rate as a float.
    report_path = combo_dir / "report.json"
    assert report_path.exists()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert "perturbation_flip_rate" in report["metrics"]
    assert isinstance(report["metrics"]["perturbation_flip_rate"], float)
    # All 3 variants flipped -> rate = 1.0.
    assert report["metrics"]["perturbation_flip_rate"] == pytest.approx(1.0)


def _detect_perturbation(cases: list[dict]) -> bool:
    """The detection expression run_eval uses (mirrors jevmlx/evalrun.py)."""
    return any((c.get("meta") or {}).get("perturbation") is not None for c in cases)


def test_detect_perturbation_flag_perturbed():
    """run_eval's detection returns True when any case has meta.perturbation."""
    cases = [
        {"id": "c1", "meta": {}},
        {"id": "c1#p1", "meta": {"perturbation": "ws"}},
    ]
    assert _detect_perturbation(cases) is True


def test_detect_perturbation_flag_plain():
    """run_eval's detection returns False for a non-perturbed dataset."""
    cases = [{"id": "c1", "meta": {}}, {"id": "c2", "meta": {}}]
    assert _detect_perturbation(cases) is False


def test_detect_perturbation_flag_empty_meta():
    """Cases with no meta key at all are handled safely (False)."""
    cases = [{"id": "c1"}, {"id": "c2"}]
    assert _detect_perturbation(cases) is False
