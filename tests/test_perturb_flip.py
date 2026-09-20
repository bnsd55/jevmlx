"""Fix: perturbation_flip_rate was None on perturbed combos.

Root cause: run_eval carries the ``perturbation`` key to prediction lines
only when ``carry_perturbation=True``, but the CLI never set it. So even on
perturbed datasets (where cases carry ``meta.perturbation``), the prediction
lines lacked the key, and perturbation_flip_rate returned None (no pairs).

Fix: the CLI detects perturbation metadata in the cases and passes
``carry_perturbation=True`` to run_eval.

This test runs the FULL path: a small perturbed jsonl (original + #p1..#p3
variants) through run_eval with a fake decide_fn, then compute_metrics, and
asserts perturbation_flip_rate is a float (not None).
"""

from __future__ import annotations

import json
from pathlib import Path

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
                "context": "The applicant pays late.",  # perturbed text doesn't matter for the fake
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


def test_perturbation_flip_rate_is_float_with_carry(tmp_path):
    """End-to-end: perturbed jsonl -> run_eval (carry_perturbation=True) ->
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
        carry_perturbation=True,
    )

    # Load the predictions and compute metrics.
    preds_path = tmp_path / "out" / "predictions.jsonl"
    records = [json.loads(line) for line in preds_path.read_text().splitlines() if line.strip()]

    # Every variant line carries the perturbation key.
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


def test_perturbation_flip_rate_none_without_carry(tmp_path):
    """Without carry_perturbation, the key is absent and the metric returns None."""
    cases_path = _perturbed_cases(tmp_path)
    cases = evalrun.load_cases(cases_path)

    decide_fn = _fake_decide_fn_all_flip()
    evalrun.run_eval(
        cases,
        decide_fn,
        track="parallel",
        model="fake",
        out_dir=str(tmp_path / "out_nocarry"),
        carry_perturbation=False,  # the bug: key never carried
    )

    preds_path = tmp_path / "out_nocarry" / "predictions.jsonl"
    records = [json.loads(line) for line in preds_path.read_text().splitlines() if line.strip()]

    # No line carries the perturbation key.
    assert all("perturbation" not in r for r in records)
    # The metric returns None (no pairs).
    assert perturbation_flip_rate(records) is None


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
        carry_perturbation=True,
    )

    preds_path = tmp_path / "out_noflip" / "predictions.jsonl"
    records = [json.loads(line) for line in preds_path.read_text().splitlines() if line.strip()]

    rate = perturbation_flip_rate(records)
    assert rate == 0.0


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
        carry_perturbation=True,
    )

    preds_path = tmp_path / "out_partial" / "predictions.jsonl"
    records = [json.loads(line) for line in preds_path.read_text().splitlines() if line.strip()]

    rate = perturbation_flip_rate(records)
    assert rate == pytest.approx(2 / 3)


def _detect_perturbation(cases: list[dict]) -> bool:
    """The detection expression the CLI uses (mirrors jevmlx/cli.py)."""
    return any((c.get("meta") or {}).get("perturbation") is not None for c in cases)


def test_detect_perturbation_flag_perturbed():
    """The CLI's detection returns True when any case has meta.perturbation."""
    cases = [
        {"id": "c1", "meta": {}},
        {"id": "c1#p1", "meta": {"perturbation": "ws"}},
    ]
    assert _detect_perturbation(cases) is True


def test_detect_perturbation_flag_plain():
    """The CLI's detection returns False for a non-perturbed dataset."""
    cases = [{"id": "c1", "meta": {}}, {"id": "c2", "meta": {}}]
    assert _detect_perturbation(cases) is False


def test_detect_perturbation_flag_empty_meta():
    """Cases with no meta key at all are handled safely (False)."""
    cases = [{"id": "c1"}, {"id": "c2"}]
    assert _detect_perturbation(cases) is False
