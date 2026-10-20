"""Legacy combo folders journal a case-level failure as ONE
``field="_error"`` line with no label. Read-time expansion (in
``load_predictions``, the shared loader) must count that failed case as
wrong for every labelled field of the case — otherwise committed folders
runs silently drop failures from every accuracy denominator.

The fixture rewrites a REAL run_eval-produced predictions file into the
legacy shape (one ``_error`` line replacing the case's canonical lines) —
the legacy format predates the per-field failure writer, so it cannot be
produced by today's evalrun.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

from jevmlx import evalmetrics, evalrun


def _case(case_id: str, label: str) -> dict:
    return {
        "id": case_id,
        "group_id": "g1",
        "source": "typesafe",
        "workflow": "triage",
        "schema": {
            "risk": {
                "type": "enum",
                "description": "risk level",
                "choices": ["LOW", "MED", "HIGH"],
            },
            "severity": {
                "type": "enum",
                "description": "severity",
                "choices": ["low", "high"],
            },
        },
        "context": f"ctx {case_id}",
        "labels": {"risk": label, "severity": "high"},
        "split": "train",
        "meta": {},
    }


def _write_legacy_combo(tmp_path: Path) -> Path:
    """Run a real eval, then rewrite one case's lines to the legacy shape."""
    combo = tmp_path / "parallel-labels-typesafe"
    cases = [_case("c-ok", "LOW"), _case("c-fail", "MED")]
    evalrun.run_eval(
        cases,
        lambda schema_dict, context: {
            "risk": {"prediction": schema_dict["risk"]["choices"][0]},  # LOW == label
            "severity": {"prediction": schema_dict["severity"]["choices"][1]},  # high == label
        },
        track="parallel",
        model="fake/model",
        out_dir=str(combo),
        run_id="r1",
        chat_template="T",
    )
    records = evalmetrics.load_predictions(combo / "predictions.jsonl")
    # Rewrite c-fail as the legacy single _error line (the old writer shape).
    legacy = []
    for r in records:
        if r["case_id"] == "c-fail":
            if r.get("permutation") not in (None, "canonical"):
                continue  # rotations of the failed case never committed
            if legacy and legacy[-1]["field"] == "_error":
                continue
            legacy.append(
                {
                    **r,
                    "field": "_error",
                    "label": None,
                    "prediction": None,
                    "valid": False,
                    "correct": False,
                    "error": "RuntimeError:metal allocation failed",
                }
            )
        else:
            legacy.append(r)
    with gzip.open(combo / "predictions.jsonl.gz", "wt", encoding="utf-8") as f:
        for r in legacy:
            f.write(json.dumps(r) + "\n")
    (combo / "predictions.jsonl").unlink()
    return combo


def test_legacy_case_error_counts_every_labelled_field_wrong(tmp_path):
    combo = _write_legacy_combo(tmp_path)
    records = evalmetrics.load_predictions(combo / "predictions.jsonl.gz")

    # The _error marker is replaced by one line per labelled field (2).
    assert not any(r["field"] == "_error" for r in records)
    failed = [r for r in records if r["case_id"] == "c-fail"]
    assert sorted(r["field"] for r in failed) == ["risk", "severity"]
    assert all(r["correct"] is False and r["label"] is None for r in failed)

    # 2 correct canonical fields (c-ok) + 2 failed fields = 2/4, not 2/2.
    assert evalmetrics.field_accuracy(records) == 0.5
    assert evalmetrics.case_exact_match(records) == 0.5
    agreement = evalmetrics.typesafe_agreement(records)
    assert agreement["agreement_common_subset"] == 0.5
    assert agreement["n_fields"] == 4

    # The unobservable failure contributes no majority class: the baseline
    # for both fields stays 1/2 (one LOW-label case of two... the failed
    # case's true labels are unknown offline), never a "null" class.
    baseline = evalmetrics.majority_class_baseline(records)
    assert set(baseline) == {"risk", "severity"}
    assert all(0.0 < v <= 1.0 for v in baseline.values())


def test_load_predictions_reads_gzip(tmp_path):
    combo = _write_legacy_combo(tmp_path)
    # Decompress the same lines to plain jsonl: the loader must handle both
    # paths and return identical records.
    import gzip as _gzip

    plain_path = combo / "copy.jsonl"
    with _gzip.open(combo / "predictions.jsonl.gz", "rt", encoding="utf-8") as f:
        plain_path.write_text(f.read(), encoding="utf-8")
    assert evalmetrics.load_predictions(
        combo / "predictions.jsonl.gz"
    ) == evalmetrics.load_predictions(plain_path)
    assert any(
        r["field"] == "risk" and r["case_id"] == "c-fail"
        for r in evalmetrics.load_predictions(plain_path)
    )
