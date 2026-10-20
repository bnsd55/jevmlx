"""Headline metrics count canonical answers only, failures included,
multi as sets.

Every fixture here is built through the REAL evalrun writer (``run_eval``),
so the tests exercise the prediction-line contract end to end — not
hand-made dicts.

Three regressions pinned:
1. Rotation lines (``permutation="rotN:<field>"``) are order probes. They
   must feed only the order-sensitivity metrics — never the headline
   accuracy / agreement counts.
2. A case-level failure must count every labelled field of that case as
   WRONG (error-inclusive denominator), not vanish from the denominator.
3. Multi-select correctness compares SETS: the same choices in a different
   order are correct.
"""

from __future__ import annotations

import json

from jevmlx import evalmetrics, evalrun


def _read_lines(path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _enum_case(case_id: str, label: str, source: str = "typesafe") -> dict:
    return {
        "id": case_id,
        "group_id": "g1",
        "source": source,
        "workflow": "triage",
        "schema": {
            "risk": {
                "type": "enum",
                "description": "risk level",
                "choices": ["LOW", "MED", "HIGH"],
            }
        },
        "context": f"ctx {case_id}",
        "labels": {"risk": label},
        "split": "train",
        "meta": {},
    }


def _run(cases, decide_fn, tmp_path, permutations="none"):
    evalrun.run_eval(
        cases,
        decide_fn,
        track="parallel",
        model="fake/model",
        permutations=permutations,
        out_dir=str(tmp_path),
        run_id="r1",
        chat_template="T",
    )
    return _read_lines(tmp_path / "predictions.jsonl")


def test_rotation_lines_never_enter_headline_accuracy(tmp_path):
    """Canonical + rotation lines: headline accuracy counts CANONICAL only.

    The decide_fn always picks the first choice of the order it sees, so the
    canonical answer is wrong while a rotation's remapped answer can be
    right — exactly the pollution that inflated the published number.
    """
    cases = [_enum_case("c1", "MED"), _enum_case("c2", "HIGH")]
    records = _run(
        cases,
        lambda schema_dict, context: {"risk": {"prediction": schema_dict["risk"]["choices"][0]}},
        tmp_path,
        permutations="rotations",
    )
    # 2 cases x (canonical + 2 rotations) = 6 lines; canonical answers wrong.
    assert len(records) == 6
    assert evalmetrics.field_accuracy(records) == 0.0
    agreement = evalmetrics.typesafe_agreement(records)
    assert agreement["overall"] == 0.0
    assert agreement["agreement_common_subset"] == 0.0
    # n_fields counts canonical lines only (2), not the 6 raw lines.
    assert agreement["n_fields"] == 2


def test_case_level_error_counts_every_labelled_field_wrong(tmp_path):
    """A case whose decide call fails must LOWER accuracy per labelled field.

    evalrun journals one failure line per labelled field (error-inclusive
    denominator); the metrics must count them as wrong.
    """
    cases = [
        _enum_case("c-ok", "LOW"),
        _enum_case("c-bad", "MED"),
        _enum_case("c-bad2", "HIGH"),
    ]

    def decide(schema_dict, context):
        if "c-bad" in context:
            raise ValueError("malformed context row")
        return {"risk": {"prediction": schema_dict["risk"]["choices"][0]}}

    records = _run(cases, decide, tmp_path)
    # c-ok correct; c-bad and c-bad2 fail wholesale: 1/3 fields correct.
    assert evalmetrics.field_accuracy(records) == 1 / 3
    assert evalmetrics.case_exact_match(records) == 1 / 3


def test_multi_same_set_different_order_is_correct(tmp_path):
    """Multi-select correctness compares sets, not lists."""
    case = {
        "id": "m1",
        "group_id": "g1",
        "source": "typesafe",
        "workflow": "triage",
        "schema": {
            "tags": {
                "type": "multi",
                "description": "tags",
                "choices": ["a", "b", "c"],
            }
        },
        "context": "ctx",
        "labels": {"tags": ["a", "b"]},
        "split": "train",
        "meta": {},
    }
    records = _run(
        [case],
        lambda s, c: {"tags": {"prediction": ["b", "a"], "type": "multi"}},
        tmp_path,
    )
    assert records[0]["correct"] is True
    assert evalmetrics.field_accuracy(records) == 1.0
