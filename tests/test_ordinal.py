"""W6-B1: ordered-enum telemetry (engine, API, metrics).

Unit tests on finalize_scalar_evidence (the ONE finalizer), the public
FieldResult.ordinal record, and the ordinal metrics — no live model.
"""

from __future__ import annotations

import math

import pytest
from conftest import make_engine_result, make_field_semantics

from jevmlx.api import (
    Ordered,
    OrdinalFieldRecord,
    _build_field_results,
    schema_from_model,
)
from jevmlx.engine import ScalarEvidence, finalize_scalar_evidence, ordinal_telemetry
from jevmlx.evalmetrics import ordinal_confusion, ordinal_mae, ordinal_mae_expected
from jevmlx.schema import StructuredSchema

# ---- the spec check: p=[.2, .8] on 2 levels --------------------------------


def _evidence(probs: list[float]) -> ScalarEvidence:
    """Evidence whose finalized distribution (T=1, no prior) is `probs`."""
    return ScalarEvidence(
        choices=tuple(str(i) for i in range(len(probs))),
        log_scores_raw=tuple(math.log(p) for p in probs),
        legal_mass_logs=tuple(0.0 for _ in probs),
        source_shape="batch1",
    )


def test_spec_p02_p08_expected_index_and_variance():
    """The W6-B1 spec check: p=[.2, .8] on 2 levels gives expected_index
    0.8 and variance 0.16."""
    decision, rescored = finalize_scalar_evidence(
        _evidence([0.2, 0.8]),
        prior_entry=None,
        temperature=1.0,
        ordered=True,
    )
    assert rescored is False
    ord_record = decision.ordinal
    assert ord_record is not None
    assert ord_record.argmax_level == 1
    assert ord_record.expected_index == pytest.approx(0.8)
    assert ord_record.variance == pytest.approx(0.16)
    assert ord_record.expected_score_normalized == pytest.approx(0.8)


def test_ordinal_telemetry_three_levels():
    """p=[.1, .2, .7] on 3 levels: E = 0*.1 + 1*.2 + 2*.7 = 1.6;
    var = .1*2.56 + .2*.36 + .7*.16 = .256 + .072 + .112 = .44; norm 0.8."""
    t = ordinal_telemetry([0.1, 0.2, 0.7])
    assert t.argmax_level == 2
    assert t.expected_index == pytest.approx(1.6)
    assert t.variance == pytest.approx(0.44)
    assert t.expected_score_normalized == pytest.approx(0.8)


def test_ordinal_telemetry_single_level():
    t = ordinal_telemetry([1.0])
    assert t.argmax_level == 0
    assert t.expected_index == 0.0
    assert t.variance == 0.0
    assert t.expected_score_normalized == 1.0  # degenerate scale


def test_unordered_field_has_ordinal_none():
    """The decided value stays the winning level; ordinal=None without the
    ordered flag (no behavior change for existing fields)."""
    decision, _ = finalize_scalar_evidence(
        _evidence([0.2, 0.8]),
        prior_entry=None,
        temperature=1.0,
    )
    assert decision.value == "1"  # winning level, a plain string
    assert decision.ordinal is None


def test_ordered_field_value_stays_the_winning_level():
    """NO new public field type: ordered=True only adds telemetry."""
    decision, _ = finalize_scalar_evidence(
        _evidence([0.3, 0.5, 0.2]),
        prior_entry=None,
        temperature=1.0,
        ordered=True,
    )
    assert decision.value == "1"  # still the argmax choice string
    assert decision.ordinal is not None
    assert decision.ordinal.argmax_level == 1
    assert decision.ordinal.expected_index == pytest.approx(0 * 0.3 + 1 * 0.5 + 2 * 0.2)


def test_dependency_pass_threads_ordered():
    """The dependency/oracle finalizer call carries the ordered flag too
    (both call sites pass fdef.ordered)."""
    # The flag is a parameter (the finalizer is the one place the record is
    # derived); the two call sites pass fdef.ordered.
    engine_src = open("jevmlx/engine.py").read()
    assert engine_src.count("ordered=fdef.ordered") >= 2


# ---- the public API record ---------------------------------------------------


def _telemetry(ordinal: dict | None = None) -> dict:
    entry = {
        "value": "1",
        "type": "enum",
        "probability": 0.8,
        "cardinality": 2,
        "log_scores": {"0": math.log(0.2), "1": math.log(0.8)},
        "top_choices": [
            {"choice": "1", "probability": 0.8},
            {"choice": "0", "probability": 0.2},
        ],
        "rows": 2,
        "tie": False,
        "rescored": False,
        "legal_mass": 1.0,
        "legal_mass_logs": {"0": 0.0, "1": 0.0},
        "semantics": make_field_semantics(),
    }
    if ordinal is not None:
        entry["ordinal"] = ordinal
    return entry


def test_field_result_ordinal_record_from_telemetry():
    result = make_engine_result(
        fields={
            "level": _telemetry(
                {
                    "argmax_level": 1,
                    "expected_index": 0.8,
                    "variance": 0.16,
                    "expected_score_normalized": 0.8,
                }
            )
        }
    )
    fields = _build_field_results(result, "slots")
    record = fields["level"].ordinal
    assert isinstance(record, OrdinalFieldRecord)
    assert record.expected_index == pytest.approx(0.8)
    assert record.variance == pytest.approx(0.16)


def test_field_result_ordinal_none_for_unordered():
    result = make_engine_result(fields={"level": _telemetry(None)})
    fields = _build_field_results(result, "slots")
    assert fields["level"].ordinal is None


def test_ordinal_telemetry_ignores_none():
    """A None ordinal in telemetry (should not happen) does not produce a
    record — additive contract, absent == unordered."""
    entry = _telemetry(None)
    entry["ordinal"] = None
    result = make_engine_result(fields={"level": entry})
    fields = _build_field_results(result, "slots")
    assert fields["level"].ordinal is None


# ---- the schema flag ----------------------------------------------------------


def test_schema_dict_ordered_flag_roundtrip():
    schema = StructuredSchema(
        {"level": {"type": "enum", "description": "d", "choices": ["0", "1", "2"], "ordered": True}}
    )
    assert schema.fields["level"].ordered is True


def test_schema_dict_unordered_default():
    schema = StructuredSchema({"pick": {"type": "enum", "description": "d", "choices": ["a", "b"]}})
    assert schema.fields["pick"].ordered is False


def test_multi_cannot_be_ordered():
    with pytest.raises(ValueError, match="multi fields cannot be ordered"):
        StructuredSchema(
            {"tags": {"type": "multi", "description": "d", "choices": ["a", "b"], "ordered": True}}
        )


def test_boolean_cannot_be_ordered():
    with pytest.raises(ValueError, match="boolean fields cannot be ordered"):
        StructuredSchema({"flag": {"type": "boolean", "description": "d", "ordered": True}})


def test_pydantic_ordered_dict_flag():
    from typing import Literal

    from pydantic import BaseModel, Field

    class M(BaseModel):
        level: Literal["0", "1", "2"] = Field(description="l", json_schema_extra={"ordered": True})

    schema = schema_from_model(M)
    assert schema["level"]["ordered"] is True


def test_pydantic_ordered_marker():
    from typing import Literal

    from pydantic import BaseModel, Field

    class M(BaseModel):
        level: Literal["0", "1", "2"] = Field(description="l", json_schema_extra=Ordered())

    assert schema_from_model(M)["level"]["ordered"] is True


def test_pydantic_unordered_default():
    from typing import Literal

    from pydantic import BaseModel, Field

    class M(BaseModel):
        pick: Literal["a", "b"] = Field(description="l")

    assert schema_from_model(M)["pick"]["ordered"] is False


def test_pydantic_bad_ordered_flag_raises():
    from typing import Literal

    from pydantic import BaseModel, Field

    class M(BaseModel):
        level: Literal["0", "1"] = Field(description="l", json_schema_extra={"ordered": "yes"})

    with pytest.raises(TypeError, match="ordered"):
        schema_from_model(M)


# ---- the metrics ---------------------------------------------------------------


def _line(field: str, label, prediction, expected, valid=True) -> dict:
    """One ordered prediction line (the evalrun contract)."""
    return {
        "case_id": f"c-{field}-{label}-{prediction}-{expected}",
        "field": field,
        "label": label,
        "prediction": prediction,
        "valid": valid,
        "ordinal_choices": ["0", "1", "2", "3", "4"],
        "ordinal": {
            "argmax_level": prediction,
            "expected_index": expected,
            "variance": 0.0,
            "expected_score_normalized": expected / 4,
        },
    }


def test_ordinal_mae_hand_made():
    """The hand-made set: gold 1 -> pred 2 (gap 1), gold 0 -> pred 0
    (gap 0) => MAE 0.5. An unordered field in the same set is ignored."""
    records = [
        _line("sentiment", "1", "2", 1.7),
        _line("sentiment", "0", "0", 0.3),
        {"field": "topic", "label": "a", "prediction": "b", "valid": True},
    ]
    mae = ordinal_mae(records)
    assert mae == {"sentiment": 0.5}


def test_ordinal_mae_expected_hand_made():
    """Soft MAE: |E[gold] - gold| per line: |1.7-1| = .7, |0.3-0| = .3
    => 0.5."""
    records = [
        _line("sentiment", "1", "2", 1.7),
        _line("sentiment", "0", "0", 0.3),
    ]
    assert ordinal_mae_expected(records) == {"sentiment": pytest.approx(0.5)}


def test_ordinal_mae_invalid_prediction_counts_worst():
    """Invalid => wrong: an off-scale prediction sits at the worst level
    distance (len-1 = 4)."""
    records = [_line("s", "0", None, 0.0, valid=False)]
    assert ordinal_mae(records) == {"s": 4.0}


def test_ordinal_confusion_matrix():
    records = [
        _line("s", "1", "2", 1.7),
        _line("s", "1", "1", 1.0),
        _line("s", "0", "0", 0.3),
    ]
    matrix = ordinal_confusion(records)
    assert matrix == {
        "s": {
            "1": {"2": 1, "1": 1},
            "0": {"0": 1},
        }
    }


def test_unordered_fields_never_enter_ordinal_metrics():
    records = [{"field": "topic", "label": "a", "prediction": "b", "valid": True}]
    assert ordinal_mae(records) == {}
    assert ordinal_mae_expected(records) == {}
    assert ordinal_confusion(records) == {}


def test_compute_metrics_includes_ordinal():
    from jevmlx.evalmetrics import compute_metrics

    records = [_line("s", "1", "2", 1.7)]
    metrics = compute_metrics(records)
    assert "ordinal_mae" in metrics and "ordinal_mae_expected" in metrics
    assert "ordinal_confusion" in metrics


def test_ordinal_mae_missing_label_skipped():
    records = [_line("s", None, "2", 1.7)]
    assert ordinal_mae(records) == {}


# ---- evalrun contract: the ordinal keys ride prediction lines -----------------


def test_evalrun_prediction_lines_carry_ordinal_keys(tmp_path):
    """run_eval writes ordinal_choices + ordinal on ordered-enum lines;
    unordered lines stay clean (results contract v2 additive)."""
    import json

    from jevmlx import evalrun

    cases = [
        {
            "id": "wf/case-1",
            "group_id": "g1",
            "source": "quality-eval",
            "workflow": None,
            "schema": {
                "level": {
                    "type": "enum",
                    "description": "sentiment level",
                    "choices": ["0", "1", "2"],
                    "ordered": True,
                },
                "flag": {"type": "boolean", "description": "is it urgent"},
            },
            "context": "customer text",
            "labels": {"level": "1", "flag": True},
            "split": "train",
            "meta": {},
        }
    ]

    def parallel_decide(schema_dict, context):
        return {
            "level": {
                "prediction": "1",
                "probability": 0.8,
                "type": "enum",
                "ordered": True,
                "ordinal_choices": ["0", "1", "2"],
                "ordinal": {
                    "argmax_level": 1,
                    "expected_index": 0.8,
                    "variance": 0.16,
                    "expected_score_normalized": 0.4,
                },
            },
            "flag": {"prediction": True, "probability": 0.9, "type": "boolean"},
            "_meta": {
                "latency_ms": 10.0,
                "rows": 3,
                "passes": 1,
                "prior_ms": 0.0,
                "prefill_ms": 5.0,
                "plan_compile_ms": 0.1,
                "cache_broadcast_ms": 0.2,
                "suffix_eval_ms": 4.0,
                "lm_head_gather_ms": 0.3,
                "second_pass_ms": 0.0,
                "total_ms": 10.0,
                "peak_active_bytes": 2048,
                "padded_token_positions": 24,
                "rescored_fields_count": 0,
                "rerun_fields_count": 0,
                "num_fields": 2,
            },
        }

    evalrun.run_eval(
        cases,
        parallel_decide,
        track="parallel",
        model="fake/model",
        out_dir=str(tmp_path),
        run_id="r-ordinal",
    )
    lines = [
        json.loads(line)
        for line in (tmp_path / "predictions.jsonl").read_text().splitlines()
        if line.strip()
    ]
    level_lines = [r for r in lines if r["field"] == "level"]
    flag_lines = [r for r in lines if r["field"] == "flag"]
    assert level_lines and flag_lines
    assert level_lines[0]["ordinal_choices"] == ["0", "1", "2"]
    assert level_lines[0]["ordinal"]["expected_index"] == 0.8
    assert "ordinal" not in flag_lines[0] and "ordinal_choices" not in flag_lines[0]


# ---- review fixes F2/F3 -------------------------------------------------------


def test_dict_schema_rejects_non_bool_ordered():
    """F2: the dict-schema path is as strict as the Pydantic path — a
    truthy non-bool ordered raises TypeError, never silently orders."""
    with pytest.raises(TypeError, match="ordered must be a bool"):
        StructuredSchema(
            {"level": {"type": "enum", "description": "d", "choices": ["0", "1"], "ordered": "yes"}}
        )
    with pytest.raises(TypeError, match="ordered must be a bool"):
        StructuredSchema(
            {"level": {"type": "enum", "choices": ["0", "1"], "description": "d", "ordered": 1}}
        )


def test_cardinality_one_ordered_enum_emits_degenerate_record():
    """F3: an ordered enum with ONE choice still emits the ordinal record
    (the degenerate scale) — the key's presence is guaranteed."""
    from conftest import make_engine
    from test_engine_fake import BiasedFakeModel

    from jevmlx.engine import run_parallel_generation

    model = BiasedFakeModel(vocab_size=64, winner_token=ord("A") % 60)
    schema = StructuredSchema(
        {"level": {"type": "enum", "description": "d", "choices": ["only"], "ordered": True}}
    )
    result = run_parallel_generation(make_engine(model), "ctx", schema)
    telemetry = result["field_telemetry"]["level"]
    assert telemetry["ordinal"] == {
        "argmax_level": 0,
        "expected_index": 0.0,
        "variance": 0.0,
        "expected_score_normalized": 1.0,
    }


def test_ordinal_lines_strict_pairing():
    """F3: ordinal_choices without ordinal is a contract violation — raise,
    never a silently-skipped row."""
    from jevmlx.evalmetrics import _ordinal_lines

    broken = [{"field": "s", "label": "1", "prediction": "1", "ordinal_choices": ["0", "1"]}]
    with pytest.raises(ValueError, match="no 'ordinal' telemetry"):
        _ordinal_lines(broken)
    # Unordered lines (no key at all) still pass through untouched.
    assert _ordinal_lines([{"field": "t", "label": "a", "prediction": "b"}]) == []
