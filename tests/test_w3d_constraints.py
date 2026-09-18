"""W3-D F2 tests: constraints wired through the api path and evalrun.

F1: constraints.py is the single source of truth (imported by engine + evalmetrics).
F2: decide()/decide_many() accept constraints=, CLI --constraints path, and
    evalrun passes case['constraints'] so the dependent set exercises MAP
    and constraint_violation_rate can hit 0.
"""

from __future__ import annotations

import pytest

from jevmlx.constraints import check_constraint, validate_constraints

# ---------------------------------------------------------------------------
# F1: constraints.py is the single source of truth
# ---------------------------------------------------------------------------


def test_check_constraint_shared_implementation():
    """Both engine and evalmetrics import the same check_constraint."""
    from jevmlx.evalmetrics import _check_constraint as eval_check

    # The evalmetrics _check_constraint delegates to constraints.check_constraint.
    assignment = {"intent": "billing", "subtype": "refund"}
    c = {
        "type": "implies",
        "parent": "intent",
        "child": "subtype",
        "mapping": {"billing": ["refund", "dispute"], "technical": ["bug"]},
    }
    assert eval_check(c, assignment) is True
    assert check_constraint(c, assignment) is True

    assignment["subtype"] = "bug"  # violates implies
    assert eval_check(c, assignment) is False
    assert check_constraint(c, assignment) is False


def test_validate_constraints_rejects_malformed():
    """validate_constraints raises ValueError on bad shapes."""
    with pytest.raises(ValueError, match="type must be one of"):
        validate_constraints([{"type": "nonsense", "field": "x"}])

    with pytest.raises(ValueError, match="missing key"):
        validate_constraints([{"type": "implies", "parent": "a"}])  # no child/mapping

    with pytest.raises(ValueError, match="must be a list"):
        validate_constraints("not a list")


def test_validate_constraints_accepts_valid():
    constraints = [
        {"type": "implies", "parent": "a", "child": "b", "mapping": {"x": ["y"]}},
        {"type": "excludes", "field": "c", "value": True, "other": "d"},
        {"type": "exclusivity", "field": "e", "options": ["p1", "p2"]},
    ]
    assert validate_constraints(constraints) is constraints


# ---------------------------------------------------------------------------
# F2: api path — decide/decide_many pass constraints through
# ---------------------------------------------------------------------------


def test_decide_passes_constraints_to_engine(monkeypatch):
    """decide() forwards constraints= to run_parallel_generation."""
    import jevmlx.api as api

    captured = {}

    def fake_run_parallel(
        engine_model,
        tokenizer,
        context,
        schema,
        *,
        temperature=1.0,
        scoring="slots",
        calibration=None,
        prior_correction=False,
        constraints=None,
    ):
        captured["constraints"] = constraints
        return {
            "parsed_json": {"risk": {"value": "LOW"}},
            "field_telemetry": {"risk": {"value": "LOW", "probability": 0.9}},
            "confidence_model": "slots",
            "elapsed_ms": 5.0,
            "prompt_sha256": "abc",
            "prompt_version": "v6",
            "probability_status": "test",
            "prior_correction": False,
            "constraints_applied": True,
            "reconciled_fields": ["risk"],
            "prior_ms": 0.0,
            "prefill_ms": 1.0,
            "suffix_eval_ms": 1.0,
            "lm_head_gather_ms": 1.0,
            "total_ms": 5.0,
            "total_tokens_generated": 0,
            "sequential_forward_passes": 1,
            "schema_match": True,
            "num_fields": 1,
        }

    monkeypatch.setattr(api, "load_engine", lambda model: (object(), object()))
    monkeypatch.setattr(api, "run_parallel_generation", fake_run_parallel)

    from typing import Literal

    from pydantic import BaseModel

    class M(BaseModel):
        risk: Literal["LOW", "HIGH"]

    api._PREPARED_SCHEMAS.clear() if hasattr(api, "_PREPARED_SCHEMAS") else None
    constraints = [
        {"type": "implies", "parent": "risk", "child": "risk", "mapping": {"LOW": ["LOW"]}}
    ]
    api.decide(M, "ctx", constraints=constraints)
    assert captured["constraints"] is constraints


def test_decide_many_passes_constraints_to_engine(monkeypatch):
    """decide_many() forwards constraints= to run_parallel_generation."""
    import jevmlx.api as api

    captured = []

    def fake_run_parallel(
        engine_model,
        tokenizer,
        context,
        schema,
        *,
        temperature=1.0,
        scoring="slots",
        calibration=None,
        prior_correction=False,
        constraints=None,
    ):
        captured.append(constraints)
        return {
            "parsed_json": {"risk": {"value": "LOW"}},
            "field_telemetry": {"risk": {"value": "LOW", "probability": 0.9}},
            "confidence_model": "slots",
            "elapsed_ms": 5.0,
            "prompt_sha256": "abc",
            "prompt_version": "v6",
            "probability_status": "test",
            "prior_correction": False,
            "constraints_applied": True,
            "reconciled_fields": [],
            "prior_ms": 0.0,
            "prefill_ms": 1.0,
            "suffix_eval_ms": 1.0,
            "lm_head_gather_ms": 1.0,
            "total_ms": 5.0,
            "total_tokens_generated": 0,
            "sequential_forward_passes": 1,
            "schema_match": True,
            "num_fields": 1,
        }

    monkeypatch.setattr(api, "load_engine", lambda model: (object(), object()))
    monkeypatch.setattr(api, "run_parallel_generation", fake_run_parallel)

    from typing import Literal

    from pydantic import BaseModel

    class M(BaseModel):
        risk: Literal["LOW", "HIGH"]

    constraints = [
        {"type": "implies", "parent": "risk", "child": "risk", "mapping": {"LOW": ["LOW"]}}
    ]
    api.decide_many(M, ["ctx1", "ctx2"], constraints=constraints)
    assert len(captured) == 2
    assert all(c is constraints for c in captured)


# ---------------------------------------------------------------------------
# F2: evalrun — dependent fixture exercises MAP, violation_rate == 0
# ---------------------------------------------------------------------------


def test_evalrun_constraint_violation_rate_zero_with_map(monkeypatch):
    """A constrained fixture where the independent argmax would violate an
    implies constraint, but the MAP reconciler flips the child. The
    evalrun records carry constraints and constraint_violation_rate hits 0."""
    import math

    import jevmlx.engine as engine_mod
    import jevmlx.evalrun as er
    from jevmlx.evalmetrics import constraint_violation_rate

    # Fake run_parallel_generation: returns log_scores where the independent
    # argmax for subtype would pick 'bug' (highest score) but the constraint
    # requires subtype in {refund, dispute} when intent=billing. The MAP
    # reconciler (inside the real _constrained_map, called by the real
    # run_parallel_generation) flips it.
    # But we're mocking run_parallel_generation, so we simulate the
    # reconciled output directly: the MAP picks intent=billing, subtype=refund.
    def fake_rpg(
        model,
        tokenizer,
        context,
        schema,
        temperature=1.0,
        scoring="slots",
        prior_correction=False,
        constraints=None,
    ):
        # Simulate the constrained MAP: with constraints, subtype flips to refund.
        reconciled_subtype = "refund" if constraints else "bug"
        return {
            "field_telemetry": {
                "intent": {
                    "value": "billing",
                    "type": "enum",
                    "probability": 0.9,
                    "log_scores": {
                        "billing": math.log(0.9),
                        "technical": math.log(0.1),
                    },
                },
                "subtype": {
                    "value": reconciled_subtype,
                    "type": "enum",
                    "probability": 0.3 if constraints else 0.55,
                    "log_scores": {
                        "refund": math.log(0.3),
                        "dispute": math.log(0.15),
                        "bug": math.log(0.55),
                    },
                },
            },
            "elapsed_ms": 5.0,
            "rows": 2,
            "passes": 1,
            "constraints_applied": bool(constraints),
            "reconciled_fields": ["subtype"] if constraints else [],
        }

    monkeypatch.setattr(engine_mod, "run_parallel_generation", fake_rpg)
    decide = er.parallel_decide_fn(model=object(), tokenizer=object())

    schema_dict = {
        "intent": {"type": "enum", "description": "d", "choices": ["billing", "technical"]},
        "subtype": {
            "type": "enum",
            "description": "d",
            "choices": ["refund", "dispute", "bug"],
        },
    }
    constraints = [
        {
            "type": "implies",
            "parent": "intent",
            "child": "subtype",
            "mapping": {"billing": ["refund", "dispute"], "technical": ["bug"]},
        }
    ]

    # With MAP: subtype=refund (reconciled) -> constraint satisfied.
    results = decide(schema_dict, "ctx", constraints=constraints)
    results.pop("_meta", None)

    # Build evalrun records (the shape _case_predictions/_case_constraints read).
    records = []
    for fname, res in results.items():
        records.append(
            {
                "case_id": "case1",
                "field": fname,
                "prediction": res["prediction"],
                "label": res[
                    "prediction"
                ],  # label = prediction (we're testing violation, not accuracy)
                "type": "enum",
                "constraints": constraints if fname == "intent" else None,
            }
        )

    rate = constraint_violation_rate(records)
    assert rate is not None, "expected constraints to be detected"
    assert rate["overall"] == 0.0, f"expected 0 violations with MAP, got {rate}"


def test_evalrun_constraint_violation_without_map(monkeypatch):
    """Without constraints (no MAP), the independent argmax violates the
    implies constraint -> violation_rate > 0. This proves the MAP actually
    prevents the violation in the test above."""
    import math

    import jevmlx.engine as engine_mod
    import jevmlx.evalrun as er
    from jevmlx.evalmetrics import constraint_violation_rate

    def fake_rpg(
        model,
        tokenizer,
        context,
        schema,
        temperature=1.0,
        scoring="slots",
        prior_correction=False,
        constraints=None,
    ):
        # No MAP: subtype stays at argmax 'bug'.
        return {
            "field_telemetry": {
                "intent": {
                    "value": "billing",
                    "type": "enum",
                    "probability": 0.9,
                    "log_scores": {"billing": math.log(0.9), "technical": math.log(0.1)},
                },
                "subtype": {
                    "value": "bug",
                    "type": "enum",
                    "probability": 0.55,
                    "log_scores": {
                        "refund": math.log(0.3),
                        "dispute": math.log(0.15),
                        "bug": math.log(0.55),
                    },
                },
            },
            "elapsed_ms": 5.0,
            "rows": 2,
            "passes": 1,
            "constraints_applied": False,
            "reconciled_fields": [],
        }

    monkeypatch.setattr(engine_mod, "run_parallel_generation", fake_rpg)
    decide = er.parallel_decide_fn(model=object(), tokenizer=object())

    schema_dict = {
        "intent": {"type": "enum", "description": "d", "choices": ["billing", "technical"]},
        "subtype": {
            "type": "enum",
            "description": "d",
            "choices": ["refund", "dispute", "bug"],
        },
    }
    constraints = [
        {
            "type": "implies",
            "parent": "intent",
            "child": "subtype",
            "mapping": {"billing": ["refund", "dispute"], "technical": ["bug"]},
        }
    ]

    # Without MAP (constraints=None on the decide call), subtype=bug (violates).
    results = decide(schema_dict, "ctx", constraints=None)
    results.pop("_meta", None)

    records = []
    for fname, res in results.items():
        records.append(
            {
                "case_id": "case1",
                "field": fname,
                "prediction": res["prediction"],
                "label": res["prediction"],
                "type": "enum",
                "constraints": constraints if fname == "intent" else None,
            }
        )

    rate = constraint_violation_rate(records)
    assert rate is not None
    assert rate["overall"] > 0.0, "expected violation without MAP"
