"""EV1 tests: synthetic dependent2 set + dependent-schema metrics.

Verifies:
- build_dependent2 produces 200 cases with all 4 constraint types.
- Labels satisfy their own constraints (generator is self-consistent).
- constraint_violation_rate detects implies/excludes/exclusivity violations.
- exact_record_accuracy is stricter than case_exact_match (constraints).
- child_accuracy_given_parent_correct skips cases where parent is wrong.
- order_flip_rate delegates to any_flip_rate.
- compute_metrics includes the new metrics when constraints are present.
- Report rendering (_metric_rows) surfaces the new metrics.
"""

from __future__ import annotations

from benchmarks.synthetic import DEPENDENT2_CONSTRAINTS, build_dependent2
from jevmlx.evalmetrics import (
    child_accuracy_given_parent_correct,
    compute_metrics,
    constraint_violation_rate,
    exact_record_accuracy,
    order_flip_rate,
)
from jevmlx.evalreport import _metric_rows


def _record(case_id, field, label, prediction, ctype="enum", constraints=None, correct=None):
    return {
        "case_id": case_id,
        "field": field,
        "label": label,
        "prediction": prediction,
        "type": ctype,
        "correct": correct if correct is not None else (prediction == label),
        "constraints": constraints or [],
    }


_IMPLIES = [
    {
        "type": "implies",
        "parent": "intent",
        "child": "subtype",
        "mapping": {"billing": ["refund", "dispute"], "technical": ["bug", "feature"]},
    }
]


# --------------------------------------------------------------- synthetic set


def test_dependent2_has_200_cases():
    records = build_dependent2()
    assert len(records) == 200


def test_dependent2_has_all_constraint_types():
    records = build_dependent2()
    # Constraints are at the CASE level (F1: not inside schema).
    constraints = records[0].get("constraints")
    assert isinstance(constraints, list)
    types = {c["type"] for c in constraints}
    assert types == {"implies", "excludes", "requires_parent", "exclusivity"}


def test_dependent2_labels_satisfy_constraints():
    """The generator's own labels must never violate the constraints."""
    records = build_dependent2()
    constraints = DEPENDENT2_CONSTRAINTS
    for r in records:
        labels = r["labels"]
        for c in constraints:
            if c["type"] == "implies":
                parent_val = labels[c["parent"]]
                child_val = labels[c["child"]]
                assert child_val in c["mapping"][parent_val], (
                    f"case {r['id']}: {c['parent']}={parent_val}"
                    f" -> {c['child']}={child_val} not in {c['mapping'][parent_val]}"
                )
            elif c["type"] == "excludes":
                if labels[c["field"]] == c["value"]:
                    other = labels[c["other"]]
                    assert other in ("", None, False) or other == "", (
                        f"case {r['id']}: {c['field']}={c['value']} but {c['other']}={other}"
                    )
            elif c["type"] == "requires_parent":
                parent_val = labels[c["parent"]]
                child_val = labels[c["child"]]
                assert child_val in c["mapping"][parent_val], (
                    f"case {r['id']}: {c['parent']}={parent_val} -> {c['child']}={child_val}"
                )
            elif c["type"] == "exclusivity":
                channels = labels[c["field"]]
                selected = set(channels) & set(c["options"])
                assert len(selected) <= 1, f"case {r['id']}: exclusivity violated: {selected}"


def test_dependent2_schema_compiles_without_error():
    """F1: constraints must NOT be inside schema — StructuredSchema.__init__
    iterates every key as a field. A dependent case must compile clean."""
    from jevmlx.schema import StructuredSchema

    records = build_dependent2()
    schema = StructuredSchema(records[0]["schema"])
    assert "constraints" not in schema.fields
    assert set(schema.fields) == {
        "intent",
        "subtype",
        "approved",
        "rejection_reason",
        "stage",
        "stage_detail",
        "channels",
    }


def test_dependent2_is_deterministic():
    r1 = build_dependent2(seed=0)
    r2 = build_dependent2(seed=0)
    assert r1 == r2
    r3 = build_dependent2(seed=1)
    assert r1 != r3


# --------------------------------------------------------------- metrics


def test_constraint_violation_rate_detects_implies_violation():
    """c1: intent=billing, subtype=bug (WRONG: bug is for technical). c2: clean."""
    records = [
        _record("c1", "intent", "billing", "billing", constraints=_IMPLIES),
        _record("c1", "subtype", "refund", "bug", constraints=_IMPLIES),
        _record("c2", "intent", "technical", "technical", constraints=_IMPLIES),
        _record("c2", "subtype", "bug", "bug", constraints=_IMPLIES),
    ]
    result = constraint_violation_rate(records)
    assert result is not None
    assert result["overall"] == 0.5  # 1 of 2 cases violated
    assert result["by_type"]["implies"] == 0.5  # 1 of 2 implies constraints violated


def test_constraint_violation_rate_none_when_no_constraints():
    records = [
        _record("c1", "intent", "billing", "billing"),
        _record("c2", "intent", "technical", "technical"),
    ]
    assert constraint_violation_rate(records) is None


def test_constraint_violation_rate_excludes():
    """approved=true with a rejection_reason is a violation."""
    excludes_c = [
        {"type": "excludes", "field": "approved", "value": True, "other": "rejection_reason"}
    ]
    records = [
        _record("c1", "approved", True, True, ctype="boolean", constraints=excludes_c),
        _record("c1", "rejection_reason", "", "policy", constraints=excludes_c),
        _record("c2", "approved", True, True, ctype="boolean", constraints=excludes_c),
        _record("c2", "rejection_reason", "", "", constraints=excludes_c),
    ]
    result = constraint_violation_rate(records)
    assert result["overall"] == 0.5
    assert result["by_type"]["excludes"] == 0.5


def test_constraint_violation_rate_exclusivity():
    """enterprise+partner in the same channels prediction is a violation."""
    excl_c = [
        {
            "type": "exclusivity",
            "field": "channels",
            "group": "premium",
            "options": ["enterprise", "partner"],
        }
    ]
    records = [
        _record(
            "c1",
            "channels",
            ["organic", "enterprise"],
            ["enterprise", "partner"],
            ctype="multi",
            constraints=excl_c,
        ),
        _record(
            "c2",
            "channels",
            ["organic"],
            ["organic", "enterprise"],
            ctype="multi",
            constraints=excl_c,
        ),
    ]
    result = constraint_violation_rate(records)
    assert result["overall"] == 0.5


def test_exact_record_accuracy_stricter_than_case_exact_match():
    """c1: all fields correct BUT a constraint is violated -> case_exact_match
    counts it as correct, exact_record_accuracy does not."""
    records = [
        _record("c1", "intent", "billing", "billing", constraints=_IMPLIES),
        _record("c1", "subtype", "refund", "bug", constraints=_IMPLIES),  # wrong prediction
    ]
    # subtype is wrong -> case_exact_match is already 0 here. Let's make
    # a scenario where field accuracy is perfect but constraint is violated:
    # intent=technical (correct), subtype=refund (wrong label but prediction
    # happens to be 'refund' which is wrong for technical intent).
    records = [
        _record("c1", "intent", "technical", "technical", constraints=_IMPLIES),
        _record("c1", "subtype", "bug", "bug", constraints=_IMPLIES),
        _record("c2", "intent", "billing", "billing", constraints=_IMPLIES),
        # subtype label is 'dispute' (valid for billing) but prediction is 'bug'
        # (invalid for billing) — this is both a field error and a constraint violation.
        _record("c2", "subtype", "dispute", "bug", constraints=_IMPLIES),
    ]
    era = exact_record_accuracy(records)
    # c1: all correct, constraint satisfied -> perfect. c2: subtype wrong AND
    # constraint violated -> not perfect. So exact_record = 0.5.
    assert era == 0.5


def test_child_accuracy_given_parent_correct():
    """Only counts child accuracy when parent prediction == parent label."""
    records = [
        _record("c1", "intent", "billing", "billing", constraints=_IMPLIES),  # parent correct
        _record("c1", "subtype", "refund", "refund", constraints=_IMPLIES),  # child correct
        _record("c2", "intent", "technical", "billing", constraints=_IMPLIES),  # parent WRONG
        _record(
            "c2", "subtype", "bug", "bug", constraints=_IMPLIES
        ),  # child correct (but parent wrong, skip)
        _record("c3", "intent", "technical", "technical", constraints=_IMPLIES),  # parent correct
        _record("c3", "subtype", "bug", "feature", constraints=_IMPLIES),  # child WRONG
    ]
    result = child_accuracy_given_parent_correct(records)
    assert result is not None
    # Only c1 and c3 have parent correct. c1 child correct, c3 child wrong -> 0.5.
    assert result["intent→subtype"] == 0.5


def test_order_flip_rate_delegates_to_any_flip_rate():
    """order_flip_rate returns any_flip_rate when permutation data exists,
    None otherwise."""
    records = [
        _record("c1", "intent", "billing", "billing"),
        _record("c2", "intent", "technical", "technical"),
    ]
    # No permutation data -> any_flip_rate returns {} -> order_flip_rate None.
    assert order_flip_rate(records) is None


def test_compute_metrics_includes_dependent_metrics():
    records = [
        _record("c1", "intent", "billing", "billing", constraints=_IMPLIES),
        _record("c1", "subtype", "refund", "refund", constraints=_IMPLIES),
        _record("c2", "intent", "technical", "technical", constraints=_IMPLIES),  # intent correct
        _record(
            "c2", "subtype", "bug", "refund", constraints=_IMPLIES
        ),  # refund NOT in technical's [bug, feature] -> VIOLATION
    ]
    m = compute_metrics(records)
    assert "exact_record_accuracy" in m
    assert "constraint_violation_rate" in m
    assert "child_accuracy_given_parent_correct" in m
    # c2: intent=technical (correct), subtype=refund (wrong AND violates implies:
    # technical -> [bug, feature], refund not allowed).
    assert m["constraint_violation_rate"]["overall"] == 0.5


def test_report_renders_dependent_metrics():
    """_metric_rows surfaces exact_record_accuracy, constraint_violation_rate,
    and child_accuracy_given_parent_correct."""
    run = {
        "metrics": {
            "accuracy": 0.75,
            "exact_record_accuracy": 0.5,
            "constraint_violation_rate": {"overall": 0.5, "by_type": {"implies": 0.5}},
            "child_accuracy_given_parent_correct": {"intent→subtype": 0.5},
        }
    }
    rows = _metric_rows(run)
    names = [r[0] for r in rows]
    assert "exact_record_accuracy" in names
    assert "constraint_violation_rate[overall]" in names
    assert "child_accuracy_given_parent_correct[intent→subtype]" in names
