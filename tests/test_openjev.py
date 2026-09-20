"""W6-B2: OpenJev authored144/perturbations108 datasets + optrev/criterion
perturbations.

Tests the converter (fixture-based, no network), the sha256 fail-closed
guard, the two new label-preserving perturbation kinds (determinism, label
preserved, id conventions), and that perturbation_flip_rate picks them up.
"""

from __future__ import annotations

import json

import pytest

from benchmarks.openjev.fetch import (
    EXPECTED_SHA256,
    REVISION,
    build_case,
    convert_dataset,
)
from benchmarks.perturb import _criterion, _optrev, perturb_case
from jevmlx.evalmetrics import perturbation_flip_rate

# ---- converter fixtures -----------------------------------------------------


# A minimal authored144 row + a perturbation row, mirroring the upstream
# schema (family/group_id/id/label/options/provenance/state).
_AUTHORED_ROW = {
    "family": "evidence_interpretation",
    "group_id": "2fdaa8a61e6e989dd866",
    "id": "a3f18f3a63d45345942b",
    "label": 2,
    "options": [
        {"description": "The evidence establishes the claim", "id": "supported"},
        {"description": "The evidence does not establish either", "id": "insufficient"},
        {"description": "The evidence establishes the opposite", "id": "contradicted"},
    ],
    "provenance": {
        "annotation_status": "all_source_groups_independently_model_reviewed_not_human_adjudicated",
        "kind": "authored_synthetic",
        "variant": "original",
        "partition": "familiar_mechanism_new_situation",
        "source": "project-authored",
    },
    "question": "Assess the claim.",
    "split": "test",
    "state": "The optician ordered replacement lenses.",
    "target_distribution": None,
}

_PERTURB_ROW = {
    "family": "evidence_interpretation",
    "group_id": "2fdaa8a61e6e989dd866/stability",
    "id": "14790d6d50043b03420c",
    "label": 0,
    "options": [
        {"description": "The evidence establishes the opposite", "id": "contradicted"},
        {"description": "The evidence does not establish either", "id": "insufficient"},
        {"description": "The evidence establishes the claim", "id": "supported"},
    ],
    "provenance": {
        "base_id": "a3f18f3a63d45345942b",
        "kind": "project_owned_output_blind_perturbation",
        "variant": "option_reversal",
        "source_group_id": "2fdaa8a61e6e989dd866",
    },
    "question": "Assess the claim.",
    "split": "rebase_stability",
    "state": "The optician ordered replacement lenses.",
    "target_distribution": None,
}


def test_build_case_authored():
    case = build_case(_AUTHORED_ROW)
    assert case["source"] == "openjev"
    assert case["dataset"] == "authored144"
    assert case["benchmark_only"] is True
    assert case["context"] == "The optician ordered replacement lenses."
    # one enum field with the 3 option descriptions
    assert list(case["schema"]) == ["decision"]
    field = case["schema"]["decision"]
    assert field["type"] == "enum"
    assert len(field["choices"]) == 3
    # label = the option description at the label index (a STRING)
    assert case["labels"]["decision"] == "The evidence establishes the opposite"
    # group_id = the row's own group_id (no base_id)
    assert case["group_id"] == "openjev/2fdaa8a61e6e989dd866"
    # annotation_status copied verbatim
    assert case["meta"]["annotation_status"] == (
        "all_source_groups_independently_model_reviewed_not_human_adjudicated"
    )
    assert case["meta"]["source_revision"] == REVISION


def test_build_case_perturbation_clusters_with_base():
    case = build_case(_PERTURB_ROW)
    assert case["dataset"] == "perturbations108"
    # group_id clusters with the base (base_id), not the row's own group_id
    assert case["group_id"] == "openjev/a3f18f3a63d45345942b"
    # the perturbation kind rides meta.perturbation (the variant name)
    assert case["meta"]["perturbation"] == "option_reversal"
    # label is still a valid option description (string, index-independent)
    assert case["labels"]["decision"] == "The evidence establishes the opposite"
    # option order is the ROW's order (already reversed upstream)
    assert case["schema"]["decision"]["choices"][0] == "The evidence establishes the opposite"


def test_convert_dataset_writes_cases_and_lock(tmp_path, monkeypatch):
    """convert_dataset fetches, verifies, converts, writes JSONL + lock."""
    # Patch _download to serve the fixture rows from a temp file.
    import benchmarks.openjev.fetch as fetch_mod

    fixture_path = tmp_path / "fixture.jsonl"
    fixture_path.write_text(
        json.dumps(_AUTHORED_ROW) + "\n" + json.dumps(_PERTURB_ROW) + "\n", encoding="utf-8"
    )
    monkeypatch.setattr(fetch_mod, "_download", lambda name: fixture_path)

    out = tmp_path / "authored144.jsonl"
    lock = tmp_path / "authored144.lock.json"
    n, sha = convert_dataset("authored144", out, lock)
    assert n == 2
    # the sha is of the FIXTURE bytes (2 rows), not the real 144-row file
    import hashlib

    assert sha == hashlib.sha256(fixture_path.read_bytes()).hexdigest()

    records = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 2
    assert records[0]["dataset"] == "authored144"

    lock_data = json.loads(lock.read_text(encoding="utf-8"))
    assert lock_data["source_repo"] == "TheoLeeCJ/openjev"
    assert lock_data["source_revision"] == REVISION
    assert lock_data["source_sha256"] == sha  # the verified fixture bytes
    assert lock_data["expected_sha256"] == EXPECTED_SHA256["authored144"]
    assert lock_data["n_cases"] == 2


def test_convert_dataset_sha_mismatch_fails_closed(tmp_path, monkeypatch):
    """A sha256 mismatch (tampered bytes) is a hard OSError — never converted."""
    import benchmarks.openjev.fetch as fetch_mod

    fixture_path = tmp_path / "tampered.jsonl"
    fixture_path.write_text(json.dumps(_AUTHORED_ROW) + "\n", encoding="utf-8")
    # Monkeypatch EXPECTED_SHA256 to a WRONG value so the fixture's real
    # digest won't match — simulates upstream drift.
    monkeypatch.setattr(fetch_mod, "EXPECTED_SHA256", {"authored144": "0" * 64})
    with pytest.raises(OSError, match="sha256 mismatch"):
        convert_dataset("authored144", tmp_path / "out.jsonl", tmp_path / "lock.json")


def test_convert_dataset_empty_fails_closed(tmp_path, monkeypatch):
    """Zero converted cases never write a lock."""
    import benchmarks.openjev.fetch as fetch_mod

    fixture_path = tmp_path / "empty.jsonl"
    fixture_path.write_text("", encoding="utf-8")
    monkeypatch.setattr(fetch_mod, "_download", lambda name: fixture_path)
    with pytest.raises(OSError, match="0 cases"):
        convert_dataset("authored144", tmp_path / "out.jsonl", tmp_path / "lock.json")


# ---- optrev / criterion perturbation kinds ----------------------------------


def _sample_case():
    return {
        "id": "c1",
        "group_id": "c1",
        "context": "some evidence text",
        "schema": {
            "decision": {
                "type": "enum",
                "description": "Assess the claim.",
                "choices": [
                    "The evidence establishes the claim",
                    "The evidence does not establish either",
                    "The evidence establishes the opposite",
                ],
            }
        },
        "labels": {"decision": "The evidence establishes the opposite"},
        "meta": {},
    }


def test_optrev_reverses_option_order_label_preserved():
    case = _sample_case()
    rev = _optrev(case)
    assert rev["schema"]["decision"]["choices"] == list(
        reversed(case["schema"]["decision"]["choices"])
    )
    # label is the option DESCRIPTION — unchanged by reversing the order
    assert rev["labels"]["decision"] == case["labels"]["decision"]


def test_criterion_prefixes_description_label_preserved():
    case = _sample_case()
    crit = _criterion(case)
    assert crit["schema"]["decision"]["description"].startswith(
        "Using only the supplied evidence, decide the following criterion: "
    )
    assert rev_not_touched_labels(crit, case)


def rev_not_touched_labels(a, b):
    return a["labels"] == b["labels"]


def test_optrev_noop_on_single_choice():
    """An enum with one choice cannot be reversed — returns the original."""
    case = _sample_case()
    case["schema"]["decision"]["choices"] = ["only"]
    rev = _optrev(case)
    assert rev is case  # no-op


def test_optrev_noop_on_non_enum():
    """Boolean fields are skipped (optrev is enum-only)."""
    case = _sample_case()
    case["schema"]["decision"]["type"] = "boolean"
    rev = _optrev(case)
    assert rev is case


def test_perturb_case_produces_both_new_kinds():
    case = _sample_case()
    variants = perturb_case(case, variants=5, seed=42)
    kinds = [v["meta"]["perturbation"] for v in variants]
    assert "optrev" in kinds
    assert "criterion" in kinds


def test_perturb_case_deterministic_ids():
    """The same (case, seed) always produces the same variant ids."""
    case = _sample_case()
    v1 = perturb_case(case, variants=5, seed=42)
    v2 = perturb_case(case, variants=5, seed=42)
    assert [v["id"] for v in v1] == [v["id"] for v in v2]


def test_perturb_case_group_id_is_original_id():
    """Every variant's group_id is the original case id (flip-rate pairing)."""
    case = _sample_case()
    variants = perturb_case(case, variants=5, seed=42)
    assert all(v["group_id"] == "c1" for v in variants)
    assert all(v["id"].startswith("c1#p") for v in variants)


def test_perturb_case_idempotent_criterion():
    """Applying criterion twice is a no-op (the prefix is already there)."""
    case = _sample_case()
    once = _criterion(case)
    twice = _criterion(once)
    assert twice is once  # no-op


# ---- perturbation_flip_rate picks up the new kinds --------------------------


def test_flip_rate_picks_up_optrev_and_criterion():
    """The metric keys on group_id + meta.perturbation — the new kinds flow
    through with no code change."""
    records = [
        {
            "case_id": "c1",
            "group_id": "c1",
            "field": "decision",
            "prediction": "The evidence establishes the opposite",
            "permutation": "canonical",
        },
        {
            "case_id": "c1#p2",
            "group_id": "c1",
            "field": "decision",
            "prediction": "The evidence establishes the claim",
            "permutation": "canonical",
            "meta": {"perturbation": "optrev"},
        },
        {
            "case_id": "c1#p3",
            "group_id": "c1",
            "field": "decision",
            "prediction": "The evidence establishes the opposite",
            "permutation": "canonical",
            "meta": {"perturbation": "criterion"},
        },
    ]
    rate = perturbation_flip_rate(records)
    # optrev flipped, criterion didn't => 1/2 = 0.5
    assert rate == 0.5


def test_flip_rate_none_when_no_pairs():
    assert perturbation_flip_rate([]) is None
