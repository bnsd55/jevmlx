"""Tests for the TypeSafe eval fetcher (round 2): node identity, consensus,
read-set splitting, and the dataset lock.

No network: tests run against synthetic ``__VIEWER_DATA__`` fixtures written
to a temp cache dir, mirroring the structure of the published pages.
"""

import hashlib
import json
from pathlib import Path

import pytest

from benchmarks.typesafe import fetch as fetch_module
from benchmarks.typesafe.fetch import (
    PARSER_VERSION,
    _consensus,
    _field_schema,
    _viewer_json,
    fetch_all,
    fetch_workflow,
    split_for,
    write_outputs,
)


def _demo_eval() -> dict:
    """A minimal but structurally faithful eval payload.

    Case 'case-a' has two reference nodes with DIFFERENT read-sets (triage
    reads doc 0, containment reads doc 1) -> two records. Case 'case-b'
    has one node -> one record. q_auth is answered at both nodes -> field
    names carry node and per-case occurrence (triage=1, containment=2).
    """
    return {
        "id": "security_incidents",
        "title": "Security Incidents",
        "documents": [
            "Alert IDS-1001: beaconing observed from workstation-7.",
            {"kind": "log", "lines": ["auth failure x50"]},
        ],
        "questions": [
            {
                "type": "noul",
                "instructions": "Is this unauthorized activity?",
                "criteria": {"true": "proof of attack", "false": "authorized"},
            },
            {
                "type": "choice",
                "instructions": "How far does this reach?",
                "criteria": {
                    "workgroup": "A team",
                    "organization_wide": "Everyone",
                    "single_entity": "One host",
                },
            },
            {
                "type": "score",
                "instructions": "How strong is the evidence?",
                "criteria": ["Speculative", "Suggestive"],
            },
            {"type": "text", "instructions": "Summarize the incident in free text."},
        ],
        "examples": [
            {"case_id": "case-a", "name": "A"},
            {"case_id": "case-b", "name": "B"},
        ],
        "cases": {
            "case-a": {
                "models": {
                    "opus": {
                        "nodes": [
                            {
                                "node": "triage",
                                "doc": 0,
                                "questions": {"q_auth": 0, "q_scope": 1, "q_strength": 2},
                                "answers": {
                                    "q_auth": {"type": "noul", "noul": False},
                                    "q_scope": {"type": "choice", "choice": "single_entity"},
                                },
                            },
                            {
                                "node": "containment",
                                "doc": 1,
                                "questions": {"q_auth": 0},
                                "answers": {"q_auth": {"type": "noul", "noul": True}},
                            },
                        ]
                    },
                    "sol": {
                        "nodes": [
                            {
                                "node": "triage",
                                "doc": 0,
                                "answers": {
                                    "q_scope": {"type": "choice", "choice": "organization_wide"}
                                },
                            }
                        ]
                    },
                },
                "reference_answers": {
                    "triage": {
                        "q_auth": {
                            "type": "noul",
                            "sets": [
                                {"value": False, "probabilities": {"true": 0.02, "false": 0.98}},
                                {"value": False, "probabilities": {"true": 0.04, "false": 0.96}},
                            ],
                        },
                        "q_scope": {
                            "type": "choice",
                            "sets": [
                                {
                                    "value": "workgroup",
                                    "probabilities": {
                                        "workgroup": 0.6,
                                        "organization_wide": 0.2,
                                        "single_entity": 0.2,
                                    },
                                },
                                {
                                    "value": "workgroup",
                                    "probabilities": {
                                        "workgroup": 0.8,
                                        "organization_wide": 0.1,
                                        "single_entity": 0.1,
                                    },
                                },
                            ],
                        },
                        "q_strength": {"type": "score", "sets": [{"value": 1}]},
                        "q_summary": {"type": "text", "sets": [{"value": "free text"}]},
                    },
                    "containment": {
                        "q_auth": {
                            "type": "noul",
                            "sets": [{"value": True, "probabilities": {"true": 0.9, "false": 0.1}}],
                        }
                    },
                },
            },
            "case-b": {
                "models": {
                    "opus": {
                        "nodes": [
                            {
                                "node": "triage",
                                "doc": 1,
                                "questions": {"q_summary": 3},
                                "answers": {},
                            }
                        ]
                    }
                },
                "reference_answers": {
                    "triage": {"q_summary": {"type": "text", "sets": [{"value": "n/a"}]}}
                },
            },
        },
    }


@pytest.fixture()
def cache_dir(tmp_path: Path) -> Path:
    """A cache dir in the real content-addressed layout for the demo page."""
    path = tmp_path / "typesafe"
    path.mkdir(parents=True)
    # The real pages embed {"eval": {...}} (plus viewer chrome) in the call.
    page = "__VIEWER_DATA__(" + json.dumps({"eval": _demo_eval()}) + ");"
    raw = page.encode("utf-8")
    sha256 = hashlib.sha256(raw).hexdigest()
    (path / f"demo-{sha256[:12]}.js").write_bytes(raw)
    (path / "demo-cases.js.meta").write_text(
        json.dumps(
            {
                "url": "https://evals.typesafe.ai/demo-cases.js",
                "sha256": sha256,
                "fetched_at": "2026-09-17T00:00:00+00:00",
                "etag": None,
            }
        ),
        encoding="utf-8",
    )
    return path


def _records(cache_dir: Path) -> list[dict]:
    records, _summary, _sources = fetch_all(["demo"], cache_dir)
    return records


def test_viewer_json_extracts_payload():
    payload = _viewer_json('__VIEWER_DATA__({"a": 1});')
    assert payload == {"a": 1}
    with pytest.raises(ValueError, match="__VIEWER_DATA__"):
        _viewer_json("nothing here")


def test_split_is_deterministic_and_matches_documented_rule():
    """split_for is a pure function of the id, per the documented sha1 rule."""
    for case_id in ("a", "typesafe/x/y", "z" * 40):
        assert split_for(case_id) == split_for(case_id)
        digest = hashlib.sha1(case_id.encode()).hexdigest()
        expected = "holdout" if int(digest[:8], 16) % 5 == 0 else "train"
        assert split_for(case_id) == expected


def test_field_schema_maps_question_types():
    assert _field_schema(
        {"type": "noul", "instructions": "Is this bad?", "criteria": {"true": "x", "false": "y"}}
    ) == {"type": "boolean", "description": "Is this bad?"}
    choice = _field_schema(
        {"type": "choice", "instructions": "Reach?", "criteria": {"a": "A", "b": "B"}}
    )
    assert choice["type"] == "enum" and choice["choices"] == ["a", "b"]
    score = _field_schema(
        {"type": "score", "instructions": "Strength?", "criteria": ["weak", "strong"]}
    )
    assert score["type"] == "enum" and score["choices"] == ["0", "1", "2", "3"]


def test_read_set_split_and_field_identity(cache_dir):
    """Different read-sets split the case; repeated qids carry node identity."""
    records = _records(cache_dir)
    by_id = {record["id"]: record for record in records}

    # case-a's two reference nodes read different documents -> two records.
    triage = by_id["typesafe/demo/case-a/n0"]
    containment = by_id["typesafe/demo/case-a/n1"]
    assert triage["group_id"] == "typesafe/demo/case-a"
    assert containment["group_id"] == "typesafe/demo/case-a"
    # Read-sets are preserved exactly: triage reads doc 0, containment doc 1.
    assert triage["context"].startswith("## Document 0\nAlert IDS-1001")
    assert "## Document 1" not in triage["context"]
    assert containment["context"].startswith("## Document 1\n{")
    assert "## Document 0" not in containment["context"]

    # q_auth is answered at both nodes -> node-qualified field names.
    assert set(triage["schema"]) == {"triage__q_auth__1", "q_scope", "q_strength"}
    assert set(containment["schema"]) == {"containment__q_auth__2"}
    assert triage["labels"]["triage__q_auth__1"] is False
    assert containment["labels"]["containment__q_auth__2"] is True

    # case-b has a single node -> one record, no /n suffix.
    assert list(by_id) == [
        "typesafe/demo/case-a/n0",
        "typesafe/demo/case-a/n1",
    ]

    # case-b's only question (q_summary) is free-text (type=text) -> no
    # decidable field -> the group is skipped (empty schema). The fetch
    # summary reports it.
    _, summary, _ = fetch_all(["demo"], cache_dir)
    assert summary["skipped_groups"] == 1


def test_empty_schema_group_is_skipped_and_counted(cache_dir):
    """A group where every question was skipped (unmapped qid or no
    consensus) is NOT yielded — it has no decidable field. The fetch summary
    counts it in 'skipped_groups' so the drop is visible.

    Uses the real fetcher functions on the in-memory demo cache (no network):
    case-b's only question (q_summary, type=text) is free-text, unmapped by
    _field_schema, so case-b's single group yields an empty schema and is
    skipped. case-a's groups have mapped questions and are kept."""
    records, summary, _ = fetch_all(["demo"], cache_dir)
    # case-a's two groups (triage + containment) survive.
    assert len(records) == 2
    assert all(r["schema"] for r in records)
    # case-b's single group (all questions skipped) is dropped + counted.
    assert summary["skipped_groups"] == 1
    # No yielded record has an empty schema or empty labels.
    assert all(r["schema"] and r["labels"] for r in records)


def test_record_contract_keys_and_benchmark_only(cache_dir):
    """Every record carries the frozen contract keys and benchmark_only=true."""
    for record in _records(cache_dir):
        assert set(record) == {
            "id",
            "group_id",
            "source",
            "workflow",
            "benchmark_only",
            "schema",
            "context",
            "labels",
            "split",
            "meta",
        }
        assert record["source"] == "typesafe"
        assert record["workflow"] == "demo"
        assert record["benchmark_only"] is True
        assert set(record["meta"]) == {"consensus", "margin", "ambiguous", "models"}


def test_consensus_meta_margin_and_ambiguity(cache_dir):
    """meta carries the full distribution, margin, and ambiguity flag."""
    records = _records(cache_dir)
    triage = next(r for r in records if r["id"] == "typesafe/demo/case-a/n0")

    # q_scope: averaged distribution, margin = 0.7 - 0.15.
    distribution = triage["meta"]["consensus"]["q_scope"]
    assert distribution == pytest.approx(
        {"workgroup": 0.7, "organization_wide": 0.15, "single_entity": 0.15}
    )
    assert triage["meta"]["margin"]["q_scope"] == pytest.approx(0.55)
    assert "q_scope" not in triage["meta"]["ambiguous"]

    # q_auth triage: 0.97 vs 0.03 -> high margin, not ambiguous.
    assert triage["meta"]["consensus"]["triage__q_auth__1"] == pytest.approx(
        {"true": 0.03, "false": 0.97}
    )
    assert triage["meta"]["margin"]["triage__q_auth__1"] == pytest.approx(0.94)
    assert triage["meta"]["ambiguous"] == []


def test_consensus_normalizes_over_contributing_reviewers_only():
    """A bare-value set among probability sets is a mixed form: rejected."""
    entry = {
        "type": "noul",
        "sets": [
            {"value": False, "probabilities": {"true": 0.9, "false": 0.1}},
            {"value": True},
        ],
    }
    with pytest.raises(ValueError, match="mixed consensus forms"):
        _consensus(entry, "w/case/node/q")


def test_consensus_bare_votes_become_share_distribution():
    entry = {
        "type": "noul",
        "sets": [{"value": True}, {"value": True}, {"value": False}],
    }
    label, distribution, margin, ambiguous = _consensus(entry, "w/c/node/q")
    assert label is True
    assert distribution == pytest.approx({"true": 2 / 3, "false": 1 / 3})
    assert margin == pytest.approx(1 / 3)
    assert not ambiguous


def test_consensus_tie_is_ambiguous_and_broken_by_choice_order():
    """Exact tie: ambiguous, winner = first in choice order (not insertion)."""
    entry = {
        "type": "choice",
        "sets": [
            {"value": "workgroup", "probabilities": {"workgroup": 1.0, "single_entity": 0.0}},
            {"value": "single_entity", "probabilities": {"workgroup": 0.0, "single_entity": 1.0}},
        ],
    }
    label, distribution, margin, ambiguous = _consensus(
        entry, "w/c/node/q", choices=["single_entity", "workgroup"]
    )
    assert distribution == pytest.approx({"workgroup": 0.5, "single_entity": 0.5})
    assert margin == pytest.approx(0.0)
    assert ambiguous is True
    assert label == "single_entity"  # first in the published choice order


def test_consensus_ambiguity_boundary_is_strict():
    """ambiguous means margin < 0.1: exactly 0.1 is not, below it is."""
    at_boundary = {
        "type": "noul",
        "sets": [{"value": True, "probabilities": {"true": 0.55, "false": 0.45}}],
    }
    _label, _dist, margin, ambiguous = _consensus(at_boundary, "w/c/node/q")
    assert margin == pytest.approx(0.1)
    assert ambiguous is False

    below = {
        "type": "noul",
        "sets": [{"value": True, "probabilities": {"true": 0.54, "false": 0.46}}],
    }
    _label, _dist, margin, ambiguous = _consensus(below, "w/c/node/q")
    assert margin == pytest.approx(0.08)
    assert ambiguous is True


def test_fetch_workflow_offline_cache_with_sha256_and_refresh(cache_dir, monkeypatch):
    """Cached fetch never touches the network; refresh re-downloads."""
    calls: list[str] = []

    def fake_download(url):
        calls.append(url)
        page = "__VIEWER_DATA__(" + json.dumps({"eval": _demo_eval()}) + ");"
        return page.encode(), '"etag-v2"'

    monkeypatch.setattr(fetch_module, "_download", fake_download)

    # Warm cache from the fixture: no download.
    eval_obj, meta = fetch_workflow("demo", cache_dir)
    assert eval_obj["id"] == "security_incidents"
    page_sha = hashlib.sha256(next(p for p in cache_dir.glob("demo-*.js")).read_bytes()).hexdigest()
    assert meta["sha256"] == page_sha
    assert meta["fetched_at"]
    assert calls == []

    # --refresh re-downloads and records the new fetch time.
    eval_obj, meta = fetch_workflow("demo", cache_dir, refresh=True)
    assert calls == [meta["url"]]
    assert meta["etag"] == '"etag-v2"'
    # Content-addressed storage: raw bytes live under their sha256 name.
    stored = cache_dir / f"demo-{meta['sha256'][:12]}.js"
    assert stored.exists()


def test_dataset_lock_contents(cache_dir, tmp_path):
    """Lock records sources, parser version, counts, and the cases sha256."""
    records, summary, sources = fetch_all(["demo"], cache_dir)
    lock_path = write_outputs(records, tmp_path / "cases.jsonl", sources, summary)
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    cases_sha = hashlib.sha256((tmp_path / "cases.jsonl").read_bytes()).hexdigest()

    assert lock["parser_version"] == PARSER_VERSION
    assert lock["counts"] == summary
    assert lock["cases_sha256"] == cases_sha
    assert lock["sources"] == [
        {
            "url": "https://evals.typesafe.ai/demo-cases.js",
            "sha256": hashlib.sha256(
                next(p for p in cache_dir.glob("demo-*.js")).read_bytes()
            ).hexdigest(),
            "fetched_at": lock["sources"][0]["fetched_at"],
            "etag": None,
        }
    ]
    # Fixture page has no ETag header; real fetches record one when sent.
    assert lock["sources"][0]["fetched_at"].startswith("20")


def test_jsonl_contract_keys(cache_dir, tmp_path):
    """Each line carries exactly the frozen harness contract keys."""
    records, _summary, _sources = fetch_all(["demo"], cache_dir)
    lock_path = write_outputs(records, tmp_path / "cases.jsonl", [], {"records": len(records)})
    lines = [json.loads(line) for line in (tmp_path / "cases.jsonl").read_text().splitlines()]
    assert lines == records
    for line in lines:
        assert set(line) == {
            "id",
            "group_id",
            "source",
            "workflow",
            "benchmark_only",
            "schema",
            "context",
            "labels",
            "split",
            "meta",
        }
    assert lock_path.exists()


def test_cli_main_writes_jsonl_lock_and_summary(cache_dir, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(fetch_module, "CACHE_DIR", cache_dir)
    monkeypatch.setattr(fetch_module, "WORKFLOWS", ("demo",))  # synthetic workflow
    out = tmp_path / "cases.jsonl"
    rc = fetch_module.main(["--out", str(out), "--workflow", "demo"])
    assert rc == 0
    assert len(out.read_text(encoding="utf-8").splitlines()) == 2
    printed = capsys.readouterr().out
    assert "records: 2" in printed
    assert "skipped questions (free text): 2" in printed
    assert "skipped groups (empty schema): 1" in printed
    assert (tmp_path / "dataset.lock.json").exists()
