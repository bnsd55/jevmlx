"""Tests for benchmarks.public.jabr (W6-B1: jabr classifier-benchmark).

Network-free: the AST parser runs against a fixture source file; the
download path is monkeypatched. The pin test asserts the revision is a
40-hex commit sha and EXPECTED_SHA256 is set.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import benchmarks.public.jabr as jabr
from benchmarks.public.jabr import (
    EXPECTED_SHA256,
    LICENSE,
    PARSER_VERSION,
    RAW_URL,
    REPO_ID,
    REVISION,
    SOURCE_FILE,
    build_records,
    main,
    parse_cases_source,
    write_dataset,
)
from benchmarks.typesafe.questions import field_schema

FIXTURE = Path(__file__).parent / "fixtures" / "jabr" / "cases_fixture.py"


# ---- AST parser ------------------------------------------------------------


def test_parse_fixture_extracts_three_tasks():
    """The parser extracts one task per primitive (choice, noul, score)
    from the fixture source — never exec'ing it."""
    source = FIXTURE.read_text(encoding="utf-8")
    tasks = parse_cases_source(source)
    assert len(tasks) == 3
    ids = [t["id"] for t in tasks]
    assert ids == ["support_department", "refund_eligible", "frustration_level"]


def test_parse_fixture_case_counts():
    source = FIXTURE.read_text(encoding="utf-8")
    tasks = parse_cases_source(source)
    counts = {t["id"]: len(t["cases"]) for t in tasks}
    assert counts == {
        "support_department": 3,
        "refund_eligible": 2,
        "frustration_level": 3,
    }


def test_parse_real_source_extracts_8_tasks_78_cases():
    """Against the REAL bench/cases.py at the pinned revision: 8 tasks,
    78 cases (the headline suite). Requires network to fetch the file."""
    try:
        source, sha = jabr._download()
    except OSError:
        pytest.skip("offline: cannot fetch the real jabr source")
    assert sha == EXPECTED_SHA256
    tasks = parse_cases_source(source)
    assert len(tasks) == 8
    total = sum(len(t["cases"]) for t in tasks)
    assert total == 78


def test_parse_rejects_empty_source():
    """No Task definitions -> ValueError, not an empty list."""
    with pytest.raises(ValueError, match="no Task definitions"):
        parse_cases_source("# nothing here\n")


# ---- schema + label mapping ------------------------------------------------


def test_choice_task_maps_to_enum():
    source = FIXTURE.read_text(encoding="utf-8")
    tasks = parse_cases_source(source)
    choice_task = next(t for t in tasks if t["type"] == "choice")
    schema = jabr._task_schema(choice_task)
    field = schema["support_department"]
    assert field["type"] == "enum"
    assert field["choices"] == ["billing", "tech", "other"]
    # Came from the SHARED mapping (same field_schema call the HF datasets use).
    base = field_schema(
        {"type": "choice", "instructions": "x", "criteria": dict.fromkeys(field["choices"], "")}
    )
    assert base["type"] == field["type"] == "enum"


def test_noul_task_maps_to_boolean():
    source = FIXTURE.read_text(encoding="utf-8")
    tasks = parse_cases_source(source)
    noul_task = next(t for t in tasks if t["type"] == "noul")
    schema = jabr._task_schema(noul_task)
    assert schema["refund_eligible"]["type"] == "boolean"  # noul -> boolean


def test_score_task_maps_to_ordered_enum():
    source = FIXTURE.read_text(encoding="utf-8")
    tasks = parse_cases_source(source)
    score_task = next(t for t in tasks if t["type"] == "score")
    schema = jabr._task_schema(score_task)
    field = schema["frustration_level"]
    assert field["type"] == "enum"
    assert field["choices"] == ["0", "1", "2"]  # 3-level ordinal
    assert field["ordered"] is True  # OrdinalTelemetry applies


def test_label_form_booleans_become_real_bools():
    source = FIXTURE.read_text(encoding="utf-8")
    tasks = parse_cases_source(source)
    noul_task = next(t for t in tasks if t["type"] == "noul")
    assert jabr._task_label(noul_task, True) is True
    assert jabr._task_label(noul_task, False) is False


def test_option_order_enum_vs_boolean():
    source = FIXTURE.read_text(encoding="utf-8")
    tasks = parse_cases_source(source)
    choice_task = next(t for t in tasks if t["type"] == "choice")
    noul_task = next(t for t in tasks if t["type"] == "noul")
    assert jabr._option_order(choice_task, jabr._task_schema(choice_task)) == [
        "billing",
        "tech",
        "other",
    ]
    assert jabr._option_order(noul_task, jabr._task_schema(noul_task)) == ["false", "true"]


# ---- F5: download pin is VERIFIED, not just recorded ----------------------


def test_pin_is_commit_sha_and_sha256_recorded():
    """The pin: a 40-hex commit sha (never a floating branch) and a
    known-good sha256 of the fetched file."""
    assert len(REVISION) == 40
    int(REVISION, 16)  # hex
    assert len(EXPECTED_SHA256) == 64
    int(EXPECTED_SHA256, 16)
    assert REPO_ID == "jabr/classifier-benchmark"
    assert SOURCE_FILE == "bench/cases.py"
    assert REVISION in RAW_URL


def test_license_is_cc0_redistribution_allowed():
    """jabr is CC0 (public domain) — redistribution IS allowed (unlike the
    HF datasets whose text stays in the uncommitted cache)."""
    assert "CC0" in LICENSE
    assert "public domain" in LICENSE


def test_download_verifies_sha256_and_fails_closed(monkeypatch, tmp_path):
    """F5: a sha256 mismatch is a hard OSError (never parsed, never locked)."""

    def fake_urlopen(request, timeout=60):
        class FakeResponse:
            def read(self):
                return b"tampered bytes"

            def __enter__(self):
                return self

            def __exit__(self, *a):
                pass

        return FakeResponse()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    with pytest.raises(OSError, match="sha256 mismatch"):
        jabr._download()


def test_download_failure_is_oserror_not_systemexit(monkeypatch):
    """F6: build_datasets catches OSError for the offline skip; a download
    failure must be one."""

    def boom(request, timeout=60):
        raise ConnectionError("refused")

    monkeypatch.setattr("urllib.request.urlopen", boom)
    with pytest.raises(OSError, match="download failed"):
        jabr._download()


# ---- build + write --------------------------------------------------------


@pytest.fixture()
def hubless(monkeypatch):
    """Monkeypatch _download to return the fixture source (verified)."""
    fixture_source = FIXTURE.read_text(encoding="utf-8")
    fixture_sha = hashlib.sha256(fixture_source.encode("utf-8")).hexdigest()

    def fake_download():
        return fixture_source, fixture_sha

    monkeypatch.setattr(jabr, "_download", fake_download)
    # Also align EXPECTED_SHA256 so the verification passes.
    monkeypatch.setattr(jabr, "EXPECTED_SHA256", fixture_sha)
    return fixture_sha


def test_build_records_produces_correct_shape(hubless):
    """8 tasks, 78 cases from the real source — but via the fixture (3
    tasks, 8 cases) for offline testing."""
    records, sha = build_records()
    assert len(records) == 8  # 3 + 2 + 3 from the fixture
    assert sha == hubless
    # Every record carries the REAL text in context.
    for record in records:
        assert record["context"]
        assert record["benchmark_only"] is True
        assert record["source"] == "jabr"
        assert record["view"] == "all"
        assert record["split"] == "test"
        assert record["workflow"] == record["meta"]["task"]
    # Per-task counts.
    tasks = {}
    for r in records:
        tasks[r["meta"]["task"]] = tasks.get(r["meta"]["task"], 0) + 1
    assert tasks == {"support_department": 3, "refund_eligible": 2, "frustration_level": 3}


def test_build_records_carry_provenance(hubless):
    records, _ = build_records()
    for record in records:
        assert record["meta"]["source_repo"] == REPO_ID
        assert record["meta"]["source_revision"] == REVISION
        assert record["meta"]["source_file"] == SOURCE_FILE
        assert record["meta"]["selected_input_hash"]
        assert len(record["meta"]["selected_input_hash"]) == 64


def test_write_dataset_lock_records_pin_and_sha(hubless, tmp_path):
    records, _ = build_records()
    out = tmp_path / "jabr.jsonl"
    lock_path = write_dataset(records, out)
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    assert lock["sources"][0]["repo_id"] == REPO_ID
    assert lock["sources"][0]["revision"] == REVISION
    assert lock["sources"][0]["license"] == LICENSE
    assert lock["sources"][0]["redistribution_allowed"] is True
    assert lock["sources"][0]["files"][SOURCE_FILE] == jabr.EXPECTED_SHA256
    assert lock["parser_version"] == PARSER_VERSION
    assert lock["cases_sha256"] == hashlib.sha256(out.read_bytes()).hexdigest()
    assert lock["counts"]["records"] == 8
    assert lock["counts"]["tasks"] == 3


def test_write_dataset_refuses_empty(tmp_path):
    with pytest.raises(OSError, match="empty dataset"):
        write_dataset([], tmp_path / "x.jsonl")


def test_main_writes_files(hubless, tmp_path):
    rc = main(["--out", str(tmp_path / "jabr.jsonl"), "--lock", str(tmp_path / "jabr.lock.json")])
    assert rc == 0
    assert (tmp_path / "jabr.jsonl").exists()
    assert (tmp_path / "jabr.lock.json").exists()
    records = [
        json.loads(line)
        for line in (tmp_path / "jabr.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(records) == 8


# ---- bench wiring ---------------------------------------------------------


def test_bench_wires_jabr_name(hubless, monkeypatch, tmp_path):
    """jevmlx.bench builds 'jabr' with the same lock shape (cases_sha256-
    verified reuse) as the other datasets — offline."""
    import jevmlx.bench as bench

    monkeypatch.setattr(bench, "BENCH_CACHE", tmp_path)
    paths, locks = bench.build_datasets(["jabr"], offline_ok=False)
    assert "jabr" in paths
    lock = json.loads(locks["jabr"].read_text(encoding="utf-8"))
    assert lock["cases_sha256"] == hashlib.sha256(paths["jabr"].read_bytes()).hexdigest()
    assert lock["sources"][0]["redistribution_allowed"] is True


def test_bench_pin_problem_detects_drift(hubless, monkeypatch, tmp_path):
    """F5 on reuse: a lock whose recorded file sha256 differs from the pin
    forces a rebuild; a matching lock passes."""
    import jevmlx.bench as bench

    monkeypatch.setattr(bench, "BENCH_CACHE", tmp_path)
    _paths, locks = bench.build_datasets(["jabr"], offline_ok=False)
    lock = locks["jabr"]
    assert bench._jabr_pin_problem(lock) is None

    data = json.loads(lock.read_text(encoding="utf-8"))
    data["sources"][0]["files"][SOURCE_FILE] = "0" * 64
    drifted = tmp_path / "drifted.json"
    drifted.write_text(json.dumps(data))
    problem = bench._jabr_pin_problem(drifted)
    assert problem and "sha256 drift" in problem


def test_jabr_in_datasets_all():
    """'jabr' is registered in DATASETS_ALL so the CLI accepts it."""
    import jevmlx.bench as bench

    assert "jabr" in bench.DATASETS_ALL


def test_jabr_not_in_public_views():
    """jabr is single-view — it must NOT be in PUBLIC_DATASETS (which
    expands to balanced/natural). It passes through normalize_dataset_names
    unchanged."""
    import jevmlx.bench as bench

    assert "jabr" not in bench.PUBLIC_DATASETS
    assert "jabr" not in bench.PUBLIC_DATASET_NAMES
    # normalize does not expand jabr (no .balanced / .natural suffixes).
    assert bench.normalize_dataset_names(["jabr"]) == ["jabr"]


def test_leaderboard_jabr_group_registered():
    """jabr is a _LOCAL_GROUPS entry so _local_rows emits a row for it."""
    from benchmarks.leaderboard import _LOCAL_GROUPS

    assert "jabr" in _LOCAL_GROUPS
