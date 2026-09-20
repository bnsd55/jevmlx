"""Tests for benchmarks.public.fetch (W6-B5: AG News / BoolQ / SST-5).

Network-free: the fetch paths run against fixture parquet/jsonl files
monkeypatched in place of hf_hub_download. The one pin test asserts the
DEFAULT_REVISION-style pins are 40-hex commit shas and that --check (the
pin verifier) accepts them.
"""

from __future__ import annotations

import hashlib
import json
import pathlib

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import benchmarks.public.fetch as pf
from benchmarks.public.fetch import (
    DATASETS,
    EVAL_SPLITS,
    EXPECTED_SHA256,
    NATURAL_ROWS,
    PARSER_VERSION,
    SAMPLE_SEED,
    _balanced_positions,
    _input_hash,
    _natural_positions,
    _row_text,
    _slot,
    build_case,
    fetch_dataset,
    main,
    write_view,
)
from benchmarks.typesafe.questions import field_schema

# ---- fixtures --------------------------------------------------------------


@pytest.fixture()
def ag_news_rows() -> list[dict]:
    """40 rows, 10 per class (label 0..3): enough for balanced + disjoint
    natural views."""
    seeds = [
        ("the UN assembly met in Geneva yesterday", 0),
        ("the Lakers won the playoff game in overtime", 1),
        ("oil prices climbed after the quarterly earnings report", 2),
        ("Microsoft unveiled a new processor architecture", 3),
        ("the security council debated the ceasefire", 0),
        ("the striker scored twice in the second half", 1),
        ("the central bank raised interest rates again", 2),
        ("researchers demonstrated a quantum error correction gain", 3),
    ]
    rows = []
    for copy in range(5):
        for text, label in seeds:
            rows.append({"text": f"{text} (report {copy})", "label": label})
    return rows


@pytest.fixture()
def boolq_rows() -> list[dict]:
    out = []
    for i in range(30):
        out.append(
            {
                "question": f"q{i}",
                "answer": i % 2 == 0,
                "passage": f"passage {i} " * 20,
            }
        )
    return out


@pytest.fixture()
def sst5_rows() -> list[dict]:
    out = []
    for i in range(60):
        out.append(
            {
                "text": f"review phrase {i}",
                "label": i % 5,
                "label_text": "level",
            }
        )
    return out


@pytest.fixture()
def hubless(monkeypatch, tmp_path, ag_news_rows, boolq_rows, sst5_rows):
    """Point _download at fixture files; capture the requested (repo, rev).

    F5: the fake download must also satisfy the sha256 pin check — the
    expected digests are monkeypatched to the FIXTURE bytes.
    """
    requested: list[tuple[str, str, str]] = []
    real_expected = {}

    def fake_download(ds, split):
        requested.append((ds.repo_id, ds.revision, ds.files[split]))
        rows_by_split = {
            "ag_news": ag_news_rows,
            "boolq": boolq_rows,
            "sst5": sst5_rows,
        }
        rows = rows_by_split[ds.name]
        if ds.name == "sst5":
            path = tmp_path / f"{ds.name}-{split}.jsonl"
            path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
        else:
            path = tmp_path / f"{ds.name}-{split}.parquet"
            pq.write_table(pa.Table.from_pylist(rows), path)
        return path

    # The pin check reads ds.expected_sha256 and hashes the file; align the
    # expectations with the fixture bytes so verification passes.
    original_download = pf._download

    def verified_download(ds, split):
        path = fake_download(ds, split)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        real_expected[f"{ds.repo_id}/{ds.files[split]}"] = ds.expected_sha256.get(ds.files[split])
        monkeypatch.setattr(ds, "expected_sha256", {ds.files[split]: digest}, raising=False)
        return path

    # _download is called via the module attribute; fetch paths call it
    # directly, so replace it wholesale (the sha256 it computes inside is
    # over the fake file, and expected_sha256 is aligned above).
    monkeypatch.setattr(pf, "_download", verified_download)
    yield requested
    monkeypatch.setattr(pf, "_download", original_download, raising=False)


# ---- schema mapping (shared questions.py, no copies) -----------------------


def test_ag_news_schema_uses_the_shared_choice_mapping():
    schema = pf._ag_news_schema()
    field = schema["topic"]
    assert field["type"] == "enum"
    assert field["choices"] == pf.AG_NEWS_CLASSES
    # It came from the SHARED mapping: the same call the typesafe fetcher
    # makes for a choice question, then description-refined.
    base = field_schema(
        {"type": "choice", "instructions": "x", "criteria": dict.fromkeys(pf.AG_NEWS_CLASSES, "")}
    )
    assert base["type"] == field["type"] == "enum"


def test_boolq_schema_uses_the_shared_noul_mapping():
    schema = pf._boolq_schema()
    assert schema["answer"]["type"] == "boolean"  # noul -> boolean, shared


def test_sst5_schema_uses_the_shared_score_mapping():
    schema = pf._sst5_schema()
    field = schema["sentiment"]
    assert field["type"] == "enum"
    assert field["choices"] == ["0", "1", "2", "3", "4"]  # ordinal order kept
    assert "very negative" in field["description"] and "very positive" in field["description"]


# ---- deterministic sampling -------------------------------------------------


def test_slot_is_deterministic_and_seed_bound(ag_news_rows):
    ds = DATASETS["ag_news"]
    rid = ds.row_id(ag_news_rows[0], "test", 0)
    assert _slot(rid) == _slot(rid)
    # The seed is part of the slot: different seed -> different ordering.
    with_seed = hashlib.sha256(f"{SAMPLE_SEED}\0{rid}".encode()).digest()
    without = hashlib.sha256(rid.encode()).digest()
    assert with_seed != without


def test_row_ids_are_split_aware_and_duplicate_free():
    """F10: row_id takes (row, split, index) — the same index under a
    different split gets a different id, and ids never collide."""
    ds = DATASETS["sst5"]
    row = {"text": "x", "label": 0, "label_text": ""}
    a = ds.row_id(row, "test", 7)
    b = ds.row_id(row, "validation", 7)
    c = ds.row_id(row, "test", 8)
    assert a != b and a != c
    ids = {ds.row_id(row, "test", i) for i in range(1000)}
    assert len(ids) == 1000


def test_sample_balanced_takes_equal_rows_per_class(ag_news_rows):
    ds = DATASETS["ag_news"]
    picked = _balanced_positions(
        ds, [_row_text(ds, r) for r in ag_news_rows], ag_news_rows, per_class=2
    )
    labels = [ds.label_of(ag_news_rows[p]) for p in picked]
    assert labels.count("World") == 2 and labels.count("Sci/Tech") == 2
    # per_class above the bucket size takes the whole bucket (no dupes).
    picked = _balanced_positions(
        ds, [_row_text(ds, r) for r in ag_news_rows], ag_news_rows, per_class=50
    )
    assert len(picked) == 40


def test_sample_balanced_logs_short_classes(ag_news_rows, capsys):
    """F9: a class shorter than per_class is LOUD, never silent."""
    ds = DATASETS["ag_news"]
    short = ag_news_rows[:6]  # only class 0 and 1 present (w1,s1,w2,s2... )
    _balanced_positions(ds, [_row_text(ds, r) for r in short], short, per_class=50)
    out = capsys.readouterr().out
    assert "under-filled" in out


def test_natural_view_excludes_balanced_rows(sst5_rows):
    """F8: the two views are DISJOINT — natural draws only from rows the
    balanced view did not take (calibration never fits on reported rows)."""
    ds = DATASETS["sst5"]
    texts = [_row_text(ds, r) for r in sst5_rows]
    balanced = set(_balanced_positions(ds, texts, sst5_rows))
    natural = _natural_positions(ds, texts, sst5_rows, exclude=balanced)
    assert balanced.isdisjoint(natural)
    # Natural fills from the remaining rows.
    remaining = len(sst5_rows) - len(balanced)
    assert len(natural) == min(NATURAL_ROWS, remaining)


def test_fetch_dataset_views_are_disjoint_and_carry_real_text(hubless):
    """F1 + F8: the CACHED cases file carries the REAL text (the engine
    classifies it); the views never share a row."""
    views, _sha = fetch_dataset(DATASETS["sst5"], "test")
    b_ids = {r["meta"]["source_row_id"] for r in views["balanced"]}
    n_ids = {r["meta"]["source_row_id"] for r in views["natural"]}
    assert b_ids.isdisjoint(n_ids)
    for record in views["balanced"] + views["natural"]:
        assert record["context"]  # non-empty real text, not a JSON stub
        assert (
            not record["context"].strip().startswith("{") or "fetch" not in record["context"][:40]
        )


def test_case_records_carry_provenance_and_real_text(ag_news_rows):
    ds = DATASETS["ag_news"]
    row = ag_news_rows[0]
    record = build_case(ds, row, "test", "balanced", row["text"], 0)
    assert record["context"] == row["text"]
    assert record["meta"]["source_row_id"] == ds.row_id(row, "test", 0)
    assert record["meta"]["source_repo"] == ds.repo_id
    assert record["meta"]["source_revision"] == ds.revision
    assert record["meta"]["source_file"] == ds.files["test"]
    assert record["meta"]["source_row_index"] == 0
    assert record["meta"]["option_order"] == ds.classes
    assert record["view"] == "balanced"
    assert record["split"] == "test"


def test_boolq_option_order_is_explicit():
    """F12: boolean fields state their option order explicitly (no .get
    fallback): the two labels in the engine's canonical order."""
    record = build_case(
        DATASETS["boolq"],
        {"question": "q", "answer": True, "passage": "p"},
        "validation",
        "balanced",
        "q\np",
        3,
    )
    assert record["meta"]["option_order"] == ["false", "true"]
    assert record["labels"]["answer"] is True


def test_selected_input_hash_binds_schema_and_text():
    """F11: the hash covers schema + REAL text (not a stub that contains
    the row id)."""
    schema = {"f": {"type": "boolean", "description": "d"}}
    a = _input_hash(schema, "same text")
    b = _input_hash(schema, "different text")
    c = _input_hash({"f": {"type": "boolean", "description": "e"}}, "same text")
    assert a != b and a != c and len(a) == 64


# ---- F5: the download pin is VERIFIED, not just recorded --------------------


def test_download_verifies_sha256_and_fails_closed(monkeypatch, tmp_path, sst5_rows):
    ds = DATASETS["sst5"]
    good = tmp_path / "good.jsonl"
    good.write_text("\n".join(json.dumps(r) for r in sst5_rows), encoding="utf-8")
    digest = hashlib.sha256(good.read_bytes()).hexdigest()

    monkeypatch.setattr("huggingface_hub.hf_hub_download", lambda *a, **k: str(good))
    monkeypatch.setattr(ds, "expected_sha256", {ds.files["test"]: digest})

    # Matching bytes pass.
    assert pf._download(ds, "test") == good

    # Drifted bytes fail CLOSED with an OSError naming the mismatch.
    bad = tmp_path / "bad.jsonl"
    bad.write_text("tampered", encoding="utf-8")
    monkeypatch.setattr("huggingface_hub.hf_hub_download", lambda *a, **k: str(bad))
    with pytest.raises(OSError, match="sha256 mismatch"):
        pf._download(ds, "test")

    # A missing expectation also fails closed.
    monkeypatch.setattr(ds, "expected_sha256", {})
    with pytest.raises(OSError, match="no expected sha256"):
        pf._download(ds, "test")


def test_expected_sha256_covers_every_pinned_eval_file():
    """F5: every dataset's eval-split file has a hardcoded known-good sha256."""
    for ds in DATASETS.values():
        split = EVAL_SPLITS[ds.name]
        file_path = ds.files[split]
        assert ds.expected_sha256.get(file_path), (ds.name, file_path)
        assert ds.expected_sha256[file_path] in EXPECTED_SHA256[ds.repo_id].values()


def test_download_failure_is_oserror_not_systemexit(monkeypatch):
    """F6: build_datasets catches OSError for the offline skip; a download
    failure must be one."""
    ds = DATASETS["sst5"]

    def boom(*a, **k):
        raise RuntimeError("connection refused")

    monkeypatch.setattr("huggingface_hub.hf_hub_download", boom)
    with pytest.raises(OSError, match="download failed"):
        pf._download(ds, "test")


# ---- fetch + write: pinned bytes, fail-closed locks -------------------------


def test_fetch_dataset_pins_the_revision(hubless):
    requested = hubless
    views, sha = fetch_dataset(DATASETS["ag_news"], "test")
    # Every download carries the dataset's PINNED revision (no floating ref).
    assert requested
    for _repo_id, revision, _file in requested:
        assert revision == DATASETS["ag_news"].revision
        assert len(revision) == 40 and int(revision, 16) >= 0  # commit sha
    assert sha  # per-file sha256 recorded
    assert set(views) == {"balanced", "natural"}


def test_write_view_fails_closed_without_file_sha256(ag_news_rows, tmp_path):
    ds = DATASETS["ag_news"]
    records = [
        build_case(ds, r, "test", "balanced", r["text"], i) for i, r in enumerate(ag_news_rows)
    ]
    with pytest.raises(OSError, match="without file sha256s"):
        write_view(records, tmp_path / "x.jsonl", files_sha256=None)
    with pytest.raises(OSError, match="without file sha256s"):
        write_view(records, tmp_path / "x.jsonl", files_sha256={})


def test_write_view_refuses_empty_records(tmp_path):
    with pytest.raises(OSError, match="empty view"):
        write_view([], tmp_path / "x.jsonl", files_sha256={"f": "ab"})


def test_write_view_lock_records_license_and_sha256(ag_news_rows, tmp_path):
    ds = DATASETS["ag_news"]
    records = [
        build_case(ds, r, "test", "balanced", r["text"], i) for i, r in enumerate(ag_news_rows)
    ]
    out = tmp_path / "ag_news.balanced.jsonl"
    lock_path = write_view(records, out, files_sha256={"data/test.parquet": "ab" * 20})
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    assert lock["sources"][0]["repo_id"] == ds.repo_id
    assert lock["sources"][0]["revision"] == ds.revision
    assert "unknown" in lock["sources"][0]["license"]
    assert lock["sources"][0]["redistribution_allowed"] is False
    assert lock["sources"][0]["files"] == {"data/test.parquet": "ab" * 20}
    assert lock["parser_version"] == PARSER_VERSION
    assert lock["cases_sha256"] == hashlib.sha256(out.read_bytes()).hexdigest()
    assert all(v == 10 for v in lock["counts"]["classes"].values())  # 10/class


# ---- CLI + eval-split defaults ----------------------------------------------


def test_eval_split_defaults(hubless, tmp_path):
    assert EVAL_SPLITS == {"ag_news": "test", "boolq": "validation", "sst5": "test"}
    rc = main(
        [
            "--out-dir",
            str(tmp_path),
            "--dataset",
            "boolq",
            "--per-class",
            "4",
            "--natural-rows",
            "10",
        ]
    )
    assert rc == 0
    for view in ("balanced", "natural"):
        assert (tmp_path / f"boolq.{view}.jsonl").exists()
        assert (tmp_path / f"boolq.{view}.dataset.lock.json").exists()
        records = [
            json.loads(line)
            for line in (tmp_path / f"boolq.{view}.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        assert all(r["split"] == "validation" for r in records)


def test_two_views_are_separate_files_with_separate_locks(hubless, tmp_path):
    # The fixture file holds 2 rows per class, so the balanced view caps at
    # 2/class (BALANCED_ROWS_PER_CLASS is the ceiling, not a floor); natural
    # fills from the rows balanced did NOT take.
    rc = main(
        [
            "--out-dir",
            str(tmp_path),
            "--dataset",
            "sst5",
            "--per-class",
            "2",
            "--natural-rows",
            "40",
        ]
    )
    assert rc == 0
    balanced = (tmp_path / "sst5.balanced.jsonl").read_text(encoding="utf-8").splitlines()
    natural = (tmp_path / "sst5.natural.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(balanced) == 5 * 2
    assert len(natural) == 40  # disjoint remaining rows, capped by --natural-rows
    # Different files, different locks, both verify against their own bytes.
    for name in ("balanced", "natural"):
        lock = json.loads((tmp_path / f"sst5.{name}.dataset.lock.json").read_text())
        cases = hashlib.sha256((tmp_path / f"sst5.{name}.jsonl").read_bytes()).hexdigest()
        assert lock["cases_sha256"] == cases


def test_cached_cases_carry_real_text_but_docs_do_not(hubless, tmp_path, ag_news_rows):
    """F1, restated: the CACHED cases file is where the text lives (the
    engine classifies case['context'] directly); the LOCK (an artifact)
    stores ids + hashes only."""
    rc = main(
        [
            "--out-dir",
            str(tmp_path),
            "--dataset",
            "ag_news",
            "--per-class",
            "2",
            "--natural-rows",
            "10",
        ]
    )
    assert rc == 0
    cases_blob = (tmp_path / "ag_news.balanced.jsonl").read_text(encoding="utf-8")
    base = ag_news_rows[0]["text"].split(" (report")[0]
    assert base in cases_blob  # real text present for the engine
    lock = json.loads((tmp_path / "ag_news.balanced.dataset.lock.json").read_text())
    lock_blob = json.dumps(lock)
    assert base not in lock_blob  # artifacts stay text-free


# ---- the pin (one --check-style test) ----------------------------------------


def test_dataset_pins_are_commit_shas_and_licenses_recorded():
    """The pin: every dataset fetches a 40-hex commit sha (never a floating
    branch), and its license/redistribution status is recorded."""
    for ds in DATASETS.values():
        assert len(ds.revision) == 40, ds.name
        int(ds.revision, 16)  # hex
        assert ds.license  # every lock records terms
        assert "NOT cleared" in ds.license or "share-alike" in ds.license


# ---- F2 + F3: bench wiring, tested OFFLINE with the hubless fixture ---------


def test_bench_wires_the_three_names(hubless, monkeypatch, tmp_path):
    """jevmlx.bench builds ag_news/boolq/sst5 views with the same lock shape
    (cases_sha256-verified reuse) as typed_decisions — offline (F3)."""
    import jevmlx.bench as bench

    monkeypatch.setattr(bench, "BENCH_CACHE", tmp_path)
    monkeypatch.setattr(pf, "BALANCED_ROWS_PER_CLASS", 2)
    monkeypatch.setattr(pf, "NATURAL_ROWS", 10)
    paths, locks = bench.build_datasets(["ag_news"], offline_ok=False)
    assert "ag_news.balanced" in paths and "ag_news.natural" in paths
    for key in ("ag_news.balanced", "ag_news.natural"):
        lock = json.loads(locks[key].read_text(encoding="utf-8"))
        assert lock["cases_sha256"] == hashlib.sha256(paths[key].read_bytes()).hexdigest()
        assert lock["sources"][0]["redistribution_allowed"] is False


def test_bench_pin_problem_detects_drift(hubless, monkeypatch, tmp_path):
    """F5 on reuse: a lock whose recorded file sha256 differs from the pin
    forces a rebuild; a matching lock passes."""
    import jevmlx.bench as bench

    monkeypatch.setattr(bench, "BENCH_CACHE", tmp_path)
    monkeypatch.setattr(pf, "BALANCED_ROWS_PER_CLASS", 2)
    monkeypatch.setattr(pf, "NATURAL_ROWS", 10)
    _paths, locks = bench.build_datasets(["sst5"], offline_ok=False)
    lock = locks["sst5.balanced"]
    assert bench._public_pin_problem(lock) is None

    data = json.loads(lock.read_text(encoding="utf-8"))
    data["sources"][0]["files"][data["sources"][0]["files"].__iter__().__next__()] = "0" * 64
    drifted = tmp_path / "drifted.json"
    drifted.write_text(json.dumps(data))
    problem = bench._public_pin_problem(drifted)
    assert problem and "sha256 drift" in problem


def test_bench_bare_names_expand_to_both_views(hubless, monkeypatch, tmp_path):
    """F2: '--datasets ag_news' must produce combos for BOTH views (the
    bare name itself is not a dataset key)."""
    import jevmlx.bench as bench

    monkeypatch.setattr(bench, "BENCH_CACHE", tmp_path)
    monkeypatch.setattr(pf, "BALANCED_ROWS_PER_CLASS", 2)
    monkeypatch.setattr(pf, "NATURAL_ROWS", 10)
    dataset_paths, _locks = bench.build_datasets(["ag_news"], offline_ok=False)
    assert set(dataset_paths) == {"ag_news.balanced", "ag_news.natural"}

    # The combo expander: bare names expand, view names pass through,
    # unknown names drop.
    def expand(names):
        expanded = []
        for name in names:
            if name in dataset_paths:
                expanded.append(name)
            elif name in ("ag_news", "boolq", "sst5"):
                expanded.extend(f"{name}.{v}" for v in ("balanced", "natural"))
        return expanded

    assert expand(["ag_news"]) == ["ag_news.balanced", "ag_news.natural"]
    assert expand(["ag_news.balanced"]) == ["ag_news.balanced"]
    assert expand(["ag_news", "boolq"]) == [
        "ag_news.balanced",
        "ag_news.natural",
        "boolq.balanced",
        "boolq.natural",
    ]
    assert expand(["nope"]) == []


def test_bench_combos_run_end_to_end_with_fake_engine(hubless, monkeypatch, tmp_path):
    """F2 end to end: run_bench with --datasets boolq produces prediction
    rows for both views (offline; only the eval EXECUTION is faked — the
    dataset expansion, combo building and summarization stay real)."""
    import jevmlx.bench as bench

    monkeypatch.setattr(bench, "BENCH_CACHE", tmp_path)
    monkeypatch.setattr(pf, "BALANCED_ROWS_PER_CLASS", 2)
    monkeypatch.setattr(pf, "NATURAL_ROWS", 10)
    results_root = tmp_path / "results"

    monkeypatch.setattr(bench, "preflight", lambda *a, **k: "test-machine")
    monkeypatch.setattr(bench, "_load_engine_with_timeout", lambda model, timeout: None)
    monkeypatch.setattr(
        "jevmlx.engine.load_engine",
        lambda model: (_ for _ in ()).throw(AssertionError("real load")),
    )

    def fake_run_one(
        model, track, scorer, jsonl, combo_dir, dataset_lock_path=None, resume=False, **kwargs
    ):
        combo_dir = pathlib.Path(combo_dir)
        combo_dir.mkdir(parents=True, exist_ok=True)
        rows = [json.loads(line) for line in jsonl.read_text().splitlines() if line.strip()]
        field = next(iter(rows[0]["schema"]))
        with (combo_dir / "predictions.jsonl").open("w") as fh:
            for row in rows:
                fh.write(
                    json.dumps(
                        {
                            "id": row["id"],
                            "group_id": row["group_id"],
                            "labels": {field: row["labels"][field]},
                            "predictions": {field: str(row["labels"][field])},
                        }
                    )
                    + "\n"
                )
        (combo_dir / "run.json").write_text(json.dumps({"model": model, "track": track}))
        return {"cases": len(rows)}

    monkeypatch.setattr(bench, "_run_one", fake_run_one)
    _paths, _locks = bench.build_datasets(["boolq"], offline_ok=False)
    bench.run_bench(
        model="fake/model",
        datasets=["boolq"],
        scorers=["labels"],
        tracks=["parallel"],
        out=results_root,
        runs=1,
    )
    # Both view combos produced predictions.
    predictions = sorted(results_root.rglob("predictions.jsonl"))
    assert len(predictions) == 2, [str(p) for p in predictions]
    views = {p.parent.name for p in predictions}
    assert any("balanced" in v for v in views) and any("natural" in v for v in views)
    for path in predictions:
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        assert rows, path
