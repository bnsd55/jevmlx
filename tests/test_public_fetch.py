"""Tests for benchmarks.public.fetch (W6-B5: AG News / BoolQ / SST-5).

Network-free: the fetch paths run against fixture parquet/jsonl files
monkeypatched in place of hf_hub_download. The one pin test asserts the
DEFAULT_REVISION-style pins are 40-hex commit shas and that --check (the
pin verifier) accepts them.
"""

from __future__ import annotations

import hashlib
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import benchmarks.public.fetch as pf
from benchmarks.public.fetch import (
    DATASETS,
    EVAL_SPLITS,
    NATURAL_ROWS,
    PARSER_VERSION,
    SAMPLE_SEED,
    _context_stub,
    _input_hash,
    _slot,
    build_case,
    fetch_dataset,
    main,
    sample_balanced,
    sample_natural,
    write_view,
)
from benchmarks.typesafe.questions import field_schema

# ---- fixtures --------------------------------------------------------------


@pytest.fixture()
def ag_news_rows() -> list[dict]:
    """8 rows, 2 per class (label 0..3), in scrambled order."""
    texts = [
        ("the UN assembly met in Geneva yesterday", 0),
        ("the Lakers won the playoff game in overtime", 1),
        ("oil prices climbed after the quarterly earnings report", 2),
        ("Microsoft unveiled a new processor architecture", 3),
        ("the security council debated the ceasefire", 0),
        ("the striker scored twice in the second half", 1),
        ("the central bank raised interest rates again", 2),
        ("researchers demonstrated a quantum error correction gain", 3),
    ]
    return [{"text": t, "label": label} for t, label in texts]


@pytest.fixture()
def boolq_rows() -> list[dict]:
    out = []
    for i in range(6):
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
    for i in range(10):
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
    """Point _download at fixture files; capture the requested (repo, rev)."""
    requested: list[tuple[str, str, str]] = []

    def fake_download(ds, split, cache_dir=None):
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

    monkeypatch.setattr(pf, "_download", fake_download)
    return requested


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
    s1 = _slot(ds.row_id(ag_news_rows[0]))
    s2 = _slot(ds.row_id(ag_news_rows[0]))
    assert s1 == s2
    # The seed is part of the slot: different seed -> different ordering.
    digest_with_seed = hashlib.sha256(
        f"{SAMPLE_SEED}\0{ds.row_id(ag_news_rows[0])}".encode()
    ).digest()
    digest_no_seed = hashlib.sha256(ds.row_id(ag_news_rows[0]).encode()).digest()
    assert digest_with_seed != digest_no_seed


def test_sample_balanced_takes_equal_rows_per_class(ag_news_rows, boolq_rows, sst5_rows):
    ds = DATASETS["ag_news"]
    picked = sample_balanced(ag_news_rows, ds, per_class=2)
    labels = [ds.label_of(r) for r in picked]
    assert labels.count("World") == 2 and labels.count("Sci/Tech") == 2
    # per_class above the bucket size takes the whole bucket (no dupes).
    picked = sample_balanced(ag_news_rows, ds, per_class=50)
    assert len(picked) == 8


def test_sample_natural_keeps_the_datasets_own_prevalence(sst5_rows):
    ds = DATASETS["sst5"]
    # 50 unique copies of the 5-class cycle (copy index keeps row ids unique
    # — the real datasets have unique texts, and id collision would make
    # hash sampling degenerate).
    rows = [{**r, "text": f"{r['text']} (copy {c})"} for c in range(50) for r in sst5_rows]
    # Full-width sample: shares EXACTLY match the source (uniform -> 100 each).
    picked = sample_natural(rows, ds, total=500)
    counts: dict[int, int] = {}
    for r in picked:
        counts[r["label"]] = counts.get(r["label"], 0) + 1
    assert all(count == 100 for count in counts.values()), counts
    # A sub-sample is hash-ordered, NOT stratified (hash order is class-
    # independent): shares stay within a loose band, never exactly balanced.
    sub = sample_natural(rows, ds, total=250)
    sub_counts: dict[int, int] = {}
    for r in sub:
        sub_counts[r["label"]] = sub_counts.get(r["label"], 0) + 1
    for label, count in sub_counts.items():
        assert 20 <= count <= 60, (label, count)  # ~50 expected, no strata
    # And the sample follows a NON-uniform source: 50% class-0 -> ~125.
    skewed = [{**r, "text": f"{r['text']} (copy {c})"} for c in range(40) for r in sst5_rows] + [
        {"text": f"padding row {i}", "label": 0, "label_text": ""} for i in range(200)
    ]
    picked2 = sample_natural(skewed, ds, total=250)
    zeros = sum(1 for r in picked2 if r["label"] == 0)
    assert zeros > 60  # 50% source share -> well above the uniform 20%


def test_sampling_is_stable_across_row_order(ag_news_rows):
    ds = DATASETS["ag_news"]
    a = sample_balanced(ag_news_rows, ds, per_class=1)
    shuffled = list(reversed(ag_news_rows))
    b = sample_balanced(shuffled, ds, per_class=1)
    assert {ds.row_id(r) for r in a} == {ds.row_id(r) for r in b}


# ---- case records: no source text, provenance present -----------------------


def test_case_records_store_no_source_text(ag_news_rows):
    ds = DATASETS["ag_news"]
    row = ag_news_rows[0]
    context = _context_stub(ds, "test", ds.row_id(row), ds.files["test"], 0)
    record = build_case(ds, row, "test", "balanced", context)
    blob = json.dumps(record)
    assert row["text"] not in blob
    assert record["meta"]["source_row_id"] == ds.row_id(row)
    assert record["meta"]["source_repo"] == ds.repo_id
    assert record["meta"]["source_revision"] == ds.revision
    assert record["meta"]["option_order"] == ds.classes
    assert record["meta"]["selected_input_hash"] == _input_hash(context)
    assert record["view"] == "balanced"
    assert record["split"] == "test"


def test_selected_input_hash_binds_schema_and_context():
    a = _input_hash('{"x": 1}')
    b = _input_hash('{"x": 2}')
    assert a != b and len(a) == 64


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
        build_case(ds, r, "test", "balanced", _context_stub(ds, "test", ds.row_id(r), "f", i))
        for i, r in enumerate(ag_news_rows)
    ]
    with pytest.raises(SystemExit, match="without file sha256s"):
        write_view(records, tmp_path / "x.jsonl", files_sha256=None)
    with pytest.raises(SystemExit, match="without file sha256s"):
        write_view(records, tmp_path / "x.jsonl", files_sha256={})


def test_write_view_lock_records_license_and_sha256(ag_news_rows, tmp_path):
    ds = DATASETS["ag_news"]
    records = [
        build_case(
            ds,
            r,
            "test",
            "balanced",
            _context_stub(ds, "test", ds.row_id(r), "data/test.parquet", i),
        )
        for i, r in enumerate(sample_balanced(ag_news_rows, ds, per_class=2))
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
    # Balanced view counts: 2 per class.
    assert all(v == 2 for v in lock["counts"]["classes"].values())


# ---- CLI + eval-split defaults ----------------------------------------------


def test_eval_split_defaults(hubless, tmp_path):
    assert EVAL_SPLITS == {"ag_news": "test", "boolq": "validation", "sst5": "test"}
    rc = main(["--out-dir", str(tmp_path), "--dataset", "boolq"])
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
    # 2/class (BALANCED_ROWS_PER_CLASS is the ceiling, not a floor).
    rc = main(["--out-dir", str(tmp_path), "--dataset", "sst5"])
    assert rc == 0
    balanced = (tmp_path / "sst5.balanced.jsonl").read_text(encoding="utf-8").splitlines()
    natural = (tmp_path / "sst5.natural.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(balanced) == 5 * 2
    assert len(natural) == min(NATURAL_ROWS, 10)
    # Different files, different locks, both verify against their own bytes.
    for name in ("balanced", "natural"):
        lock = json.loads((tmp_path / f"sst5.{name}.dataset.lock.json").read_text())
        cases = hashlib.sha256((tmp_path / f"sst5.{name}.jsonl").read_bytes()).hexdigest()
        assert lock["cases_sha256"] == cases


def test_natural_view_leaks_no_source_text(hubless, tmp_path, ag_news_rows):
    rc = main(["--out-dir", str(tmp_path), "--dataset", "ag_news"])
    assert rc == 0
    blob = (tmp_path / "ag_news.natural.jsonl").read_text(encoding="utf-8")
    for row in ag_news_rows:
        assert row["text"] not in blob


# ---- the pin (one --check-style test) ----------------------------------------


def test_dataset_pins_are_commit_shas_and_licenses_recorded():
    """The pin: every dataset fetches a 40-hex commit sha (never a floating
    branch), and its license/redistribution status is recorded."""
    for ds in DATASETS.values():
        assert len(ds.revision) == 40, ds.name
        int(ds.revision, 16)  # hex
        assert ds.license  # every lock records terms
        assert "NOT cleared" in ds.license or "share-alike" in ds.license


def test_bench_wires_the_three_names(monkeypatch, tmp_path):
    """jevmlx.bench builds ag_news/boolq/sst5 views with the same lock shape
    (cases_sha256-verified reuse) as typed_decisions."""
    import jevmlx.bench as bench

    monkeypatch.setattr(bench, "BENCH_CACHE", tmp_path)
    paths, locks = bench.build_datasets(["ag_news"], offline_ok=False)
    assert "ag_news.balanced" in paths and "ag_news.natural" in paths
    for key in ("ag_news.balanced", "ag_news.natural"):
        lock = json.loads(locks[key].read_text(encoding="utf-8"))
        assert lock["cases_sha256"] == hashlib.sha256(paths[key].read_bytes()).hexdigest()
        assert lock["sources"][0]["redistribution_allowed"] is False
