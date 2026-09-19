"""Fetch the three PUBLIC gold datasets (W6-B5) as jevmlx eval JSONL.

- AG News (``fancyzhx/ag_news``): 4-class topic classification — enum.
- BoolQ (``google/boolq``): reading-comprehension yes/no — boolean (noul).
- SST-5 (``SetFit/sst5``): ordinal sentiment, 5 levels — ordered enum
  (the B1 derivation: the ORDER lives in the choices list and the level
  texts in the description; no new engine field type).

Every dataset is PINNED by its repo commit sha (``DEFAULT_REVISION`` per
dataset, like typed_decisions/fetch.py), every downloaded file's sha256
lands in the lock, and a mismatch FAILS CLOSED (OSError) instead of
writing a lock over unknown bytes.

Two sampling views per dataset, written as SEPARATE cases files
(``<name>.balanced.jsonl`` / ``<name>.natural.jsonl``):

- **balanced** (diagnostic): equal rows per class for per-class accuracy,
  macro-F1, confusion matrices, ordinal MAE. NOT a deployment estimate.
- **natural** (calibration): the dataset's own class prevalence, the view
  NLL / Brier / ECE describe.

Selection is DETERMINISTIC: ``hash(seed, source_row_id)`` (sha256, first
8 bytes) rank-orders each class bucket; the same seed+revision always
yields the same rows.

License/terms (from each dataset's card, recorded in the lock):
AG News "unknown", BoolQ CC-BY-SA 3.0, SST-5 unspecified. Redistribution
status is therefore NOT cleared for any of them, so the SOURCE TEXT lives
only in the CACHED cases files under ``BENCH_CACHE`` (never committed,
re-expanded from the pinned Hub bytes) — while every RESULTS ARTIFACT
(predictions.jsonl, run.json, report.json, dataset.lock.json) stores row
ids, split, option order and input hashes only. The two views are
DISJOINT: the natural-distribution view draws rows the balanced view did
NOT take, so calibration rows are never the reported diagnostic rows.

Every dataset file carries a hardcoded EXPECTED sha256 (next to the
pinned revision); the download is verified against it and the verified
value is re-checked against the lock on cache reuse. A mismatch or a
missing expectation FAILS CLOSED (OSError) — unknown bytes are never
sampled or locked.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from benchmarks.typesafe.questions import field_schema

PARSER_VERSION = "1"

BALANCED_ROWS_PER_CLASS = 50
NATURAL_ROWS = 500
SAMPLE_SEED = "jevmlx-public-gold-v1"


class PublicDataset:
    """One public dataset's fetch + sampling config (name + repo + pin).

    ``expected_sha256`` maps each file to its KNOWN-GOOD digest at the
    pinned revision (F5): the download is verified against it and the
    value is re-checked on cache reuse. ``row_id`` takes (row, split,
    index) — split-aware and duplicate-free (F10).
    """

    def __init__(
        self,
        name: str,
        repo_id: str,
        revision: str,
        license_name: str,
        files: dict[str, str],
        expected_sha256: dict[str, str],
        row_reader,
        schema_builder,
        label_of,
        row_id,
        classes,
    ):
        self.name = name
        self.repo_id = repo_id
        self.revision = revision
        self.license = license_name
        self.files = files  # split -> repo-relative file path
        self.expected_sha256 = expected_sha256  # file -> known-good digest
        self.row_reader = row_reader  # local path -> iterable of rows
        self.schema_builder = schema_builder  # () -> schema dict (shared mapping)
        self.label_of = label_of  # row -> gold label (label_form output)
        self.row_id = row_id  # (row, split, index) -> stable source row id
        self.classes = classes  # ordered class list


def _read_parquet(path: Path):
    import pyarrow.parquet as pq

    return pq.read_table(path).to_pylist()


def _read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


AG_NEWS_CLASSES = ["World", "Sports", "Business", "Sci/Tech"]
# fancyzhx/ag_news encodes classes as 0..3 in this order (dataset card).
AG_NEWS_LABELS = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}

BOOLQ_CLASSES = ["false", "true"]  # the boolean field's two options

SST5_CLASSES = ["0", "1", "2", "3", "4"]  # very negative .. very positive
SST5_LEVELS = "0 = very negative; 1 = negative; 2 = neutral; 3 = positive; 4 = very positive"


def _ag_news_schema() -> dict:
    """AG News: 4-class topic enum via the SHARED mapping (choice shape)."""
    question = {
        "type": "choice",
        "instructions": "Which topic does this news story belong to?",
        "criteria": dict.fromkeys(AG_NEWS_CLASSES, ""),
    }
    field = field_schema(question)
    assert field is not None
    field["description"] = (
        "Which topic does this news story belong to? "
        "Choices: World = world news; Sports = sports; "
        "Business = business and finance; Sci/Tech = science and technology."
    )
    field.pop("choice_descriptions", None)
    return {"topic": field}


def _boolq_schema() -> dict:
    """BoolQ: yes/no reading comprehension via the SHARED mapping (noul)."""
    field = field_schema(
        {
            "type": "noul",
            "instructions": ("Answer the question using only the passage: is the answer yes?"),
        }
    )
    assert field is not None
    field["description"] = (
        "Answer the question using only the passage: is the answer yes? "
        "The passage and question follow."
    )
    return {"answer": field}


def _sst5_schema() -> dict:
    """SST-5: ordinal 5-level sentiment via the SHARED mapping (score
    shape, widened to 0-4). The ORDER (very negative .. very positive) is
    the enum's choice order — ordinal MAE reads from it."""
    question = {
        "type": "score",
        "instructions": "Rate the sentiment of this movie-review phrase.",
        "criteria": [
            "very negative",
            "negative",
            "neutral",
            "positive",
            "very positive",
        ],
    }
    # SST-5 is ordinal 0-4: the shared mapping's scale override (one
    # mapping owner, no local copy).
    field = field_schema(question, score_choices=SST5_CLASSES)
    assert field is not None
    field["description"] = f"Rate the sentiment of this movie-review phrase. Scale: {SST5_LEVELS}."
    return {"sentiment": field}


# The canonical EVAL split per dataset (the fetcher's default; train stays
# available for future few-shot work but is not sampled by the bench).
EVAL_SPLITS = {"ag_news": "test", "boolq": "validation", "sst5": "test"}

# F5: the KNOWN-GOOD sha256 of every pinned file (verified on download and
# re-checked on cache reuse). A mismatch fails closed — these are the bytes
# the locks are computed against.
EXPECTED_SHA256 = {
    "fancyzhx/ag_news": {
        "data/test-00000-of-00001.parquet": (
            "71de87ec66bc5737752a2502204dfa6d7fe9856ade3ea444dc6317789a4f13fb"
        ),
    },
    "google/boolq": {
        "data/validation-00000-of-00001.parquet": (
            "52355d11524b4b874a9b9dcc278feb10f672d52c4f4eff9872e695ede59820f8"
        ),
    },
    "SetFit/sst5": {
        "test.jsonl": "1384216112a34f3d70b6fa210762f3399bb080410c0456ac6e54a5cb413f04b2",
    },
}

DATASETS: dict[str, PublicDataset] = {
    "ag_news": PublicDataset(
        name="ag_news",
        repo_id="fancyzhx/ag_news",
        revision="eb185aade064a813bc0b7f42de02595523103ca4",
        license_name="unknown (dataset card: no license listed; "
        "redistribution NOT cleared — text stays in the uncommitted cache)",
        files={
            "train": "data/train-00000-of-00001.parquet",
            "test": "data/test-00000-of-00001.parquet",
        },
        expected_sha256=EXPECTED_SHA256["fancyzhx/ag_news"],
        row_reader=_read_parquet,
        schema_builder=_ag_news_schema,
        label_of=lambda row: AG_NEWS_LABELS[int(row["label"])],
        row_id=lambda row, split, index: f"{split}/{index:08d}",
        classes=AG_NEWS_CLASSES,
    ),
    "boolq": PublicDataset(
        name="boolq",
        repo_id="google/boolq",
        revision="35b264d03638db9f4ce671b711558bf7ff0f80d5",
        license_name="CC-BY-SA 3.0 (share-alike; redistribution NOT "
        "cleared — text stays in the uncommitted cache)",
        files={
            "train": "data/train-00000-of-00001.parquet",
            "validation": "data/validation-00000-of-00001.parquet",
        },
        expected_sha256=EXPECTED_SHA256["google/boolq"],
        row_reader=_read_parquet,
        schema_builder=_boolq_schema,
        label_of=lambda row: bool(row["answer"]),
        row_id=lambda row, split, index: f"{split}/{index:08d}",
        classes=BOOLQ_CLASSES,
    ),
    "sst5": PublicDataset(
        name="sst5",
        repo_id="SetFit/sst5",
        revision="e51bdcd8cd3a30da231967c1a249ba59361279a3",
        license_name="unspecified (dataset card lists no license; "
        "redistribution NOT cleared — text stays in the uncommitted cache)",
        files={"dev": "dev.jsonl", "test": "test.jsonl"},
        expected_sha256=EXPECTED_SHA256["SetFit/sst5"],
        row_reader=_read_jsonl,
        schema_builder=_sst5_schema,
        label_of=lambda row: str(int(row["label"])),
        row_id=lambda row, split, index: f"{split}/{index:08d}",
        classes=SST5_CLASSES,
    ),
}


# ---- deterministic sampling ----------------------------------------------


def _slot(source_row_id: str) -> int:
    """Deterministic slot: first 8 bytes of sha256(seed, id), big-endian.

    Same (seed, id) -> same slot at every dataset revision, so the sample
    is reproducible across runs and machines.
    """
    digest = hashlib.sha256(f"{SAMPLE_SEED}\0{source_row_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


# F11: the selected-input hash covers the REAL rendered content — schema
# (sorted, stable) + the case's real text — not a stub that already contains
# the row id (which would be circular).
def _input_hash(schema: dict, text: str) -> str:
    payload = json.dumps({"schema": schema, "text": text}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_case(
    ds: PublicDataset,
    row: dict,
    split: str,
    view: str,
    context: str,
    row_index: int,
) -> dict:
    """One eval record for either view.

    The cached cases file carries the REAL TEXT (the engine classifies it);
    results artifacts (predictions/run/lock) never do — they carry the
    row id, split, option order and the selected-input hash only.
    """
    schema = ds.schema_builder()
    field = next(iter(schema.values()))
    # F12: explicit option order per field type — enum: its choices;
    # boolean: the two labels in the engine's canonical order.
    option_order = field["choices"] if field["type"] == "enum" else list(BOOLQ_CLASSES)
    record_id = f"{ds.name}/{view}/{ds.row_id(row, split, row_index)}"
    return {
        "id": record_id,
        "group_id": record_id,
        "source": ds.name,
        "view": view,
        "benchmark_only": True,
        "schema": schema,
        "context": context,
        "labels": {next(iter(schema)): ds.label_of(row)},
        "split": split,
        "meta": {
            "source_row_id": ds.row_id(row, split, row_index),
            "source_repo": ds.repo_id,
            "source_revision": ds.revision,
            "source_file": ds.files[split],
            "source_row_index": row_index,
            "option_order": option_order,
            "selected_input_hash": _input_hash(schema, context),
        },
    }


# ---- fetch + write --------------------------------------------------------


def _download(ds: PublicDataset, split: str) -> Path:
    """Download the pinned file and VERIFY it (F5): the digest must equal the
    hardcoded expectation for the pinned revision; any mismatch is a hard
    OSError (never sampled, never locked)."""
    from huggingface_hub import hf_hub_download

    file_path = ds.files[split]
    try:
        path = Path(
            hf_hub_download(
                ds.repo_id,
                file_path,
                repo_type="dataset",
                revision=ds.revision,
            )
        )
    except Exception as exc:  # noqa: BLE001 — one clear message for any cause
        # F6: OSError (not SystemExit) so build_datasets' offline skip works.
        raise OSError(
            f"{ds.repo_id}@{ds.revision} [{file_path}]: download failed "
            f"({exc.__class__.__name__}). The fetch needs the PINNED revision "
            "in the local HF cache; locally run: huggingface-cli download "
            f"{ds.repo_id} --repo-type dataset --revision {ds.revision}"
        ) from exc
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    expected = ds.expected_sha256.get(file_path)
    if expected is None:
        raise OSError(
            f"{ds.repo_id}@{ds.revision} [{file_path}]: no expected sha256 "
            "recorded for this file — refusing to sample unknown bytes"
        )
    if actual != expected:
        raise OSError(
            f"{ds.repo_id}@{ds.revision} [{file_path}]: sha256 mismatch — "
            f"expected {expected}, got {actual}. The pinned bytes changed; "
            "FAILING CLOSED (do not sample, do not lock)."
        )
    return path


def _row_text(ds: PublicDataset, row: dict) -> str:
    """The REAL text the engine classifies (BoolQ: question + passage)."""
    if ds.name == "boolq":
        return f"{row['question']}\n{row['passage']}"
    return row["text"]


def _balanced_positions(
    ds: PublicDataset,
    split: str,
    rows: list[dict],
    per_class: int = BALANCED_ROWS_PER_CLASS,
    log=print,
) -> list[int]:
    """Class-balanced diagnostic positions: per_class rows per class, chosen
    by lowest deterministic slot of the RECORDED source_row_id (the same id
    the case stores — N2: the split rides along, no placeholder)."""
    by_class: dict[object, list[int]] = {}
    for pos, row in enumerate(rows):
        by_class.setdefault(ds.label_of(row), []).append(pos)
    picked: list[int] = []
    for label in sorted(by_class, key=lambda x: str(x)):
        bucket = sorted(by_class[label], key=lambda pos: _slot(ds.row_id(rows[pos], split, pos)))
        take = bucket[:per_class]
        if len(take) < per_class:
            log(
                f"{ds.name}: class {label!r} has only {len(bucket)} rows "
                f"(< per_class={per_class}) — diagnostic sample under-filled"
            )
        picked.extend(take)
    return picked


def _natural_positions(
    ds: PublicDataset,
    split: str,
    rows: list[dict],
    exclude: set[int],
    total: int = NATURAL_ROWS,
    log=print,
) -> list[int]:
    """F8: natural view = the dataset's own prevalence, drawn from rows the
    balanced view did NOT take (disjoint), by lowest deterministic slot of
    the recorded source_row_id."""
    remaining = [pos for pos in range(len(rows)) if pos not in exclude]
    ranked = sorted(remaining, key=lambda pos: _slot(ds.row_id(rows[pos], split, pos)))
    picked = ranked[:total]
    if len(picked) < total:
        log(
            f"{ds.name}: natural view requested {total} rows, "
            f"{len(picked)} available after excluding the balanced view"
        )
    return picked


def fetch_dataset(
    ds: PublicDataset,
    split: str,
    per_class: int | None = None,
    natural_total: int | None = None,
) -> tuple[dict[str, list[dict]], dict]:
    """Both views for one dataset; return ({view: records}, file sha256 map).

    The split file is downloaded and parsed ONCE; the balanced view is
    sampled first, the natural view draws DISJOINT remaining rows (F8).
    """
    per_class = BALANCED_ROWS_PER_CLASS if per_class is None else per_class
    natural_total = NATURAL_ROWS if natural_total is None else natural_total
    path = _download(ds, split)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    rows = list(ds.row_reader(path))
    indexes = list(range(len(rows)))
    texts = [_row_text(ds, row) for row in rows]

    balanced_pos = _balanced_positions(ds, split, rows, per_class=per_class)
    natural_pos = _natural_positions(
        ds, split, rows, exclude=set(balanced_pos), total=natural_total
    )

    views: dict[str, list[dict]] = {"balanced": [], "natural": []}
    for view, positions in (("balanced", balanced_pos), ("natural", natural_pos)):
        for pos in positions:
            views[view].append(build_case(ds, rows[pos], split, view, texts[pos], indexes[pos]))
    return views, {ds.files[split]: digest}


def write_view(
    records: list[dict],
    out_path: Path,
    lock_path: Path | None = None,
    files_sha256: dict[str, str] | None = None,
) -> Path:
    """Write one view's cases JSONL + lock; return the lock path.

    ``files_sha256`` (repo-relative file path -> VERIFIED sha256 of the
    downloaded file the rows were drawn from) is REQUIRED — the lock must
    record the exact bytes; a missing/empty map fails closed.
    """
    if not files_sha256:
        raise OSError(
            "public fetch: refusing to write a lock without file sha256s "
            "(the lock must record the downloaded bytes)"
        )
    if not records:
        raise OSError("public fetch: refusing to write an empty view")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    ds = DATASETS[records[0]["source"]]
    lock = {
        "sources": [
            {
                "repo_id": ds.repo_id,
                "revision": ds.revision,
                "license": ds.license,
                "redistribution_allowed": False,
                "files": files_sha256,
            }
        ],
        "parser_version": PARSER_VERSION,
        "counts": {
            "records": len(records),
            "view": records[0]["view"],
            "classes": {
                str(label): sum(1 for r in records if r["labels"][next(iter(r["schema"]))] == label)
                for label in ds.classes
            },
        },
        "cases_sha256": hashlib.sha256(out_path.read_bytes()).hexdigest(),
    }
    lock_path = Path(lock_path) if lock_path else out_path.parent / "dataset.lock.json"
    lock_path.write_text(json.dumps(lock, indent=1) + "\n", encoding="utf-8")
    return lock_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m benchmarks.public.fetch",
        description="Fetch AG News / BoolQ / SST-5 as jevmlx eval JSONL (two views).",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help="output directory (<name>.<view>.jsonl + <name>.<view>.dataset.lock.json)",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        choices=sorted(DATASETS),
        help="fetch only this dataset (repeatable; default: all)",
    )
    parser.add_argument(
        "--split",
        default=None,
        help="override the eval split (default: the dataset's canonical "
        "split — ag_news/sst5 test, boolq validation)",
    )
    parser.add_argument(
        "--per-class",
        type=int,
        default=BALANCED_ROWS_PER_CLASS,
        help="balanced-view rows per class (default: %(default)s)",
    )
    parser.add_argument(
        "--natural-rows",
        type=int,
        default=NATURAL_ROWS,
        help="natural-view row total, drawn disjoint from the balanced view (default: %(default)s)",
    )
    args = parser.parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in args.dataset or sorted(DATASETS):
        ds = DATASETS[name]
        split = args.split or EVAL_SPLITS[name]
        views, sha_by_file = fetch_dataset(
            ds, split, per_class=args.per_class, natural_total=args.natural_rows
        )
        for view, records in views.items():
            out_path = out_dir / f"{name}.{view}.jsonl"
            lock = out_dir / f"{name}.{view}.dataset.lock.json"
            write_view(records, out_path, lock, files_sha256=sha_by_file)
            print(f"wrote {out_path} ({len(records)} cases)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
