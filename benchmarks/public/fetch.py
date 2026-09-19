"""Fetch the three PUBLIC gold datasets (W6-B5) as jevmlx eval JSONL.

- AG News (``fancyzhx/ag_news``): 4-class topic classification — enum.
- BoolQ (``google/boolq``): reading-comprehension yes/no — boolean (noul).
- SST-5 (``SetFit/sst5``): ordinal sentiment, 5 levels — ordered enum
  (the B1 derivation: the ORDER lives in the choices list and the level
  texts in the description; no new engine field type).

Every dataset is PINNED by its repo commit sha (``DEFAULT_REVISION`` per
dataset, like typed_decisions/fetch.py), every downloaded file's sha256
lands in the lock, and a mismatch FAILS CLOSED (SystemExit) instead of
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
status is therefore NOT cleared for any of them: the cases files store
row ids + splits + option order + selected-input hashes ONLY — never the
source text (the engine renders the text at eval time from a locally
cached copy the fetch produced; text never lands in a result artifact).
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
    """One public dataset's fetch + sampling config (name + repo + pin)."""

    def __init__(
        self,
        name: str,
        repo_id: str,
        revision: str,
        license_name: str,
        files: dict[str, str],
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
        self.row_reader = row_reader  # local path -> iterable of rows
        self.schema_builder = schema_builder  # () -> schema dict (shared mapping)
        self.label_of = label_of  # row -> gold label (label_form output)
        self.row_id = row_id  # row -> stable source row id
        self.classes = classes  # ordered class list


# ---- dataset definitions -------------------------------------------------


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

DATASETS: dict[str, PublicDataset] = {
    "ag_news": PublicDataset(
        name="ag_news",
        repo_id="fancyzhx/ag_news",
        revision="eb185aade064a813bc0b7f42de02595523103ca4",
        license_name="unknown (dataset card: no license listed; "
        "redistribution NOT cleared — store row ids only)",
        files={
            "train": "data/train-00000-of-00001.parquet",
            "test": "data/test-00000-of-00001.parquet",
        },
        row_reader=_read_parquet,
        schema_builder=_ag_news_schema,
        label_of=lambda row: AG_NEWS_LABELS[int(row["label"])],
        row_id=lambda row: f"test/{hashlib.sha256(row['text'].encode('utf-8')).hexdigest()[:16]}",
        classes=AG_NEWS_CLASSES,
    ),
    "boolq": PublicDataset(
        name="boolq",
        repo_id="google/boolq",
        revision="35b264d03638db9f4ce671b711558bf7ff0f80d5",
        license_name="CC-BY-SA 3.0 (share-alike; redistribution NOT "
        "cleared for result artifacts — store row ids only)",
        files={
            "train": "data/train-00000-of-00001.parquet",
            "validation": "data/validation-00000-of-00001.parquet",
        },
        row_reader=_read_parquet,
        schema_builder=_boolq_schema,
        label_of=lambda row: bool(row["answer"]),
        row_id=lambda row: (
            "validation/"
            + hashlib.sha256((row["question"] + "\0" + row["passage"]).encode("utf-8")).hexdigest()[
                :16
            ]
        ),
        classes=BOOLQ_CLASSES,
    ),
    "sst5": PublicDataset(
        name="sst5",
        repo_id="SetFit/sst5",
        revision="e51bdcd8cd3a30da231967c1a249ba59361279a3",
        license_name="unspecified (dataset card lists no license; "
        "redistribution NOT cleared — store row ids only)",
        files={"dev": "dev.jsonl", "test": "test.jsonl"},
        row_reader=_read_jsonl,
        schema_builder=_sst5_schema,
        label_of=lambda row: str(int(row["label"])),
        row_id=lambda row: f"test/{hashlib.sha256(row['text'].encode('utf-8')).hexdigest()[:16]}",
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


def _input_hash(context: str) -> str:
    """The selected-input hash recorded per case (schema + context bytes)."""
    return hashlib.sha256(context.encode("utf-8")).hexdigest()


def build_case(ds: PublicDataset, row: dict, split: str, view: str, context: str) -> dict:
    """One eval record for either view.

    Stores NO source text: ``context`` here is the id-embedded stub the
    engine's data loader expands from the local fetch cache (the cases
    file's job is to name the row, not to carry it).
    """
    schema = ds.schema_builder()
    field = next(iter(schema.values()))
    # The option order the scorer judges: enum fields carry choices; a
    # boolean field's ordered options are its two labels (false, true —
    # the typed-decisions noul convention).
    option_order = field.get("choices", BOOLQ_CLASSES)
    record_id = f"{ds.name}/{view}/{ds.row_id(row)}"
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
            "source_row_id": ds.row_id(row),
            "source_repo": ds.repo_id,
            "source_revision": ds.revision,
            "option_order": option_order,
            "selected_input_hash": _input_hash(context),
        },
    }


def sample_balanced(
    rows: list[dict], ds: PublicDataset, per_class: int = BALANCED_ROWS_PER_CLASS
) -> list[dict]:
    """Class-balanced diagnostic view: per_class rows per class, chosen by
    lowest deterministic slot."""
    by_class: dict[object, list[dict]] = {}
    for row in rows:
        by_class.setdefault(ds.label_of(row), []).append(row)
    picked: list[dict] = []
    for label in sorted(by_class, key=lambda x: str(x)):
        bucket = sorted(by_class[label], key=lambda r: _slot(ds.row_id(r)))
        picked.extend(bucket[:per_class])
    return picked


def sample_natural(rows: list[dict], ds: PublicDataset, total: int = NATURAL_ROWS) -> list[dict]:
    """Natural-distribution view: the dataset's own class prevalence,
    total rows chosen by lowest deterministic slot."""
    ranked = sorted(rows, key=lambda r: _slot(ds.row_id(r)))
    return ranked[:total]


def _context_stub(
    ds: PublicDataset, split: str, row_id: str, file_path: str, row_index: int
) -> str:
    """The id-embedded context: the eval runner expands this from the local
    HF cache (source text never enters the artifact).

    Names the EXACT source location — repo, pinned revision, file, row
    index — so the pinned row can be re-read offline at eval time; the row
    id hash ties it to the deterministic sample.
    """
    return json.dumps(
        {
            "fetch": "public",
            "dataset": ds.name,
            "repo_id": ds.repo_id,
            "revision": ds.revision,
            "file": file_path,
            "row_index": row_index,
            "row_id": row_id,
        }
    )


# ---- fetch + write --------------------------------------------------------


def _download(ds: PublicDataset, split: str, cache_dir: Path | None = None) -> Path:
    from huggingface_hub import hf_hub_download

    try:
        return Path(
            hf_hub_download(
                ds.repo_id,
                ds.files[split],
                repo_type="dataset",
                revision=ds.revision,
            )
        )
    except Exception as exc:  # noqa: BLE001 — one clear message for any cause
        raise SystemExit(
            f"{ds.repo_id}@{ds.revision} [{ds.files[split]}]: download failed "
            f"({exc.__class__.__name__}). The fetch needs the PINNED revision "
            "in the local HF cache; locally run: huggingface-cli download "
            f"{ds.repo_id} --repo-type dataset --revision {ds.revision}"
        ) from exc


def fetch_view(
    ds: PublicDataset,
    split: str,
    view: str,
    *,
    file_sha256: dict | None = None,
) -> tuple[list[dict], dict]:
    """Fetch + sample one view; return (records, per-file sha256 map)."""
    path = _download(ds, split)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if file_sha256 is not None:
        file_sha256[ds.files[split]] = digest
    rows = list(ds.row_reader(path))
    if view == "balanced":
        picked = sample_balanced(rows, ds)
    else:
        picked = sample_natural(rows, ds)
    index_of = {id(row): i for i, row in enumerate(rows)}
    records = []
    for row in picked:
        row_id = ds.row_id(row)
        context = _context_stub(ds, split, row_id, ds.files[split], index_of[id(row)])
        records.append(build_case(ds, row, split, view, context))
    return records, {ds.files[split]: digest}


def fetch_dataset(ds: PublicDataset, split: str) -> tuple[dict[str, list[dict]], dict]:
    """Both views for one dataset; return ({view: records}, file sha256 map)."""
    views: dict[str, list[dict]] = {}
    sha_by_file: dict[str, str] = {}
    for view in ("balanced", "natural"):
        records, sha = fetch_view(ds, split, view, file_sha256=sha_by_file)
        views[view] = records
    return views, sha_by_file


def write_view(
    records: list[dict],
    out_path: Path,
    lock_path: Path | None = None,
    files_sha256: dict[str, str] | None = None,
) -> Path:
    """Write one view's cases JSONL + lock; return the lock path.

    ``files_sha256`` (repo-relative file path -> sha256 of the downloaded
    file the rows were drawn from) is REQUIRED — the lock must record the
    exact bytes; a missing/empty map fails closed.
    """
    if not files_sha256:
        raise SystemExit(
            "public fetch: refusing to write a lock without file sha256s "
            "(the lock must record the downloaded bytes)"
        )
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
    args = parser.parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in args.dataset or sorted(DATASETS):
        ds = DATASETS[name]
        split = args.split or EVAL_SPLITS[name]
        views, sha_by_file = fetch_dataset(ds, split)
        for view, records in views.items():
            out_path = out_dir / f"{name}.{view}.jsonl"
            lock = out_dir / f"{name}.{view}.dataset.lock.json"
            write_view(records, out_path, lock, files_sha256=sha_by_file)
            print(f"wrote {out_path} ({len(records)} cases)")
        print(f"{name}: files sha256 {json.dumps(sha_by_file, indent=1)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
