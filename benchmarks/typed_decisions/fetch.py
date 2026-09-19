"""Fetch ``LocalLLaMA/typed-decisions`` (Hugging Face) as a jevmlx eval JSONL.

The dataset is the TypeSafe typed-decisions task published as versioned
parquet: the same four workflows and the same ``noul``/``choice``/``score``
question types as ``benchmarks.typesafe.fetch``, with per-field consensus
``probabilities`` and an official ``train``/``test`` split. Unlike the
``typesafe`` scrape it is revision-pinned, so a results folder can be
rebuilt byte-for-byte from its ``dataset.lock.json``.

Each dataset row becomes one eval record: ``state`` (structured JSON) is the
context, ``questions`` the schema, ``gold`` the labels. The gold labels are
model-consensus pseudo-labels, not independent human gold, so every record
is ``benchmark_only`` (never used for calibration or routing thresholds).
The record shape matches the typesafe fetcher (``source``, ``workflow``,
``meta.consensus``/``margin``/``ambiguous``) so the agreement and
TVD-vs-consensus metrics apply unchanged.

Parquet files are downloaded with ``huggingface_hub`` (cached under the HF
hub cache, offline after the first run) and read with ``pyarrow``
(``pip install jevmlx[bench]``).

Usage:
    python -m benchmarks.typed_decisions.fetch --out cases.jsonl \
        [--split test] [--workflow NAME] [--revision SHA] [--lock PATH]
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Iterator
from pathlib import Path

import pyarrow.parquet as pq

from benchmarks.typesafe.questions import (
    AMBIGUOUS_MARGIN,
    WORKFLOWS,
    field_schema,
)
from benchmarks.typesafe.questions import (
    as_obj as _as_obj,
)
from benchmarks.typesafe.questions import (
    label_form as _label,
)
from benchmarks.typesafe.questions import (
    margin_of as _margin,
)

REPO_ID = "LocalLLaMA/typed-decisions"

SPLITS = ("train", "test")

# Bump when the conversion semantics change; recorded in dataset.lock.json.
PARSER_VERSION = "1"

# F6: results must be reproducible. A floating "current main" revision
# silently changes the dataset between runs; DEFAULT_REVISION pins the
# published snapshot this fetcher was written against (resolved once via
# dataset_info(REPO_ID).sha). --revision overrides it; every value —
# including a branch name — is resolved to the commit sha via
# dataset_info(REPO_ID, revision=...).sha (F9), so the lock always records
# an immutable sha.
DEFAULT_REVISION = "0af3f0e9dc6d28c2f8f1c9d1ba2e4a55f0e6c9d3"


def _parquet_path(workflow: str, split: str) -> str:
    return f"{workflow}/{split}-00000-of-00001.parquet"


def row_to_record(row: dict) -> dict:
    """One dataset row -> one jevmlx eval record (typesafe fetcher shape)."""
    workflow = row["workflow"]
    questions = _as_obj(row["questions"])
    gold = _as_obj(row["gold"])
    state = _as_obj(row["state"])

    schema: dict = {}
    labels: dict = {}
    consensus: dict = {}
    margins: dict = {}
    ambiguous: list[str] = []
    skipped = 0

    for name, question in questions.items():
        field = field_schema(question)
        answer = gold.get(name)
        if field is None or not isinstance(answer, dict):
            skipped += 1
            continue
        label = _label(answer.get("label"), question["type"])
        if label is None:
            skipped += 1
            continue
        schema[name] = field
        labels[name] = label
        distribution = {str(k): float(v) for k, v in (answer.get("probabilities") or {}).items()}
        if distribution:
            consensus[name] = distribution
            margins[name] = _margin(distribution)
            if margins[name] < AMBIGUOUS_MARGIN:
                ambiguous.append(name)

    record_id = f"typed-decisions/{workflow}/{row['id']}"
    return {
        "id": record_id,
        "group_id": record_id,
        "source": "typed-decisions",
        "workflow": workflow,
        "benchmark_only": True,
        "schema": schema,
        "context": json.dumps(state, indent=1, ensure_ascii=False),
        "labels": labels,
        "split": row["split"],
        "meta": {
            "consensus": consensus,
            "margin": margins,
            "ambiguous": ambiguous,
            "label_agreement": _as_obj(row.get("label_agreement")) or {},
            "factors": _as_obj(row.get("factors")) or {},
        },
        "skipped_questions": skipped,
    }


def _resolve_revision(revision: str) -> str:
    """Resolve any revision (sha, branch, tag) to its immutable commit sha."""
    from huggingface_hub import HfApi

    return HfApi().dataset_info(REPO_ID, revision=revision).sha


def _download(workflow: str, split: str, revision: str) -> tuple[Path, str]:
    """Download one parquet shard; return (local path, resolved revision sha).

    ``revision`` arrives pre-resolved by :func:`_resolve_revision` (or as a
    sha already); it is passed straight to hf_hub_download.
    """
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        REPO_ID,
        _parquet_path(workflow, split),
        repo_type="dataset",
        revision=revision,
    )
    return Path(path), revision


def iter_rows(path: Path) -> Iterator[dict]:
    yield from pq.read_table(path).to_pylist()


def fetch_all(
    workflows: list[str],
    splits: list[str],
    revision: str | None = None,
    *,
    resolve_revision=None,
) -> tuple[list[dict], dict, dict]:
    """Fetch every (workflow, split); return (records, summary, source metadata).

    ``revision`` defaults to :data:`DEFAULT_REVISION` (the pinned snapshot)
    and is resolved — branch name or sha alike — to the dataset's commit
    sha before downloading, so every shard in one fetch shares one
    revision and the lock records an immutable sha.
    ``resolve_revision`` is injectable for tests.
    """
    # Look the resolver up as a module global at call time (not via a
    # default-arg binding) so tests can monkeypatch it.
    if resolve_revision is None:
        resolve_revision = _resolve_revision
    if revision is None:
        revision = DEFAULT_REVISION
    revision = resolve_revision(revision)
    records: list[dict] = []
    field_types = {"boolean": 0, "enum": 0, "multi": 0}
    skipped_questions = 0
    files: list[dict] = []
    for workflow in workflows:
        for split in splits:
            path, revision = _download(workflow, split, revision)
            files.append(
                {
                    "file": _parquet_path(workflow, split),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
            for row in iter_rows(path):
                record = row_to_record(row)
                skipped_questions += record.pop("skipped_questions")
                records.append(record)
                for field in record["schema"].values():
                    field_types[field["type"]] += 1
    summary = {
        "workflows": len(workflows),
        "splits": list(splits),
        "records": len(records),
        "fields": field_types,
        "skipped_questions": skipped_questions,
    }
    source = {"repo_id": REPO_ID, "revision": revision, "files": files}
    return records, summary, source


def write_outputs(
    records: list[dict], out_path: Path, source: dict, counts: dict, lock_path: Path | None = None
) -> Path:
    """Write cases JSONL and its lock; return the lock path."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    lock = {
        "sources": [source],
        "parser_version": PARSER_VERSION,
        "counts": counts,
        "cases_sha256": hashlib.sha256(out_path.read_bytes()).hexdigest(),
    }
    lock_path = Path(lock_path) if lock_path else out_path.parent / "dataset.lock.json"
    lock_path.write_text(json.dumps(lock, indent=1) + "\n", encoding="utf-8")
    return lock_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m benchmarks.typed_decisions.fetch",
        description=f"Fetch {REPO_ID} as jevmlx eval JSONL.",
    )
    parser.add_argument("--out", required=True, help="output JSONL path")
    parser.add_argument(
        "--lock", default=None, help="lock path (default: dataset.lock.json next to --out)"
    )
    parser.add_argument(
        "--workflow",
        action="append",
        choices=WORKFLOWS,
        help="fetch only this workflow (repeatable; default: all)",
    )
    parser.add_argument(
        "--split",
        action="append",
        choices=SPLITS,
        help="fetch only this split (repeatable; default: test)",
    )
    parser.add_argument(
        "--revision",
        default=DEFAULT_REVISION,
        help="dataset revision (sha, branch or tag; resolved to a commit sha; "
        "default: the pinned DEFAULT_REVISION)",
    )
    args = parser.parse_args(argv)

    workflows = list(args.workflow or WORKFLOWS)
    splits = list(args.split or ["test"])
    records, summary, source = fetch_all(workflows, splits, args.revision)
    lock_path = write_outputs(records, Path(args.out), source, summary, args.lock)

    print(f"revision: {source['revision']}")
    print(f"workflows: {summary['workflows']}  splits: {','.join(summary['splits'])}")
    print(f"records: {summary['records']}")
    print(f"fields: {summary['fields']['boolean']} boolean, {summary['fields']['enum']} enum")
    print(f"skipped questions: {summary['skipped_questions']}")
    print(f"wrote {args.out}")
    print(f"wrote {lock_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
