"""Fetch ``LocalLLaMA/typed-decisions`` (Hugging Face) as an jevmlx eval JSONL.

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

REPO_ID = "LocalLLaMA/typed-decisions"

WORKFLOWS = (
    "security_incidents",
    "agent_trace_observability",
    "invoice_processing",
    "customer_service",
)
SPLITS = ("train", "test")

# Bump when the conversion semantics change; recorded in dataset.lock.json.
PARSER_VERSION = "1"

# score questions are answered on a fixed 0-3 scale (criteria is a 4-item list).
SCORE_CHOICES = ("0", "1", "2", "3")

# Consensus distributions with top1 - top2 below this are marked ambiguous
# (same threshold as the typesafe fetcher).
AMBIGUOUS_MARGIN = 0.1


def _parquet_path(workflow: str, split: str) -> str:
    return f"{workflow}/{split}-00000-of-00001.parquet"


def _as_obj(value):
    """Parquet stores the nested columns as JSON strings; accept dicts too."""
    if isinstance(value, str):
        return json.loads(value)
    return value


def field_schema(question: dict) -> dict | None:
    """Map one question to an jevmlx schema field, or None to skip.

    ``noul`` -> boolean; ``choice`` -> enum over the criteria keys (published
    order); ``score`` -> enum over "0".."3" with the level texts folded into
    the description. Unknown types are skipped.
    """
    instructions = question["instructions"]
    criteria = question.get("criteria")
    qtype = question["type"]
    if qtype == "noul":
        return {"type": "boolean", "description": instructions}
    if qtype == "choice":
        return {"type": "enum", "description": instructions, "choices": list(criteria)}
    if qtype == "score":
        levels = "; ".join(f"{i} = {text}" for i, text in enumerate(criteria))
        return {
            "type": "enum",
            "description": f"{instructions} Scale: {levels}.",
            "choices": list(SCORE_CHOICES),
        }
    return None


def _label(gold: dict, qtype: str):
    """Gold label in jevmlx label form (booleans become real bools)."""
    label = gold.get("label")
    if label is None:
        return None
    if qtype == "noul":
        if isinstance(label, bool):
            return label
        return str(label).lower() == "true"
    return str(label)


def _margin(distribution: dict[str, float]) -> float:
    probs = sorted((float(p) for p in distribution.values()), reverse=True)
    if not probs:
        return 0.0
    return probs[0] - (probs[1] if len(probs) > 1 else 0.0)


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
        label = _label(answer, question["type"])
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


def _download(workflow: str, split: str, revision: str | None) -> tuple[Path, str]:
    """Download one parquet shard; return (local path, resolved revision sha)."""
    from huggingface_hub import HfApi, hf_hub_download

    if revision is None:
        revision = HfApi().dataset_info(REPO_ID).sha
    path = hf_hub_download(
        REPO_ID,
        _parquet_path(workflow, split),
        repo_type="dataset",
        revision=revision,
    )
    return Path(path), revision


def iter_rows(path: Path) -> Iterator[dict]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise SystemExit(
            "pyarrow is required to read typed-decisions parquet: pip install 'jevmlx[bench]'"
        ) from exc
    yield from pq.read_table(path).to_pylist()


def fetch_all(
    workflows: list[str], splits: list[str], revision: str | None = None
) -> tuple[list[dict], dict, dict]:
    """Fetch every (workflow, split); return (records, summary, source metadata)."""
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
        "--revision", default=None, help="dataset commit sha (default: current main)"
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
