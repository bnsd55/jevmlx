#!/usr/bin/env python3
"""Convert the repo's labeled cases (benchmarks/cases.json) to eval JSONL.

The 24 in-repo quality-eval cases become the same JSONL contract as the
TypeSafe fetcher's output (with their own dataset.lock.json). Unlike the
TypeSafe records they are *not* benchmark_only: they are curated in-repo
labels and may be used for calibration.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
CASES = os.path.join(HERE, "cases.json")

# Bump when the conversion semantics change; recorded in dataset.lock.json.
PARSER_VERSION = "1"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m benchmarks.to_jsonl",
        description="Convert benchmarks/cases.json to eval JSONL + dataset.lock.json.",
    )
    parser.add_argument("--out", required=True, help="output JSONL path")
    parser.add_argument(
        "--lock",
        default=None,
        help=(
            "lock path (default: <out dir>/dataset.lock.json). Pass a "
            "per-dataset name when several datasets share one cache dir — "
            "the bench registers <name>.dataset.lock.json next to the jsonl."
        ),
    )
    args = parser.parse_args(argv)

    with open(CASES, encoding="utf-8") as f:
        data = json.load(f)

    records: list[dict] = []
    for family in data["families"].values():
        for case in family["cases"]:
            record_id = f"quality-eval/{family_name(data, family)}/{case['id']}"
            records.append(
                {
                    "id": record_id,
                    "group_id": record_id,
                    "source": "quality-eval",
                    "workflow": None,
                    "benchmark_only": False,
                    "schema": family["schema"],
                    "context": case["context"],
                    "labels": case["labels"],
                    "split": "train",
                    "meta": {},
                }
            )

    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    lock = {
        "sources": [],
        "parser_version": PARSER_VERSION,
        "counts": {
            "records": len(records),
            "families": len(data["families"]),
        },
        "cases_sha256": hashlib.sha256(open(out_path, "rb").read()).hexdigest(),
        "fetched_at": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
    }
    lock_path = args.lock or os.path.join(os.path.dirname(out_path) or ".", "dataset.lock.json")
    with open(lock_path, "w", encoding="utf-8") as f:
        json.dump(lock, f, indent=1)
        f.write("\n")

    print(f"records: {len(records)}")
    print(f"wrote {out_path}")
    print(f"wrote {lock_path}")
    return 0


def family_name(data: dict, family: dict) -> str:
    """The family's key in the source file (cases carry only a display name)."""
    for key, value in data["families"].items():
        if value is family:
            return key
    raise LookupError("family not found in source data")


if __name__ == "__main__":
    raise SystemExit(main())
