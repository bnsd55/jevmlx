"""Fetch the two OpenJev datasets as jevmlx eval JSONL.

- ``authored144`` (144 rows, 36 groups × 4 variants): 3-way evidence
  interpretation (``supported`` / ``insufficient`` / ``contradicted``), the
  label is the INDEX into the row's ``options`` list.
- ``perturbations108`` (108 rows, split ``rebase_stability``): 36 base rows
  × 3 perturbation variants (``option_reversal``, ``criterion_wrapper``,
  ``irrelevant_context``); each row's ``provenance.base_id`` points at an
  ``authored144`` row, so perturbation cases cluster with their base under
  ``group_id``.

Source: github.com/TheoLeeCJ/openjev (the files live at
``benchmarks/data/authored144.jsonl`` and
``benchmarks/data/perturbations108.jsonl``). Pinned by commit sha; every
file's sha256 is verified on download and re-checked against the lock on
cache reuse — a mismatch FAILS CLOSED (OSError), never sampled or locked.

Both datasets are **model-reviewed, not human-adjudicated**
(``provenance.annotation_status`` is copied verbatim into ``meta``); the
leaderboard row must say so.

No sampling views: every row is an eval row (the datasets are small and
fixed). The converter produces ONE cases file per dataset:

- one enum field (``decision``) with the row's three option descriptions;
- ``context`` = the row's ``state``;
- ``labels`` = the option id at the label index (a STRING, so option-order
  perturbations keep the label valid);
- ``group_id`` = ``provenance.base_id`` for perturbation rows (clusters
  with the authored144 base), the row's own ``group_id`` otherwise;
- ``meta`` carries family/variant/partition/perturbation kind +
  ``annotation_status`` verbatim from provenance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

PARSER_VERSION = "1"

# The pinned upstream commit (github.com/TheoLeeCJ/openjev). The files live
# at benchmarks/data/ in the repo — fetched raw via the GitHub API.
REPO = "TheoLeeCJ/openjev"
REVISION = "ca3ba65f142967030ecb453346e94d6f476a69df"
LICENSE = (
    "Project authored; no copied external text (per provenance.rights). "
    "Model-reviewed, not human-adjudicated (per provenance.annotation_status)."
)

# The two dataset files + their KNOWN-GOOD sha256 at the pinned revision
# (verified on download and re-checked on cache reuse — F5 fail-closed).
FILES: dict[str, str] = {
    "authored144": "benchmarks/data/authored144.jsonl",
    "perturbations108": "benchmarks/data/perturbations108.jsonl",
}
EXPECTED_SHA256: dict[str, str] = {
    "authored144": "8162d1c73f925af64453f1ec05ef36d583b3815bf698e60f0d454bd11537e079",
    "perturbations108": "1dd7ccf80518d0e34886478ca23982aa726e9daccd343b9e95cedaf6b569bec4",
}


def _download(dataset_name: str) -> Path:
    """Download the pinned file from GitHub and VERIFY it (F5).

    The raw file is fetched at the pinned revision; the digest must equal
    the hardcoded expectation — any mismatch is a hard OSError (never
    converted, never locked).
    """
    import urllib.request

    file_path = FILES[dataset_name]
    url = f"https://raw.githubusercontent.com/{REPO}/{REVISION}/{file_path}"
    try:
        with urllib.request.urlopen(url, timeout=60) as response:  # noqa: S310 — pinned public raw URL
            data = response.read()
    except Exception as exc:  # noqa: BLE001 — one clear message for any cause
        raise OSError(
            f"{REPO}@{REVISION} [{file_path}]: download failed "
            f"({exc.__class__.__name__}). The fetch needs network access to "
            f"GitHub; locally run: python -m benchmarks.openjev.fetch --out <path> "
            f"--lock <path>"
        ) from exc
    actual = hashlib.sha256(data).hexdigest()
    expected = EXPECTED_SHA256[dataset_name]
    if actual != expected:
        raise OSError(
            f"{REPO}@{REVISION} [{file_path}]: sha256 mismatch — "
            f"expected {expected}, got {actual}. The pinned bytes changed; "
            "FAILING CLOSED (do not convert, do not lock)."
        )
    # Write to a temp path the caller can read; return it.
    import tempfile

    path = Path(tempfile.gettempdir()) / f"openjev_{dataset_name}_{REVISION[:8]}.jsonl"
    path.write_bytes(data)
    return path


def _read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _option_descriptions(options: list[dict]) -> list[str]:
    """The three option descriptions, in the row's declared order."""
    return [opt["description"] for opt in options]


def _label_value(row: dict) -> str:
    """The gold label as a STRING — the option id at the label index.

    Labels are strings (not indices) so option-order perturbations
    (option_reversal reverses the choices) keep the label valid: the
    engine's enum value is the option description, and the label is the
    same description regardless of position.
    """
    options = row["options"]
    label_index = int(row["label"])
    return options[label_index]["description"]


def _build_schema(row: dict) -> dict:
    """One enum field (``decision``) with the row's three option
    descriptions."""
    return {
        "decision": {
            "type": "enum",
            "description": "Assess the claim using only the supplied evidence.",
            "choices": _option_descriptions(row["options"]),
        }
    }


def _group_id(row: dict) -> str:
    """Perturbation rows cluster with their authored144 base (base_id);
    authored144 rows use their own group_id."""
    provenance = row.get("provenance") or {}
    base_id = provenance.get("base_id")
    if base_id:
        return base_id
    return row.get("group_id") or row["id"]


def _meta(row: dict) -> dict:
    """family / variant / partition / perturbation kind + annotation_status
    copied VERBATIM from provenance (model-reviewed, not human-adjudicated).
    """
    provenance = row.get("provenance") or {}
    meta = {
        "family": row.get("family"),
        "variant": provenance.get("variant"),
        "partition": provenance.get("partition"),
        "source": "openjev",
        "source_repo": REPO,
        "source_revision": REVISION,
        "source_file": FILES[_dataset_name(row)],
        "source_row_id": row["id"],
        "annotation_status": provenance.get("annotation_status"),
    }
    # Perturbation rows carry their kind as the variant name.
    if provenance.get("base_id"):
        meta["perturbation"] = provenance.get("variant")
    return {k: v for k, v in meta.items() if v is not None}


def _dataset_name(row: dict) -> str:
    """The dataset name from the split: rebase_stability => perturbations108."""
    split = row.get("split", "")
    if split == "rebase_stability":
        return "perturbations108"
    return "authored144"


def build_case(row: dict) -> dict:
    """One eval record from an OpenJev row."""
    dataset_name = _dataset_name(row)
    schema = _build_schema(row)
    context = row["state"]
    return {
        "id": f"openjev/{dataset_name}/{row['id']}",
        "group_id": f"openjev/{_group_id(row)}",
        "source": "openjev",
        "dataset": dataset_name,
        "benchmark_only": True,
        "schema": schema,
        "context": context,
        "labels": {"decision": _label_value(row)},
        "split": row.get("split", "test"),
        "meta": _meta(row),
    }


def convert_dataset(dataset_name: str, out_path: Path, lock_path: Path) -> tuple[int, str]:
    """Fetch, verify, convert one dataset; write the cases JSONL + lock.

    Returns (n_cases, verified_sha256).
    """
    path = _download(dataset_name)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    rows = list(_read_jsonl(path))
    records = [build_case(row) for row in rows]
    if not records:
        raise OSError(f"openjev {dataset_name}: converted 0 cases — refusing to write")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    lock = {
        "dataset": dataset_name,
        "source": "openjev",
        "source_repo": REPO,
        "source_revision": REVISION,
        "source_file": FILES[dataset_name],
        "source_sha256": digest,
        "expected_sha256": EXPECTED_SHA256[dataset_name],
        "n_cases": len(records),
        "parser_version": PARSER_VERSION,
        "license": LICENSE,
    }
    lock_path = Path(lock_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(json.dumps(lock, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return len(records), digest


def main(argv: list[str] | None = None) -> int:
    """CLI: fetch + convert both (or one) dataset(s) to eval JSONL.

    Used directly for one-off fetches; the bench calls convert_dataset.
    """
    parser = argparse.ArgumentParser(description="Fetch OpenJev datasets as jevmlx eval JSONL")
    parser.add_argument(
        "--out-dir", default=".", help="output directory for the JSONL + lock files"
    )
    parser.add_argument(
        "--dataset",
        choices=["authored144", "perturbations108", "both"],
        default="both",
        help="which dataset to fetch (default: both)",
    )
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    names = ["authored144", "perturbations108"] if args.dataset == "both" else [args.dataset]
    for name in names:
        out_path = out_dir / f"{name}.jsonl"
        lock_path = out_dir / f"{name}.dataset.lock.json"
        n, sha = convert_dataset(name, out_path, lock_path)
        print(f"{name}: {n} cases, sha256 {sha[:16]}...")
    return 0
