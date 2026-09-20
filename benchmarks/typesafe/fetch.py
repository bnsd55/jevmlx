"""Fetch TypeSafe's public evaluation examples as a jevmlx eval JSONL.

This produces a *flattened, TypeSafe-derived* benchmark: each published
workflow step (reference node) becomes eval fields sharing that step's exact
read-set of documents. It is not a reproduction of TypeSafe's interactive
workflow, and the published consensus pseudo-labels are not independent gold
labels — every record is therefore ``benchmark_only`` (never used for
calibration or routing thresholds).

Downloads the published eval viewer pages from https://evals.typesafe.ai/
(no JavaScript rendering needed: each workflow ships a ``<workflow>-cases.js``
file embedding a JSON payload in a ``__VIEWER_DATA__(...)`` call), converts
them to jevmlx eval JSONL, and writes ``cases.jsonl`` plus a
``dataset.lock.json`` (source URLs, content hashes, fetch timestamps, parser
version, counts, and the hash of the written cases file). Nothing from
TypeSafe is committed to this repository; raw downloads are cached
content-addressed under ``~/.cache/jevmlx/typesafe/`` so reruns work offline.

Usage:
    python -m benchmarks.typesafe.fetch --out cases.jsonl [--workflow NAME] [--refresh]
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import re
import urllib.request
from collections.abc import Iterator
from pathlib import Path

from benchmarks.typesafe.questions import (
    AMBIGUOUS_MARGIN,
    WORKFLOWS,
)
from benchmarks.typesafe.questions import (
    field_schema as _field_schema,
)
from benchmarks.typesafe.questions import (
    margin_of as _margin_of,
)

BASE_URL = "https://evals.typesafe.ai"
CACHE_DIR = Path.home() / ".cache" / "jevmlx" / "typesafe"

# Bump when the conversion semantics change; recorded in dataset.lock.json.
PARSER_VERSION = "2"

# The site rejects requests with the default Python urllib User-Agent (HTTP 403).
USER_AGENT = "Mozilla/5.0 (compatible; jevmlx-eval-fetcher)"

_VIEWER_DATA_RE = re.compile(r"__VIEWER_DATA__\((\{.*\})\)", re.S)


def split_for(case_id: str) -> str:
    """Deterministic train/holdout split.

    ``sha1(id)`` interpreted as hex: holdout when the first 8 hex digits as an
    integer are divisible by 5 (~20% holdout), else train. Pure function of the
    id, so the split is stable across runs and machines. The harness must treat
    records as groups (``group_id``) when splitting analytically — same-family
    cases may straddle the hash split.
    """
    digest = hashlib.sha1(case_id.encode()).hexdigest()
    return "holdout" if int(digest[:8], 16) % 5 == 0 else "train"


def _viewer_json(raw_js: str) -> dict:
    """Extract the ``__VIEWER_DATA__(<json>)`` payload from a viewer page."""
    match = _VIEWER_DATA_RE.search(raw_js) or re.search(r"__VIEWER_DATA__\((.*)\)", raw_js, re.S)
    if match is None:
        raise ValueError("no __VIEWER_DATA__(...) payload found in page")
    return json.loads(match.group(1))


def _pointer_path(workflow: str, cache_dir: Path) -> Path:
    return cache_dir / f"{workflow}-cases.js.meta"


def _download(url: str) -> tuple[bytes, str | None]:
    """Fetch ``url`` and return (body, ETag header if sent)."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
        return response.read(), response.headers.get("ETag")


def _now_iso() -> str:
    return datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")


def fetch_workflow(
    workflow: str, cache_dir: Path | None = None, *, refresh: bool = False
) -> tuple[dict, dict]:
    """Return (``eval`` payload, source metadata) for one workflow.

    The cache is content-addressed: downloads are stored as
    ``<workflow>-<sha256[:12]>.js`` next to a ``<workflow>-cases.js.meta``
    pointer holding url, sha256, fetch time and ETag. With a valid pointer the
    fetch is offline; ``refresh=True`` re-downloads and stores a new entry.
    Raises if the cached content no longer matches its recorded sha256.
    """
    cache_dir = cache_dir or CACHE_DIR
    pointer_path = _pointer_path(workflow, cache_dir)
    url = f"{BASE_URL}/{workflow}-cases.js"

    if not refresh and pointer_path.exists():
        meta = json.loads(pointer_path.read_text(encoding="utf-8"))
        content_path = cache_dir / f"{workflow}-{meta['sha256'][:12]}.js"
        if content_path.exists():
            raw = content_path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != meta["sha256"]:
                raise ValueError(f"cached {workflow} content does not match its recorded sha256")
            return _viewer_json(raw.decode("utf-8", errors="replace"))["eval"], meta

    cache_dir.mkdir(parents=True, exist_ok=True)
    raw, etag = _download(url)
    sha256 = hashlib.sha256(raw).hexdigest()
    (cache_dir / f"{workflow}-{sha256[:12]}.js").write_bytes(raw)
    meta = {"url": url, "sha256": sha256, "fetched_at": _now_iso(), "etag": etag}
    pointer_path.write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return _viewer_json(raw.decode("utf-8", errors="replace"))["eval"], meta


def _canonical_value(value, qtype: str):
    """Canonicalize a published reference value to its jevmlx label form."""
    if qtype == "noul":
        if isinstance(value, bool):
            return value
        if isinstance(value, int | float):
            return float(value) >= 0.5
        return str(value).lower() == "true"
    if qtype == "score":
        try:
            return str(max(0, min(3, int(round(float(value))))))
        except (TypeError, ValueError):
            return None
    return str(value)


def _consensus(
    entry: dict, where: str, choices: list[str] | None = None
) -> tuple[object, dict[str, float], float, bool]:
    """Collapse one reference answer's reviewer sets.

    The published form is uniform per entry: every set either carries a
    probability distribution or a bare value — mixing the two raises
    ``ValueError`` (it would silently drop votes, and averaging denominators
    would be wrong). Returns ``(label, distribution, margin, ambiguous)``:

    - probability sets: the distributions are averaged over the contributing
      reviewers only, then normalized; label = argmax.
    - bare-value sets: the vote share of each canonical value becomes the
      distribution; label = plurality.
    - ties are marked ambiguous and broken deterministically by choice order
      (for booleans: True before False), never by dict insertion order.
    - margin is top1 - top2 of the distribution; ``ambiguous`` when below
      ``AMBIGUOUS_MARGIN``.
    """
    sets = entry.get("sets") or []
    prob_sets = [subset for subset in sets if subset.get("probabilities")]
    value_sets = [
        subset
        for subset in sets
        if not subset.get("probabilities") and subset.get("value") is not None
    ]
    if prob_sets and value_sets:
        raise ValueError(f"{where}: mixed consensus forms (probability and bare-value sets)")

    qtype = entry["type"]
    distribution: dict[str, float] = {}
    if prob_sets:
        totals: dict[str, float] = {}
        for subset in prob_sets:
            for key, prob in subset["probabilities"].items():
                totals[key] = totals.get(key, 0.0) + float(prob)
        total = sum(totals.values())
        distribution = {key: value / total for key, value in totals.items()} if total else {}
    elif value_sets:
        votes: dict[str, int] = {}
        for subset in value_sets:
            key = _canonical_value(subset["value"], qtype)
            key = _distribution_key(key, qtype)
            if key is not None:
                votes[key] = votes.get(key, 0) + 1
        total = sum(votes.values())
        distribution = {key: count / total for key, count in votes.items()} if total else {}

    if not distribution:
        return None, {}, 0.0, True

    ordered = _choice_order(qtype, distribution, choices)
    top = max(ordered, key=lambda key: distribution[key])
    margin = _margin_of({key: distribution[key] for key in ordered})
    ambiguous = margin < AMBIGUOUS_MARGIN
    label = _label_from_key(top, qtype)
    return label, distribution, margin, ambiguous


def _choice_order(
    qtype: str, distribution: dict[str, float], choices: list[str] | None
) -> list[str]:
    """Distribution keys in the field's choice order (extras appended sorted).

    Deterministic tie-breaking depends on choice order, not dict insertion
    order. Booleans use True-before-False (the published "true"/"false"
    probability keys); enums use the published choice order.
    """
    if qtype == "noul":
        return [key for key in ("true", "false") if key in distribution]
    ordered = [key for key in (choices or ()) if key in distribution]
    ordered += sorted(key for key in distribution if key not in ordered)
    return ordered


def _distribution_key(label, qtype: str) -> str | None:
    """Label form -> distribution key (JSON object keys are always strings)."""
    if label is None:
        return None
    if qtype == "noul":
        return "true" if label else "false"
    return str(label)


def _label_from_key(key: str, qtype: str):
    """Distribution key -> label form (booleans become real bools)."""
    if qtype == "noul":
        return key == "true"
    return key


def _render_doc(doc) -> str:
    return doc if isinstance(doc, str) else json.dumps(doc, indent=1, ensure_ascii=False)


def _node_read_sets(case: dict) -> dict[str, frozenset[int]]:
    """Read-set of document indices per workflow step (node name).

    Every model's node named ``X`` works from the same workflow step, so the
    step's read-set is the union of documents referenced by any of those
    nodes. Reference nodes with no model node (or no documents) read nothing.
    """
    read_sets: dict[str, set[int]] = {}
    for model_data in case.get("models", {}).values():
        for node in model_data.get("nodes", []):
            name = node.get("node")
            if name is None:
                continue
            read_sets.setdefault(name, set())
            if node.get("doc") is not None:
                read_sets[name].add(int(node["doc"]))
    for name in case.get("reference_answers", {}):
        read_sets.setdefault(name, set())
    return {name: frozenset(docs) for name, docs in read_sets.items()}


def _case_context(eval_obj: dict, docs: frozenset[int]) -> str:
    """Render a read-set of documents into one context, in document order."""
    documents = eval_obj["documents"]
    return "\n\n".join(
        f"## Document {i}\n{_render_doc(documents[i])}"
        for i in sorted(docs)
        if 0 <= i < len(documents)
    )


def _model_answers(case: dict, node_name: str, qid: str) -> dict:
    """Per-model published answers for one (node, question): ``{model: raw}``."""
    answers: dict = {}
    for model, model_data in case.get("models", {}).items():
        for node in model_data.get("nodes", []):
            if node.get("node") != node_name:
                continue
            answer = (node.get("answers") or {}).get(qid)
            if answer is None:
                continue
            raw = answer.get(answer.get("type"))
            if raw is not None:
                answers.setdefault(model, raw)
    return answers


def iter_case_records(workflow: str, eval_obj: dict) -> Iterator[dict]:
    """Yield one eval record per distinct read-set of each published case.

    Field identity is ``(node_id, qid, occurrence)``: a qid answered at more
    than one workflow step becomes ``<node_id>__<qid>__<occ>`` (occurrence is
    the 1-based index of that node among the nodes answering the qid); unique
    qids keep their plain name. When the reference nodes of one case have
    different read-sets, the case is split into one record per read-set with
    ``/n<k>`` id suffixes and the original case id as ``group_id``; otherwise
    it stays a single record.
    """
    catalog = eval_obj["questions"]
    for example in eval_obj["examples"]:
        case_id = example["case_id"]
        case = eval_obj["cases"][case_id]
        yield from _case_records(workflow, case_id, case, eval_obj, catalog)


def _case_records(
    workflow: str, case_id: str, case: dict, eval_obj: dict, catalog: list[dict]
) -> Iterator[dict]:
    # qid -> node index map for this case (union over every model's nodes;
    # the catalog is workflow-wide, so any node's map contributes).
    case_qmap: dict[str, int] = {}
    for model_data in case.get("models", {}).values():
        for node in model_data.get("nodes", []):
            for qid, idx in (node.get("questions") or {}).items():
                case_qmap.setdefault(qid, int(idx))

    # Occurrence numbering per qid, in reference-node order.
    qid_counts: dict[str, int] = {}
    for node_answers in case.get("reference_answers", {}).values():
        for qid in node_answers:
            qid_counts[qid] = qid_counts.get(qid, 0) + 1
    occurrences: dict[str, int] = {}

    # Group reference nodes by read-set, preserving first-appearance order.
    read_sets = _node_read_sets(case)
    groups: list[tuple[frozenset[int], list[str]]] = []
    for node_name in case.get("reference_answers", {}):
        for docs, names in groups:
            if docs == read_sets[node_name]:
                names.append(node_name)
                break
        else:
            groups.append((read_sets[node_name], [node_name]))

    for group_index, (docs, node_names) in enumerate(groups):
        schema: dict = {}
        labels: dict = {}
        consensus_meta: dict = {}
        margins: dict = {}
        ambiguous: list = []
        models_meta: dict = {}
        skipped = 0

        for node_name in node_names:
            for qid, entry in case["reference_answers"][node_name].items():
                field = None
                idx = case_qmap.get(qid)
                if idx is not None and 0 <= idx < len(catalog):
                    field = _field_schema(catalog[idx])
                occurrences[qid] = occurrences.get(qid, 0) + 1
                if field is None:
                    skipped += 1
                    continue
                value, distribution, margin, is_ambiguous = _consensus(
                    entry,
                    f"{workflow}/{case_id}/{node_name}/{qid}",
                    choices=field.get("choices"),
                )
                if value is None:
                    skipped += 1
                    continue

                name = qid
                if qid_counts[qid] > 1:
                    name = f"{node_name}__{qid}__{occurrences[qid]}"
                schema[name] = field
                labels[name] = value
                consensus_meta[name] = distribution
                margins[name] = margin
                if is_ambiguous:
                    ambiguous.append(name)
                answers = _model_answers(case, node_name, qid)
                if answers:
                    models_meta[name] = answers

        record_id = f"typesafe/{workflow}/{case_id}"
        if len(groups) > 1:
            record_id += f"/n{group_index}"
        yield {
            "id": record_id,
            "group_id": f"typesafe/{workflow}/{case_id}",
            "source": "typesafe",
            "workflow": workflow,
            "benchmark_only": True,
            "schema": schema,
            "context": _case_context(eval_obj, docs),
            "labels": labels,
            "split": split_for(record_id),
            "meta": {
                "consensus": consensus_meta,
                "margin": margins,
                "ambiguous": ambiguous,
                "models": models_meta,
            },
            "skipped_questions": skipped,
        }


def fetch_all(
    workflows: list[str], cache_dir: Path | None = None, *, refresh: bool = False
) -> tuple[list[dict], dict, list[dict]]:
    """Fetch every workflow; return (records, summary counts, source metadata)."""
    cache_dir = cache_dir or CACHE_DIR
    records: list[dict] = []
    field_types = {"boolean": 0, "enum": 0, "multi": 0}
    skipped_questions = 0
    sources: list[dict] = []
    for workflow in workflows:
        eval_obj, source_meta = fetch_workflow(workflow, cache_dir, refresh=refresh)
        sources.append(source_meta)
        for record in iter_case_records(workflow, eval_obj):
            records.append(record)
            skipped_questions += record.pop("skipped_questions")
            for field in record["schema"].values():
                field_types[field["type"]] += 1
    summary = {
        "workflows": len(workflows),
        "records": len(records),
        "fields": field_types,
        "skipped_questions": skipped_questions,
    }
    return records, summary, sources


def write_outputs(
    records: list[dict],
    out_path: Path,
    sources: list[dict],
    counts: dict,
    lock_path: Path | None = None,
) -> Path:
    """Write cases.jsonl and its dataset.lock.json; return the lock path.

    ``lock_path`` defaults to ``dataset.lock.json`` next to ``--out`` — the
    shared loop in ``jevmlx.bench`` reads the lock from there for every
    fetcher, so the two fetchers must agree on the location.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    cases_sha256 = hashlib.sha256(out_path.read_bytes()).hexdigest()
    lock = {
        "sources": sources,
        "parser_version": PARSER_VERSION,
        "counts": counts,
        "cases_sha256": cases_sha256,
    }
    lock_path = Path(lock_path) if lock_path else out_path.parent / "dataset.lock.json"
    lock_path.write_text(json.dumps(lock, indent=1) + "\n", encoding="utf-8")
    return lock_path


def build_parser() -> argparse.ArgumentParser:
    """The typesafe.fetch CLI parser (exposed for parse-only tests; see
    tests/test_m5_e2e.py TestPlannedArgvParses)."""
    parser = argparse.ArgumentParser(
        prog="python -m benchmarks.typesafe.fetch",
        description="Fetch TypeSafe's public eval examples as jevmlx eval JSONL.",
    )
    parser.add_argument("--out", required=True, help="output JSONL path")
    parser.add_argument(
        "--lock",
        default=None,
        help="lock path (default: dataset.lock.json next to --out)",
    )
    parser.add_argument(
        "--workflow",
        action="append",
        choices=WORKFLOWS,
        help="fetch only this workflow (repeatable; default: all)",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="re-download even when a cached copy exists",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    workflows = list(args.workflow or WORKFLOWS)
    records, summary, sources = fetch_all(workflows, refresh=args.refresh)
    lock_path = write_outputs(records, Path(args.out), sources, summary, args.lock)

    print(f"workflows: {summary['workflows']}")
    print(f"records: {summary['records']}")
    print(f"fields: {summary['fields']['boolean']} boolean, {summary['fields']['enum']} enum")
    print(f"skipped questions (free text): {summary['skipped_questions']}")
    print(f"wrote {args.out}")
    print(f"wrote {lock_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
