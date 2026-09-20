"""Fetch the jabr classifier-benchmark as jevmlx eval JSONL (W6-B1).

Source: https://github.com/jabr/classifier-benchmark — a public-domain
(CC0) benchmark of 8 classification tasks / 78 cases for "System One"-style
decision models. The case definitions live in ``bench/cases.py`` as Python
source that imports ``von.types`` (which we do not depend on), so the
fetcher downloads the PINNED file, verifies its sha256 against a hardcoded
expectation, and parses it with the ``ast`` module — never ``exec``.

The pinning + fail-closed verification follows the SAME pattern as
``benchmarks/public/fetch.py`` (the HF Hub datasets): pinned upstream commit
sha, ``EXPECTED_SHA256`` of the fetched file, ``OSError`` on mismatch.

Unlike the HF datasets, jabr is a SINGLE-VIEW dataset — all 78 cases are the
benchmark (no balanced/natural sampling; the set is too small and every case
is hand-curated). Each case becomes one eval record with
``workflow = task_id`` so the report's ``by_workflow`` carries per-task
accuracy, and the leaderboard shows per-task columns.

Task -> schema mapping (via the shared ``field_schema`` in
``benchmarks.typesafe.questions``, no local copy):

- ``choice`` (support_department 5-way, email_intent 5-way) -> enum over the
  criteria keys.
- ``noul`` (secret_leak bool, urgency bool, refund_eligible bool) -> boolean.
- ``score`` (frustration_level 3-level, incident_severity 5-level,
  review_sentiment 5-level) -> ordered enum (``ordered: True``) so
  OrdinalTelemetry applies; the level descriptions fold into the field
  description.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import urllib.request
from pathlib import Path

from benchmarks.typesafe.questions import field_schema, label_form

PARSER_VERSION = "1"

# The PINNED upstream commit sha (40-hex, never a floating branch).
REPO_ID = "jabr/classifier-benchmark"
REVISION = "7acb547c63ce333cd9e2a4d9b6166738e191e025"
SOURCE_FILE = "bench/cases.py"
LICENSE = "CC0 1.0 Universal (public domain; redistribution allowed)"
REDISTRIBUTION_ALLOWED = True

# F5: the KNOWN-GOOD sha256 of bench/cases.py at the pinned revision.
# Verified on 2026-09-20 against the raw.githubusercontent.com bytes.
EXPECTED_SHA256 = "fd096adc2fa1d678add69e9769a4e8b6bdf108f41c54d58ea0b7c7e04925835d"

RAW_URL = f"https://raw.githubusercontent.com/jabr/classifier-benchmark/{REVISION}/{SOURCE_FILE}"

# The site rejects the default urllib User-Agent (HTTP 403).
USER_AGENT = "jevmlx-bench/1.0"


# ---- AST parser -----------------------------------------------------------


def _const_value(node: ast.AST):
    """Extract a Python literal from an AST node, or None."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_const_value(node.operand)
    return None


def _extract_kwargs(call: ast.Call) -> dict:
    """Keyword arguments of a Call node as a dict of literal values."""
    kwargs: dict = {}
    for kw in call.keywords:
        if kw.arg is not None and kw.value is not None:
            kwargs[kw.arg] = _const_value(kw.value)
    return kwargs


def _extract_dict(node: ast.AST) -> dict | None:
    """Extract a dict literal {key: value} from an AST node."""
    if isinstance(node, ast.Dict):
        result: dict = {}
        for key, value in zip(node.keys, node.values, strict=False):
            k = _const_value(key)
            v = _const_value(value)
            if k is not None:
                result[k] = v
        return result
    return None


def _extract_list(node: ast.AST) -> list | None:
    """Extract a list literal from an AST node."""
    if isinstance(node, ast.List):
        return [_const_value(elt) for elt in node.elts]
    return None


def _extract_cases(cases_node: ast.AST) -> list[dict]:
    """Extract Case(state, expected) constructors from the cases list."""
    cases: list[dict] = []
    if not isinstance(cases_node, ast.List):
        return cases
    for elt in cases_node.elts:
        if not isinstance(elt, ast.Call):
            continue
        kwargs = _extract_kwargs(elt)
        # Case("state text", True/False/"label"/0) — positional args.
        args = [_const_value(a) for a in elt.args]
        state = kwargs.get("state") or (args[0] if len(args) > 0 else "")
        expected = (
            kwargs.get("expected") if "expected" in kwargs else (args[1] if len(args) > 1 else None)
        )
        cases.append({"state": state, "expected": expected})
    return cases


def _extract_question(call: ast.Call) -> dict:
    """Extract a Choice/Noul/Score question from its constructor Call."""
    func = call.func
    qtype = ""
    if isinstance(func, ast.Name):
        qtype = func.id  # Choice, Noul, Score
    kwargs = _extract_kwargs(call)
    question: dict = {"type": qtype.lower(), "instructions": kwargs.get("instructions", "")}
    criteria = kwargs.get("criteria")
    if isinstance(criteria, dict):
        question["criteria"] = criteria
    elif isinstance(criteria, list):
        question["criteria"] = criteria
    else:
        # Try the AST dict/list nodes directly (kwargs may be None for
        # non-literal values, but criteria is always a literal in cases.py).
        for kw in call.keywords:
            if kw.arg == "criteria" and kw.value is not None:
                d = _extract_dict(kw.value)
                if d is not None:
                    question["criteria"] = d
                else:
                    lst = _extract_list(kw.value)
                    if lst is not None:
                        question["criteria"] = lst
                break
    return question


def parse_cases_source(source: str) -> list[dict]:
    """Parse bench/cases.py with AST and return a list of task dicts.

    Each task dict: ``{id, type, question: {type, instructions, criteria},
    cases: [{state, expected}]}``.

    Never ``exec``s the source — pure AST extraction. Raises ``ValueError``
    if the structure is not the expected Task/Case/Choice/Noul/Score shape.
    """
    tree = ast.parse(source, filename=SOURCE_FILE)
    tasks: list[dict] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        # Each task function returns Task(id=..., type=..., question=..., cases=[...]).
        ret = None
        for child in ast.walk(node):
            if isinstance(child, ast.Return) and isinstance(child.value, ast.Call):
                ret = child.value
                break
        if ret is None:
            continue
        func = ret.func
        if not (isinstance(func, ast.Name) and func.id == "Task"):
            continue
        kwargs = _extract_kwargs(ret)
        task_id = kwargs.get("id", node.name)
        task_type = kwargs.get("type", "")
        # The question is a positional or keyword arg.
        question_dict: dict | None = None
        for kw in ret.keywords:
            if kw.arg == "question" and isinstance(kw.value, ast.Call):
                question_dict = _extract_question(kw.value)
                break
        if question_dict is None and ret.args:
            # question might be positional (first arg after self).
            for arg in ret.args:
                if isinstance(arg, ast.Call):
                    question_dict = _extract_question(arg)
                    break
        # The cases list.
        cases: list[dict] = []
        for kw in ret.keywords:
            if kw.arg == "cases":
                cases = _extract_cases(kw.value)
                break
        if not question_dict or not cases:
            continue
        tasks.append(
            {
                "id": task_id,
                "type": task_type or question_dict["type"],
                "question": question_dict,
                "cases": cases,
            }
        )
    if not tasks:
        raise ValueError(
            f"{SOURCE_FILE}: no Task definitions found — the source structure "
            "changed; refusing to produce an empty dataset"
        )
    return tasks


# ---- schema + label mapping -----------------------------------------------


def _task_schema(task: dict) -> dict:
    """Map a jabr task to a jevmlx schema dict (one field, shared mapping).

    Uses ``field_schema`` from ``benchmarks.typesafe.questions`` — the SAME
    mapping owner as the HF datasets. ``choice`` -> enum, ``noul`` -> boolean,
    ``score`` -> ordered enum (OrdinalTelemetry).
    """
    question = task["question"]
    qtype = question["type"]
    # score tasks with N levels: override score_choices to the task's range.
    criteria = question.get("criteria", [])
    if qtype == "score" and isinstance(criteria, list):
        score_choices = [str(i) for i in range(len(criteria))]
        field = field_schema(question, score_choices=tuple(score_choices))
    else:
        field = field_schema(question)
    if field is None:
        raise ValueError(f"jabr task {task['id']}: unknown question type {qtype!r}")
    field_name = task["id"]
    return {field_name: field}


def _task_label(task: dict, expected) -> object:
    """A gold label in jevmlx label form (booleans become real bools)."""
    return label_form(expected, task["question"]["type"])


def _option_order(task: dict, schema: dict) -> list:
    """The option order for the task's field (enum choices or bool labels)."""
    field = next(iter(schema.values()))
    if field["type"] == "enum":
        return field["choices"]
    return ["false", "true"]


def _input_hash(schema: dict, text: str) -> str:
    """sha256 of schema + real text (binds the input the engine sees)."""
    payload = json.dumps({"schema": schema, "text": text}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ---- download + verify ----------------------------------------------------


def _download() -> tuple[str, str]:
    """Download the pinned file and VERIFY it (F5).

    Returns (source_text, sha256). Any mismatch is a hard OSError (never
    parsed, never locked). Network failure is also OSError (not SystemExit)
    so build_datasets' offline skip works.
    """
    request = urllib.request.Request(RAW_URL, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
            source = response.read().decode("utf-8")
    except Exception as exc:  # noqa: BLE001 — one clear message for any cause
        raise OSError(
            f"{REPO_ID}@{REVISION} [{SOURCE_FILE}]: download failed "
            f"({exc.__class__.__name__}). The fetch needs network access to "
            f"raw.githubusercontent.com; locally run:\n"
            f"  curl -sL {RAW_URL}"
        ) from exc
    actual = hashlib.sha256(source.encode("utf-8")).hexdigest()
    if actual != EXPECTED_SHA256:
        raise OSError(
            f"{REPO_ID}@{REVISION} [{SOURCE_FILE}]: sha256 mismatch — "
            f"expected {EXPECTED_SHA256}, got {actual}. The pinned bytes "
            "changed; FAILING CLOSED (do not parse, do not lock)."
        )
    return source, actual


# ---- build + write --------------------------------------------------------


def build_records() -> tuple[list[dict], str]:
    """Fetch, verify, parse, and convert to eval JSONL records.

    Returns (records, file_sha256). Every record carries the REAL case text
    in ``context`` (the engine classifies it); results artifacts store row
    ids, task, option order and input hashes only.
    """
    source, file_sha256 = _download()
    tasks = parse_cases_source(source)
    records: list[dict] = []
    for task in tasks:
        schema = _task_schema(task)
        field_name = next(iter(schema))
        for index, case in enumerate(task["cases"]):
            context = case["state"]
            expected = case["expected"]
            label = _task_label(task, expected)
            record_id = f"jabr/test/{task['id']}/{index:04d}"
            records.append(
                {
                    "id": record_id,
                    "group_id": record_id,
                    "source": "jabr",
                    "view": "all",
                    "benchmark_only": True,
                    "workflow": task["id"],
                    "schema": schema,
                    "context": context,
                    "labels": {field_name: label},
                    "split": "test",
                    "meta": {
                        "source_repo": REPO_ID,
                        "source_revision": REVISION,
                        "source_file": SOURCE_FILE,
                        "task": task["id"],
                        "task_type": task["type"],
                        "source_row_index": index,
                        "option_order": _option_order(task, schema),
                        "selected_input_hash": _input_hash(schema, context),
                    },
                }
            )
    if not records:
        raise OSError("jabr: no cases parsed — refusing to write an empty dataset")
    return records, file_sha256


def write_dataset(records: list[dict], out_path: Path, lock_path: Path | None = None) -> Path:
    """Write the cases JSONL + lock; return the lock path."""
    if not records:
        raise OSError("jabr: refusing to write an empty dataset")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    # Per-task counts for the lock.
    by_task: dict[str, int] = {}
    for record in records:
        task = record["meta"]["task"]
        by_task[task] = by_task.get(task, 0) + 1

    lock = {
        "sources": [
            {
                "repo_id": REPO_ID,
                "revision": REVISION,
                "license": LICENSE,
                "redistribution_allowed": REDISTRIBUTION_ALLOWED,
                "files": {SOURCE_FILE: EXPECTED_SHA256},
            }
        ],
        "parser_version": PARSER_VERSION,
        "counts": {
            "records": len(records),
            "tasks": len(by_task),
            "by_task": dict(sorted(by_task.items())),
        },
        "cases_sha256": hashlib.sha256(out_path.read_bytes()).hexdigest(),
    }
    lock_path = Path(lock_path) if lock_path else out_path.parent / "dataset.lock.json"
    lock_path.write_text(json.dumps(lock, indent=1) + "\n", encoding="utf-8")
    return lock_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m benchmarks.public.jabr",
        description="Fetch the jabr classifier-benchmark as jevmlx eval JSONL.",
    )
    parser.add_argument("--out", required=True, help="output .jsonl path")
    parser.add_argument("--lock", required=True, help="output .dataset.lock.json path")
    args = parser.parse_args(argv)
    records, _sha = build_records()
    write_dataset(records, Path(args.out), Path(args.lock))
    n_tasks = len({r["meta"]["task"] for r in records})
    print(f"wrote {args.out} ({len(records)} cases, {n_tasks} tasks)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
