#!/usr/bin/env python3
"""Build the README leaderboard, mirroring TypeSafe's official table shape.

Usage::

    python -m benchmarks.leaderboard --results benchmarks/results \
        --published published_agreement.json --readme README.md

Columns mirror TypeSafe's official page
(https://evals.typesafe.ai/) so readers compare like with like::

    Model | Source | Scorer | Machine | Accuracy | Customer service |
    Agent trace | Security | Invoices | Time per case | Cost per case | Cases

Three row groups, clearly separated:

a. **TypeSafe official (cited)**: transcribed from
   ``benchmarks/typesafe/official.json`` (a citation file, not a measurement).
b. **Published models on the public examples (computed)**: L1's
   ``published_agreement.json`` — agreement on the common subset, computed
   from TypeSafe's published per-model answers.
c. **jevmlx, local (measured)**: ``benchmarks/results`` folders on the
   typesafe dataset, parallel track — Accuracy = agreement on L1's common
   subset, per-workflow agreement, Time per case = median end-to-end latency
   per case, Cost per case = "$0 (local)", Cases = n.
d. **jevmlx, local on LocalLLaMA/typed-decisions (measured)**: same columns
   for results folders on the ``typed-decisions`` dataset (the task published
   as versioned parquet; official ``test`` split, 400 cases). Kept as its own
   group: it is a different test set from the 20 public examples.

With ``--readme`` the table is written between
``<!-- leaderboard:start -->`` / ``<!-- leaderboard:end -->`` markers.
With ``--check-readme`` the table is built in memory and the script exits 1
when the README's marker block differs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

__all__ = ["build_table", "write_readme_block", "check_readme", "main"]

_MARKER_START = "<!-- leaderboard:start -->"
_MARKER_END = "<!-- leaderboard:end -->"

_WORKFLOW_COLS = [
    ("customer_service", "Customer service"),
    ("agent_trace_observability", "Agent trace"),
    ("security_incidents", "Security"),
    ("invoice_processing", "Invoices"),
]

# W6-B1: jabr per-task accuracy columns (8 tasks). These are SEPARATE from
# the TypeSafe workflow columns above — the two never share a row group.
_JABR_TASK_COLS = [
    ("support_department", "Support dept"),
    ("email_intent", "Email intent"),
    ("secret_leak", "Secret leak"),
    ("urgency", "Urgency"),
    ("refund_eligible", "Refund"),
    ("frustration_level", "Frustration"),
    ("incident_severity", "Severity"),
    ("review_sentiment", "Sentiment"),
]

# Local datasets with TypeSafe-style consensus labels -> row-group title.
_LOCAL_GROUPS = {
    "typesafe": "jevmlx, local (measured)",
    "typed-decisions": "jevmlx, local on LocalLLaMA/typed-decisions test split (measured)",
    "authored144": ("jevmlx, local on OpenJev authored144 (model-reviewed, not human-adjudicated)"),
    "perturbations108": (
        "jevmlx, local on OpenJev perturbations108 (model-reviewed, not human-adjudicated)"
    ),
    "jabr": "jevmlx, local on jabr/classifier-benchmark (measured)",
}

_HEADER = (
    "| Model | Source | Scorer | Machine | Accuracy | Parity | "
    "Customer service | "
    "Agent trace | Security | Invoices | Time per case | Cost per case | Cases |"
)
_SEP = "|---|---|---|---|---|---|---|---|---|---|---|---|---|"

# W6-B1: jabr has its OWN table (different columns: 8 task accuracies).
# The TypeSafe table and the jabr table are separate markdown blocks so
# their columns never collide.
_JABR_HEADER = (
    "| Model | Source | Scorer | Machine | Accuracy | "
    + " | ".join(label for _, label in _JABR_TASK_COLS)
    + " | Time per case | Cost per case | Cases |"
)
_JABR_SEP = "|" + "|".join("---" for _ in range(13 + len(_JABR_TASK_COLS))) + "|"


def _fmt_pct(value) -> str:
    """A 0..1 ratio as a percentage string, or — when None.

    L1's by_workflow values are dicts ``{agreed, total, agreement}``;
    official.json's are bare floats. Normalize: extract the ``agreement``
    key from a dict, else use the value directly.
    """
    if isinstance(value, dict):
        value = value.get("agreement")
    if value is None:
        return "—"
    if isinstance(value, int | float):
        return f"{value * 100:.1f}%"
    return str(value)


def _agreement_ci(accuracy, n) -> dict | None:
    """W6-B6b/F1: Wilson CI on the agreement accuracy.

    ``accuracy`` is a 0..1 ratio; ``n`` is the case count. Returns the Wilson
    interval dict, or None when n is too small (the leaderboard prints
    'n too small' instead of a bare percentage).
    """
    from jevmlx.evalmetrics import wilson_interval

    if accuracy is None or n is None or n <= 0:
        return None
    k = round(accuracy * n)
    return wilson_interval(k, n)


def _fmt_pct_with_ci(value, ci) -> str:
    """A percentage with its Wilson interval, or 'n too small'.

    F1: every accuracy on the leaderboard shows the interval next to the
    point estimate. No CI = 'n too small', not an unqualified number.
    Official cited rows (no CI key at all) show the bare percentage.
    """
    pct = _fmt_pct(value)
    if isinstance(ci, dict) and ci.get("ci_low") is not None and ci.get("ci_high") is not None:
        return f"{pct} [{ci['ci_low'] * 100:.1f}%, {ci['ci_high'] * 100:.1f}%]"
    if ci is None:
        # No CI key at all (official cited data) — bare percentage.
        return pct
    # CI key is present but None (bootstrap ran, too few cases).
    return f"{pct} (n too small)"


def _fmt_seconds(value) -> str:
    if value is None:
        return "—"
    if isinstance(value, int | float):
        return f"{value:.1f}s"
    return str(value)


def _fmt_cost(value) -> str:
    if value is None:
        return "—"
    if isinstance(value, int | float):
        return f"${value:.4f}"
    return str(value)


def _load_official(path: Path) -> tuple[list[dict], str]:
    """official.json -> (models, retrieved-date caption)."""
    if not path.exists():
        return [], ""
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("models", []), data.get("retrieved", "")


def _load_published(path: Path | None) -> tuple[list[dict], dict]:
    """L1's published_agreement.json -> (models, subset).

    Strict L1 shape only (from
    ``benchmarks.typesafe.published.published_agreement``)::

        {"subset": {"n_fields": N, "n_cases": M, "workflows": [...],
                   "field_ids": [...]},
         "models": [{"name": ..., "agreed": A, "total": T, "agreement": 0.xx,
                     "by_workflow": {wf: {"agreed", "total", "agreement"}}}]}

    Raises ``ValueError`` naming the file if ``subset`` or ``models`` is
    missing — no silent fallback to a bare list.
    """
    if path is None or not path.exists():
        return [], {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "subset" not in data or "models" not in data:
        missing = (
            sorted(set(["subset", "models"]) - set(data))
            if isinstance(data, dict)
            else ["subset", "models"]
        )
        raise ValueError(
            f"{path}: expected L1 shape {{'subset': ..., 'models': ...}}, missing keys: {missing}"
        )
    return data["models"], data["subset"]


def _per_item_end_to_end_ms(folder: Path) -> float | None:
    """The call-level per_item_end_to_end_ms median from timing.json.

    parity-gates (review fix 2): ONE definition — the same source
    summarize_results uses since #99 (timing.json median
    per_item_end_to_end_ms), NOT the per-line latency_ms from predictions
    (which sums rotations and inflated the 7B's time/case to 11.0 s while
    the call-level median was 0.59 s). Returns None when timing.json is
    absent or has no median — the caller FAILS the folder (no fallback to
    line latency).
    """
    timing_path = folder / "timing.json"
    if not timing_path.is_file():
        return None
    try:
        timing = json.loads(timing_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    median = timing.get("median") or {}
    ms = median.get("per_item_end_to_end_ms")
    if isinstance(ms, (int, float)):
        return float(ms)
    return None


def _local_rows(results_root: Path) -> list[dict]:
    """One row per results folder with dataset=typesafe and track=parallel.

    W4-A / issue parity-gates: a model appears if its folder's parity.json
    records status PASS or DRIFT. PASS = all drifts < atol. DRIFT = some
    drift >= atol but winners identical on all cases AND max drift inside the
    persisted envelope band (batch-shape noise, not a real divergence) —
    publishable, the Parity column shows the word + max drift. FAIL (a winner
    changed, or drift beyond the band) stays excluded.

    Time per case (review follow-up on #48): read ONLY the honest
    per-item end-to-end median (results contract v2). There are no pre-v2
    folders on main, so a parallel combo whose predictions lack
    ``per_item_end_to_end_ms`` is a broken folder, not a fallback case —
    it raises ``ValueError`` naming the folder (check_results already
    rejects the same shape).
    """
    from benchmarks.summarize_results import _combo_parts, _machine_model

    rows: list[dict] = []
    if not results_root.exists():
        return rows
    for machine_dir in sorted(p for p in results_root.iterdir() if p.is_dir()):
        # W4-A / parity-gates: include PASS and DRIFT; exclude FAIL and
        # missing/unreadable parity.
        parity_path = machine_dir / "parity.json"
        if not parity_path.exists():
            continue
        try:
            parity = json.loads(parity_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        parity_status = parity.get("status", "PASS")
        if parity_status not in ("PASS", "DRIFT"):
            continue
        # The max drift for the Parity column (DRIFT rows show it; PASS
        # rows show 0 / the measured value, which is < atol).
        parity_max_drift = max(
            parity.get("max_abs_drift_nats", 0.0) or 0.0,
            parity.get("max_gap_drift_nats", 0.0) or 0.0,
            parity.get("max_margin_drift_nats", 0.0) or 0.0,
        )
        for combo in sorted(p for p in machine_dir.iterdir() if p.is_dir()):
            report_path = combo / "report.json"
            if not report_path.exists():
                continue
            report = json.loads(report_path.read_text(encoding="utf-8"))
            run_path = combo / "run.json"
            run = json.loads(run_path.read_text(encoding="utf-8")) if run_path.exists() else {}
            config = run.get("config", {})
            track = config.get("track", "")
            dataset = config.get("dataset_path", "") or ""
            dataset_name = dataset if "/" not in dataset else Path(dataset).stem
            if track != "parallel" or dataset_name not in _LOCAL_GROUPS:
                continue
            metrics = report.get("metrics", {})
            ta = metrics.get("agreement", {})
            ta = ta if isinstance(ta, dict) else {}
            agreement = ta.get("agreement_common_subset")
            by_workflow = (
                ta.get("by_workflow", {}) if isinstance(ta.get("by_workflow"), dict) else {}
            )
            n_cases = ta.get("n_cases")
            n_fields = ta.get("n_fields")
            # parity-gates (review fix 3): 'Cases' comes from run.json's
            # counts.cases (the source of truth), not from the agreement
            # metrics' n_cases (which can undercount when a case has no
            # valid prediction). If counts.cases is missing, fall back to
            # the agreement n_cases, then n_fields.
            run_counts_cases = (run.get("counts") or {}).get("cases")
            cases_count = (
                run_counts_cases
                if run_counts_cases is not None
                else (n_cases if n_cases is not None else n_fields)
            )
            # W6-B1: for non-TypeSafe datasets (jabr), the per-workflow
            # breakdown comes from per_workflow_accuracy (general metric),
            # and the overall accuracy from metrics["accuracy"].
            if dataset_name == "jabr":
                agreement = metrics.get("accuracy")
                pwa = metrics.get("per_workflow_accuracy", {})
                by_workflow = pwa if isinstance(pwa, dict) else {}
                # n_cases from the per-field accuracy entries (one field per
                # case in jabr).
                pfa = metrics.get("per_field_accuracy", [])
                if isinstance(pfa, list) and pfa:
                    n_fields = sum(e.get("n", 0) for e in pfa if isinstance(e, dict))
                else:
                    n_fields = None
                n_cases = n_fields
            model = config.get("model", "")
            _, scorer, _ = _combo_parts(combo)
            machine, _ = _machine_model(combo)
            # Time per case in seconds: the honest per-item end-to-end
            # median (results contract v2) — no fallback. A parallel combo
            # without the key is a broken folder: fail loudly.
            end_to_end = _per_item_end_to_end_ms(combo)
            if end_to_end is None:
                raise ValueError(
                    f"{combo}: timing.json has no median per_item_end_to_end_ms — "
                    "results contract v2 requires it (rerun the bench; "
                    "check_results rejects this folder too)"
                )
            time_per_case_s = end_to_end / 1000.0
            rows.append(
                {
                    "dataset": dataset_name,
                    # Provenance (F7): the revision recorded in the copied
                    # <name>.dataset.lock.json (run.json carries the same
                    # lock's sha256). Rows are grouped per revision below.
                    "revision": _lock_revision(combo, dataset_name),
                    "model": model,
                    "source": "local",
                    "scorer": scorer,
                    "machine": machine,
                    "parity_status": parity_status,
                    "parity_max_drift": parity_max_drift,
                    "accuracy": agreement,
                    # W6-B6b/F1: Wilson CI on the agreement accuracy.
                    "accuracy_ci": _agreement_ci(agreement, cases_count),
                    "by_workflow": {
                        "customer_service": by_workflow.get("customer_service"),
                        "agent_trace_observability": by_workflow.get("agent_trace_observability"),
                        "security_incidents": by_workflow.get("security_incidents"),
                        "invoice_processing": by_workflow.get("invoice_processing"),
                    }
                    if dataset_name != "jabr"
                    else {key: by_workflow.get(key) for key, _ in _JABR_TASK_COLS},
                    "time_per_case_s": time_per_case_s,
                    "cost_per_case_usd": "$0 (local)",
                    "cases": cases_count,
                }
            )
    return rows


def _lock_revision(combo: Path, dataset_name: str) -> str | None:
    """The dataset revision recorded in <name>.dataset.lock.json.

    Since #84/#110 the lock lives at the MODEL folder level (the parent of
    the combo), not in the combo folder itself.
    """
    lock_path = combo.parent / f"{dataset_name}.dataset.lock.json"
    if not lock_path.exists():
        return None
    try:
        sources = json.loads(lock_path.read_text(encoding="utf-8")).get("sources", [])
    except (OSError, json.JSONDecodeError):
        return None
    for source in sources:
        revision = source.get("revision")
        if revision:
            return str(revision)
    return None


def _refuse_mixed_revisions(rows: list[dict]) -> None:
    """Fail when a dataset's local rows span more than one dataset revision.

    Grouping accuracies across revisions would average numbers measured on
    different data; the bench pins one revision per folder, so a mixed set
    is a broken results tree, not a mergeable one.
    """
    by_dataset: dict[str, set[str | None]] = {}
    for row in rows:
        if row.get("source") == "local" and row.get("revision") is not None:
            by_dataset.setdefault(row["dataset"], set()).add(row["revision"])
    for dataset, revisions in by_dataset.items():
        if len(revisions) > 1:
            raise ValueError(
                f"{dataset}: local rows span {len(revisions)} dataset revisions "
                f"({', '.join(sorted(r or 'none' for r in revisions))}) — group or "
                "rerun per revision; refusing to average across revisions"
            )


def _row_line(r: dict) -> str:
    """One markdown table row from a row dict (TypeSafe workflow columns)."""
    wf = r.get("by_workflow", {}) or {}
    wf_cells = [_fmt_pct(wf.get(key)) for key, _ in _WORKFLOW_COLS]
    cost = r.get("cost_per_case_usd")
    cost_cell = cost if isinstance(cost, str) else _fmt_cost(cost)
    # Parity column: PASS shows the word; DRIFT shows the word + max drift
    # (publishable batch-shape noise); the column never shows FAIL (those
    # models are excluded by _local_rows).
    parity_word = r.get("parity_status", "—")
    if parity_word == "DRIFT":
        parity_cell = f"DRIFT ({r.get('parity_max_drift', 0):.3f})"
    else:
        parity_cell = str(parity_word)
    return (
        f"| {r['model']} | {r['source']} | {r.get('scorer', '—')} | "
        f"{r.get('machine', '—')} | {_fmt_pct_with_ci(r.get('accuracy'), r.get('accuracy_ci'))} | "
        f"{parity_cell} | "
        f"{' | '.join(wf_cells)} | "
        f"{_fmt_seconds(r.get('time_per_case_s'))} | "
        f"{cost_cell} | "
        f"{r.get('cases', '—')} |"
    )


def _jabr_row_line(r: dict) -> str:
    """One markdown table row for the jabr table (8 task accuracy columns)."""
    wf = r.get("by_workflow", {}) or {}
    task_cells = [_fmt_pct(wf.get(key)) for key, _ in _JABR_TASK_COLS]
    cost = r.get("cost_per_case_usd")
    cost_cell = cost if isinstance(cost, str) else _fmt_cost(cost)
    return (
        f"| {r['model']} | {r['source']} | {r.get('scorer', '—')} | "
        f"{r.get('machine', '—')} | {_fmt_pct_with_ci(r.get('accuracy'), r.get('accuracy_ci'))} | "
        f"{' | '.join(task_cells)} | "
        f"{_fmt_seconds(r.get('time_per_case_s'))} | "
        f"{cost_cell} | "
        f"{r.get('cases', '—')} |"
    )


def build_table(
    results_root: Path | None = None,
    published_path: Path | None = None,
    official_path: Path | None = None,
) -> str:
    """Build the leaderboard markdown (caption + 3 row groups + table)."""
    if official_path is None:
        official_path = Path("benchmarks/typesafe/official.json")
    official_models, retrieved = _load_official(official_path)
    published_models, published_subset = _load_published(published_path)
    local_rows = _local_rows(results_root) if results_root else []
    _refuse_mixed_revisions(local_rows)

    lines: list[str] = []
    lines.append(_HEADER)
    lines.append(_SEP)

    # Group A: TypeSafe official (cited).
    if official_models:
        lines.append(
            f"| **TypeSafe official (cited, retrieved {retrieved})** | | | | | | | | | | | | |"
        )
        for m in official_models:
            bw = m.get("by_workflow", {}) or {}
            lines.append(
                _row_line(
                    {
                        "model": m.get("name", ""),
                        "source": "official (cited)",
                        "scorer": "—",
                        "machine": "—",
                        "accuracy": m.get("accuracy"),
                        "by_workflow": {
                            "customer_service": bw.get("customer_service"),
                            "agent_trace_observability": bw.get("agent_trace_observability"),
                            "security_incidents": bw.get("security_incidents"),
                            "invoice_processing": bw.get("invoice_processing"),
                        },
                        "time_per_case_s": m.get("time_per_case_s"),
                        "cost_per_case_usd": m.get("cost_per_case_usd"),
                        "cases": "—",
                    }
                )
            )

    # Group B: Published models on the public examples (computed).
    if published_models:
        lines.append(
            "| **Published models on the public examples (computed)** | | | | | | | | | | | |"
        )
        for m in published_models:
            bw = m.get("by_workflow", {}) or {}
            lines.append(
                _row_line(
                    {
                        "model": m.get("name", ""),
                        "source": "computed from published answers",
                        "scorer": "—",
                        "machine": "—",
                        "accuracy": m.get("agreement"),
                        "by_workflow": {
                            "customer_service": bw.get("customer_service"),
                            "agent_trace_observability": bw.get("agent_trace_observability"),
                            "security_incidents": bw.get("security_incidents"),
                            "invoice_processing": bw.get("invoice_processing"),
                        },
                        "time_per_case_s": None,
                        "cost_per_case_usd": None,
                        "cases": published_subset.get("n_cases", "—")
                        if published_subset
                        else m.get("total", "—"),
                    }
                )
            )

    # Groups C/D: jevmlx, local (measured), one group per dataset — they are
    # different test sets and must never share a column.
    for dataset_name, title in _LOCAL_GROUPS.items():
        if dataset_name == "jabr":
            continue  # jabr gets its own table below (different columns)
        group = [r for r in local_rows if r.get("dataset") == dataset_name]
        if group:
            lines.append(f"| **{title}** | | | | | | | | | | | | |")
            lines.extend(_row_line(r) for r in group)

    # W6-B1: jabr table — separate block with 8 per-task accuracy columns.
    jabr_rows = [r for r in local_rows if r.get("dataset") == "jabr"]
    if jabr_rows:
        if local_rows or official_models or published_models:
            lines.append("")
            lines.append("")
        lines.append("### jabr/classifier-benchmark")
        lines.append("")
        lines.append(_JABR_HEADER)
        lines.append(_JABR_SEP)
        for r in jabr_rows:
            lines.append(_jabr_row_line(r))
        lines.append("")
        lines.append(
            "_jabr rows are on the jabr/classifier-benchmark suite "
            "(https://github.com/jabr/classifier-benchmark) — 8 tasks, "
            "78 cases, pinned at commit "
            f"{jabr_rows[0].get('revision') or 'unpinned'}._"
        )
        lines.append(
            "_Per-task columns: support_department (5-way), email_intent "
            "(5-way), secret_leak (bool), urgency (bool), refund_eligible "
            "(bool), frustration_level (3-level ordinal), incident_severity "
            "(5-level ordinal), review_sentiment (5-level ordinal)._"
        )

    # Caption lines.
    lines.append("")
    lines.append(
        "_Official accuracies are on TypeSafe's full private eval; ours are on "
        "the 20 public example cases, so the numbers are indicative, not the "
        "same test._"
    )
    typed_rows = [r for r in local_rows if r.get("dataset") == "typed-decisions"]
    if typed_rows:
        n_cases = sorted({r["cases"] for r in typed_rows if r.get("cases") is not None})
        cases_txt = f"{n_cases[0]} cases" if len(n_cases) == 1 else f"{n_cases} cases per row"
        revisions = sorted({r.get("revision") or "unpinned" for r in typed_rows})
        rev_txt = (
            f"revision {revisions[0]} pinned in each folder's dataset.lock.json"
            if len(revisions) == 1
            else "revisions pinned in each folder's dataset.lock.json"
        )
        lines.append(
            "_typed-decisions rows are on LocalLLaMA/typed-decisions"
            "(https://huggingface.co/datasets/LocalLLaMA/typed-decisions) "
            f"({cases_txt}), {rev_txt}._"
        )
    lines.append(
        "_Consensus label = the agreement of GPT-6 Astra + Claude Fable 5.1 "
        "(TypeSafe's reference)._"
    )
    if published_subset:
        n_fields = published_subset.get("n_fields")
        n_cases = published_subset.get("n_cases")
        workflows = published_subset.get("workflows", [])
        if n_fields is not None:
            wf_str = ", ".join(workflows) if workflows else "—"
            lines.append(
                f"_Common subset: {n_fields} fields across {n_cases} cases (workflows: {wf_str})._"
            )
    lines.append("")
    if not local_rows:
        lines.append("No local results yet — contribute one with `jevmlx bench`.")
    return "\n".join(lines)


def _readme_block(table: str) -> str:
    return f"{_MARKER_START}\n{table}\n{_MARKER_END}"


def _extract_block(readme: str) -> str | None:
    start = readme.find(_MARKER_START)
    end = readme.find(_MARKER_END)
    if start == -1 or end == -1 or end < start:
        return None
    return readme[start : end + len(_MARKER_END)]


def write_readme_block(readme_path: Path, table: str) -> bool:
    """Replace the marker block in README; return True if it changed."""
    readme = readme_path.read_text(encoding="utf-8")
    new_block = _readme_block(table)
    current = _extract_block(readme)
    if current == new_block:
        return False
    if current is None:
        readme = readme.rstrip() + "\n\n" + new_block + "\n"
    else:
        readme = readme.replace(current, new_block)
    readme_path.write_text(readme, encoding="utf-8")
    return True


def check_readme(readme_path: Path, table: str) -> bool:
    readme = readme_path.read_text(encoding="utf-8")
    current = _extract_block(readme)
    return current == _readme_block(table)


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Build the README leaderboard.")
    ap.add_argument("--results", default="benchmarks/results", help="results root")
    ap.add_argument("--published", default=None, help="published_agreement.json (L1 output)")
    ap.add_argument(
        "--official",
        default="benchmarks/typesafe/official.json",
        help="official.json citation file",
    )
    ap.add_argument("--readme", default=None, help="README.md path (writes the marker block)")
    ap.add_argument(
        "--check-readme",
        action="store_true",
        help="build the table in memory and exit 1 if README's block differs",
    )
    args = ap.parse_args(argv)

    table = build_table(
        Path(args.results),
        Path(args.published) if args.published else None,
        Path(args.official),
    )

    if args.check_readme:
        if not args.readme:
            print("--check-readme requires --readme", file=sys.stderr)
            return 2
        if check_readme(Path(args.readme), table):
            print("README leaderboard block is up to date.")
            return 0
        print("README leaderboard block is STALE — regenerate with:", file=sys.stderr)
        cmd = (
            f"  python -m benchmarks.leaderboard --results {args.results}"
            f" --official {args.official}"
        )
        if args.published:
            cmd += f" --published {args.published}"
        cmd += f" --readme {args.readme}"
        print(cmd, file=sys.stderr)
        return 1

    print(table)
    if args.readme:
        changed = write_readme_block(Path(args.readme), table)
        print(f"\n({'updated' if changed else 'unchanged'}) {args.readme}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
