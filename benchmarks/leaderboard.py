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

With ``--readme`` the table is written between
``<!-- leaderboard:start -->`` / ``<!-- leaderboard:end -->`` markers.
With ``--check-readme`` the table is built in memory and the script exits 1
when the README's marker block differs.
"""

from __future__ import annotations

import json
import statistics
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

_HEADER = (
    "| Model | Source | Scorer | Machine | Accuracy | Customer service | "
    "Agent trace | Security | Invoices | Time per case | Cost per case | Cases |"
)
_SEP = "|---|---|---|---|---|---|---|---|---|---|---|---|"


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
    """Median per-item END-TO-END latency (ms) from predictions.jsonl.

    Results contract v2 (W5-D finding 27): batched (decide_many) prediction
    lines carry ``per_item_end_to_end_ms`` — that context's own prefill plus
    its share of the group pass — the honest per-case number. Returns None
    when the lines carry no such key: the caller FAILS the folder (results
    contract v2; no pre-v2 folders exist on main, so there is no fallback).
    """
    import gzip

    pred = folder / "predictions.jsonl"
    gz = folder / "predictions.jsonl.gz"
    if not pred.exists() and not gz.exists():
        return None

    def opener():
        if pred.exists():
            return open(pred, encoding="utf-8")
        return gzip.open(gz, "rt", encoding="utf-8")

    values: list[float] = []
    with opener() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ms = json.loads(line).get("per_item_end_to_end_ms")
            if isinstance(ms, int | float):
                values.append(float(ms))
    if not values:
        return None
    return statistics.median(values)


def _local_rows(results_root: Path) -> list[dict]:
    """One row per results folder with dataset=typesafe and track=parallel.

    W4-A: a model appears only if its folder has a passing slow parity test
    (parity.json with ``passed: true``). Without it, the model is excluded
    from the leaderboard — it hasn't proven batch/chunked log_score parity.

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
        # W4-A: skip models without a passing parity test.
        parity_path = machine_dir / "parity.json"
        if not parity_path.exists():
            continue
        try:
            parity = json.loads(parity_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not parity.get("passed"):
            continue
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
            if track != "parallel" or dataset_name != "typesafe":
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
            model = config.get("model", "")
            _, scorer, _ = _combo_parts(combo)
            machine, _ = _machine_model(combo)
            # Time per case in seconds: the honest per-item end-to-end
            # median (results contract v2) — no fallback. A parallel combo
            # without the key is a broken folder: fail loudly.
            end_to_end = _per_item_end_to_end_ms(combo)
            if end_to_end is None:
                raise ValueError(
                    f"{combo}: predictions carry no per_item_end_to_end_ms — "
                    "results contract v2 requires it (rerun the bench; "
                    "check_results rejects this folder too)"
                )
            time_per_case_s = end_to_end / 1000.0
            rows.append(
                {
                    "model": model,
                    "source": "local",
                    "scorer": scorer,
                    "machine": machine,
                    "accuracy": agreement,
                    "by_workflow": {
                        "customer_service": by_workflow.get("customer_service"),
                        "agent_trace_observability": by_workflow.get("agent_trace_observability"),
                        "security_incidents": by_workflow.get("security_incidents"),
                        "invoice_processing": by_workflow.get("invoice_processing"),
                    },
                    "time_per_case_s": time_per_case_s,
                    "cost_per_case_usd": "$0 (local)",
                    "cases": n_cases if n_cases is not None else n_fields,
                }
            )
    return rows


def _row_line(r: dict) -> str:
    """One markdown table row from a row dict."""
    wf = r.get("by_workflow", {}) or {}
    wf_cells = [_fmt_pct(wf.get(key)) for key, _ in _WORKFLOW_COLS]
    cost = r.get("cost_per_case_usd")
    cost_cell = cost if isinstance(cost, str) else _fmt_cost(cost)
    return (
        f"| {r['model']} | {r['source']} | {r.get('scorer', '—')} | "
        f"{r.get('machine', '—')} | {_fmt_pct(r.get('accuracy'))} | "
        f"{' | '.join(wf_cells)} | "
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

    lines: list[str] = []
    lines.append(_HEADER)
    lines.append(_SEP)

    # Group A: TypeSafe official (cited).
    if official_models:
        lines.append(
            f"| **TypeSafe official (cited, retrieved {retrieved})** | | | | | | | | | | | |"
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

    # Group C: jevmlx, local (measured).
    if local_rows:
        lines.append("| **jevmlx, local (measured)** | | | | | | | | | | | |")
        for r in local_rows:
            lines.append(_row_line(r))

    # Caption lines.
    lines.append("")
    lines.append(
        "_Official accuracies are on TypeSafe's full private eval; ours are on "
        "the 20 public example cases, so the numbers are indicative, not the "
        "same test._"
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
