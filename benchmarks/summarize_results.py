#!/usr/bin/env python3
"""Summarize a bench results folder into SUMMARY.md.

Reads every ``report.json`` under the results root (one folder per
track-scorer-dataset combination) and writes one table row per folder:
machine, model, track, scorer, dataset, field accuracy, case exact match,
balanced accuracy mean, ECE, any-flip rate, perturbation flip rate, median
latency, and case count. Also enforces the 5 MB folder rule (gzip large
predictions) and writes the folder README.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from jevmlx.bench import enforce_folder_size

_MACHINE_RE = re.compile(r"^(?P<machine>[a-z0-9]+-\d+gb)-(?P<model>.+)$")


def _row_from_folder(folder: Path) -> dict | None:
    """One summary row from a combo folder (report or failure run.json)."""
    report_path = folder / "report.json"
    if not report_path.is_file():
        # Failure rows: run.json with status load_failed/run_failed gets a
        # summary row with the error in the accuracy column.
        run_path = folder / "run.json"
        if run_path.is_file():
            try:
                run = json.loads(run_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return None
            status = run.get("status")
            if status in ("load_failed", "run_failed"):
                machine, model = _machine_model(folder)
                track, scorer, dataset = _combo_parts(folder)
                err = run.get("error") or {}
                msg = f"{status}: {err.get('type', 'Error')}: {err.get('message', '')}".strip()
                return {
                    "machine": machine,
                    "model": model,
                    "track": track,
                    "scorer": scorer,
                    "dataset": dataset,
                    "field_accuracy": None,
                    "accuracy_note": msg,
                    "case_exact": None,
                    "majority_baseline": None,
                    "exact_record": None,
                    "balanced_accuracy_mean": None,
                    "ece": None,
                    "any_flip_rate": None,
                    "perturbation_flip_rate": None,
                    "latency_ms_p50": None,
                    "n_calls": None,
                    "n_cases": None,
                }
        return None
    run = json.loads(report_path.read_text(encoding="utf-8"))
    metrics = run.get("metrics") or {}
    latency, n_cases, n_calls = _run_extras(folder)

    # Combo folder names are <machine>-<model>/<track>-<scorer>-<dataset>;
    # the machine/model live one level up.
    machine, model = _machine_model(folder)
    track, scorer, dataset = _combo_parts(folder)

    return {
        "machine": machine,
        "model": model,
        "track": track,
        "scorer": scorer,
        "dataset": dataset,
        "field_accuracy": metrics.get("accuracy"),
        "case_exact": metrics.get("case_exact_match", metrics.get("exact_match")),
        # P7: majority baseline (mean over fields) and exact-record accuracy
        # are first-class columns next to field accuracy.
        "majority_baseline": _mean_or_none(metrics.get("majority_class_baseline")),
        "exact_record": metrics.get("exact_record_accuracy"),
        "balanced_accuracy_mean": _balanced_mean(metrics),
        "ece": metrics.get("ece_5bin_equal_mass", metrics.get("ece")),
        "any_flip_rate": _mean_or_none(metrics.get("any_flip_rate")),
        "perturbation_flip_rate": _mean_or_none(metrics.get("perturbation_flip_rate")),
        "latency_ms_p50": metrics.get("latency_ms_p50") or latency,
        "n_calls": n_calls,
        "n_cases": metrics.get("n_cases") or n_cases,
    }


def _machine_model(folder: Path) -> tuple[str, str]:
    """Split '<chip>-<ram>gb-<model-slug>' at the '-<digits>gb-' boundary."""
    match = _MACHINE_RE.match(folder.parent.name)
    if match:
        return match.group("machine"), match.group("model")
    return folder.parent.name, ""


def _combo_parts(folder: Path) -> tuple[str, str, str]:
    parts = folder.name.split("-")
    # track-scorer-dataset; naive_local has scorer 'trie' by convention.
    if parts[0] == "naive" and len(parts) >= 2 and parts[1] == "local":
        # naive_local-trie-<dataset>
        rest = parts[2:]
        scorer = parts[2] if rest and rest[0] in ("trie", "letters") else "trie"
        dataset = "-".join(parts[3:]) if scorer == parts[2] else "-".join(rest)
        return "naive_local", scorer, dataset
    track = parts[0]
    scorer = parts[1] if len(parts) > 1 else ""
    dataset = "-".join(parts[2:]) if len(parts) > 2 else ""
    return track, scorer, dataset


def _balanced_mean(metrics: dict) -> float | None:
    per_field = metrics.get("balanced_accuracy")
    if isinstance(per_field, dict) and per_field:
        values = [v for v in per_field.values() if isinstance(v, int | float)]
        return sum(values) / len(values) if values else None
    if isinstance(per_field, int | float):
        return per_field
    return metrics.get("balanced_accuracy_mean")


def _mean_or_none(value) -> float | None:
    """Mean of a per-field dict metric, or the scalar itself."""
    if isinstance(value, dict) and value:
        values = [v for v in value.values() if isinstance(v, int | float)]
        return sum(values) / len(values) if values else None
    if isinstance(value, int | float):
        return value
    return None


def _run_extras(folder: Path) -> tuple[float | None, int | None, int | None]:
    """(median call-level latency ms, n_cases, n_calls) from the combo's
    timing.json + run.json.

    B11-naive: the 'p50 latency' column must be the CALL-level
    per_item_end_to_end_ms median from timing.json for every track —
    not the per-line latency_ms (which is shared across a call's fields
    on the parallel track and inflated by rotations). Falls back to
    nothing (returns None) if timing.json is absent.
    """
    run_path = folder / "run.json"
    n_cases = None
    if run_path.is_file():
        run = json.loads(run_path.read_text(encoding="utf-8"))
        n_cases = (run.get("counts") or {}).get("cases")
    latency = None
    n_calls = None
    timing_path = folder / "timing.json"
    if timing_path.is_file():
        timing = json.loads(timing_path.read_text(encoding="utf-8"))
        n_calls = timing.get("calls")
        median = timing.get("median") or {}
        latency = median.get("per_item_end_to_end_ms")
    return latency, n_cases, n_calls


def _model_parity_note(model_dir: Path) -> str | None:
    """W4-B: the parity-failure note for a model folder, or None when the
    folder carries a PASSING parity.json.

    A MISSING parity.json also gates: a bench-produced folder ALWAYS has
    one (the writer runs right after the engine load), so its absence
    means the check never ran or crashed — the model has not earned
    accuracy numbers. (Folders with no parity.json and no report rows
    simply produce no rows, so there is nothing to gate.)"""
    parity_path = model_dir / "parity.json"
    if not parity_path.is_file():
        return "parity_failed: parity.json missing (scoring parity not recorded)"
    try:
        parity = json.loads(parity_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "parity_failed: parity.json unreadable"
    if parity.get("passed"):
        return None
    # Results contract v2: name WHICH stage failed. Stages: winners (final
    # decisions), final log-score drift, raw pre-rescore row-logit drift
    # (finding 42 — a batch-1 rescore can mask raw batch drift).
    from benchmarks.check_results import _parity_failed_stages

    stages = "; ".join(_parity_failed_stages(parity))
    drift = parity.get("max_abs_drift_nats", parity.get("max_drift_nats"))
    raw = parity.get("max_raw_row_drift_nats")
    return (
        f"parity_failed: {stages} "
        f"max_abs_drift_nats={drift} "
        f"max_raw_row_drift_nats={raw} "
        f"atol={parity.get('atol')} winners_identical={parity.get('winners_identical')}"
    )


def summarize(out: Path, parity_note: str | None = None) -> Path:
    """Write ``out/SUMMARY.md`` from every report.json under ``out``; print it.

    W4-B: a model folder whose ``parity.json`` shows a failed scoring-parity
    check (or is missing) marks every row of that model ``parity_failed`` —
    the model cannot enter the README compat table (PR #29's gate reads the
    same file; the summary shows WHY). ``parity_note`` (run_bench's
    in-process failure message) is the fallback when the check crashed
    before writing the file."""
    out = Path(out)
    folders = sorted(p for p in out.iterdir() if p.is_dir()) if out.exists() else []
    rows = []
    for machine_dir in folders:
        # Two layouts: <out>/<machine-model>/<combo>/ or <out>/<combo>/.
        combo_dirs = (
            [p for p in machine_dir.iterdir() if p.is_dir()]
            if any(p.is_dir() for p in machine_dir.iterdir())
            else []
        )
        if combo_dirs:
            # W4-B parity gate: a parity-failing model's rows all show it.
            model_parity = _model_parity_note(machine_dir) or parity_note
            for combo in combo_dirs:
                row = _row_from_folder(combo)
                if row:
                    if model_parity and row.get("field_accuracy") is not None:
                        # Gate the accuracy: the note replaces the number.
                        row["accuracy_note"] = model_parity
                        row["field_accuracy"] = None
                    rows.append(row)
        else:
            row = _row_from_folder(machine_dir)
            if row:
                rows.append(row)

    gzipped = enforce_folder_size(out)
    _write_folder_readme(out, gzipped)

    header = (
        "| machine | model | track | scorer | dataset | field acc | majority baseline "
        "| case exact | exact record | bal acc mean | ECE | any-flip | perturb-flip "
        "| p50 latency (ms) | calls | n_cases |"
    )
    lines = [
        "# Bench summary",
        "",
        header,
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    columns = [
        "machine",
        "model",
        "track",
        "scorer",
        "dataset",
        "field_accuracy",
        "majority_baseline",
        "case_exact",
        "exact_record",
        "balanced_accuracy_mean",
        "ece",
        "any_flip_rate",
        "perturbation_flip_rate",
        "latency_ms_p50",
        "n_calls",
        "n_cases",
    ]
    for row in rows:
        cells = []
        for col in columns:
            value = row.get(col)
            if col == "field_accuracy" and value is None and row.get("accuracy_note"):
                # Failure row: the error text replaces the accuracy number.
                cells.append(row["accuracy_note"])
            else:
                cells.append(_cell(value))
        lines.append("| " + " | ".join(cells) + " |")
    if not rows:
        lines.append(f"(no report.json files found under {out})")

    summary_path = out / "SUMMARY.md"
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwrote {summary_path}")
    return summary_path


def _cell(value) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        if 0 < abs(value) < 0.001 or abs(value) >= 100000:
            return f"{value:.2e}"
        return f"{value:.4f}".rstrip("0").rstrip(".") or "0"
    return str(value)


def _write_folder_readme(out: Path, gzipped: bool) -> None:
    lines = [
        "# Bench results folder",
        "",
        "Produced by `jevmlx bench`. One subfolder per (machine-model, track,",
        "scorer, dataset) combination, each holding `predictions.jsonl`,",
        "`run.json`, `report.json`, and `report.md`; dataset lock files sit at",
        "the top level. `SUMMARY.md` is the one-glance table.",
        "",
        "Commit this folder and nothing else (no caches, no model weights).",
    ]
    if gzipped:
        lines += [
            "",
            "Predictions were gzipped (`predictions.jsonl.gz`) to keep the folder",
            "under the 5 MB commit limit; decompress with `gunzip` to inspect.",
        ]
    (out / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m benchmarks.summarize_results",
        description="Summarize every report.json under a bench results folder.",
    )
    parser.add_argument("out", nargs="?", default="benchmarks/results")
    args = parser.parse_args(argv)
    summarize(Path(args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
