"""One-command M5 runbook: the full milestone-gate sequence in order.

Each step is a subprocess logged with wall time and exit status into
``<out>/RUNBOOK.md``:

 1. ``jevmlx doctor --json`` — gate: any FAIL aborts the runbook.
 2. ``pytest -m slow -q`` once per parity model (default Qwen3-8B,
    Llama-3.1-8B, Gemma-3-12B; the model rides the ``MODEL_ID`` env var).
 3. ``jevmlx bench --model quality``.
 4. ``benchmarks.invariance`` on quality with ``--extra 1,5,20,40`` over the
    TypeSafe cases (fetched first when missing).
 5. ``benchmarks.timing --model quality --reps 5``.
 6. ``jevmlx bench --models-file`` for the remaining parity models.
 7. With ``--ab-branch``: a temp worktree of that branch, its own venv, and
    steps 3+4 rerun there (A/B against main). The worktree is removed after.
 8. ``<out>/SUMMARY.md`` comparing main vs A/B: agreement, flip rate, drift,
    time per case, peak memory — from the produced json files only.

Idempotent: a step whose outputs already exist is skipped (``--fresh``
ignores the markers and reruns). The step planner and the summary builder
are pure functions tested with fakes; no model ever loads in a unit test.

Usage:
    python -m benchmarks.m5 --out m5-2026-09-18 [--models-file models.txt]
        [--parity-models id1,id2] [--ab-branch w2a-field-local] [--fresh]
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

__all__ = [
    "DEFAULT_PARITY_MODELS",
    "Step",
    "build_summary_text",
    "combo_row",
    "invariance_rollup",
    "main",
    "plan_steps",
    "read_models_file",
]

REPO_ROOT = Path(__file__).resolve().parent.parent
BENCH_CACHE = Path.home() / ".cache" / "jevmlx" / "bench"
TYPESAFE_JSONL = BENCH_CACHE / "typesafe.jsonl"
QUALITY_ALIAS = "quality"
DEFAULT_PARITY_MODELS = (
    "mlx-community/Qwen3-8B-4bit",
    "mlx-community/Llama-3.1-8B-Instruct-4bit",
    "mlx-community/gemma-3-12b-it-4bit",
)


@dataclass
class Step:
    """One runbook step: a subprocess (or the in-process summary), its idempotency
    outputs, and logging metadata. ``extra_argv`` runs after ``argv`` into the
    same log (multi-command setup steps); every command must exit 0."""

    id: str
    title: str
    argv: tuple[str, ...]
    outputs: tuple[Path, ...]
    env: dict[str, str] = field(default_factory=dict)
    cwd: Path = REPO_ROOT
    gate: bool = False  # doctor: a failure aborts the remaining steps
    capture: bool = False  # capture stdout (small json) instead of streaming
    capture_path: Path | None = None  # where captured stdout is also written
    pre_argv: tuple[str, ...] = ()  # run before argv when pre_target is missing
    pre_target: Path | None = None
    extra_argv: tuple[tuple[str, ...], ...] = ()
    in_process: bool = False  # the summary step: rendered from artifacts


def _venv_bin(name: str) -> str:
    """Absolute path to a console script next to the running interpreter.

    uv venvs put console scripts in ``.venv/bin`` while ``sys.executable``
    may resolve to the managed CPython that the venv symlinks to, so walk
    the interpreter's directory first, then its parent's ``bin/`` (the
    venv root seen through the symlink), then fall back to PATH.
    """
    seen: list[Path] = []
    for base in (Path(sys.executable).resolve().parent, Path(sys.executable).parent):
        if base in seen:
            continue
        seen.append(base)
        candidate = base / name
        if candidate.exists():
            return str(candidate)
    return name


def _slug(model_id: str) -> str:
    from jevmlx.bench import model_slug

    return model_slug(model_id)


def read_models_file(path: str | Path) -> list[str]:
    """One model id per line, '#' comments and blank lines allowed."""
    models: list[str] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            models.append(line)
    return models


def _resolve_ab_ref(branch: str) -> str:
    """Resolve an A/B branch to a ref ``git worktree add`` can check out.

    The M5 clone frequently has the A/B branch only as a remote-tracking ref
    (``origin/<branch>``), not a local branch. Pass a bare local name (e.g.
    ``w2a-field-local``) and this returns whichever of ``<branch>`` or
    ``origin/<branch>`` resolves via ``git rev-parse --verify``. Falls back
    to the bare name if neither resolves (so ``git worktree add`` emits the
    real error — a missing ref, not our guess).
    """
    for candidate in (branch, f"origin/{branch}"):
        proc = subprocess.run(
            ["git", "rev-parse", "--verify", candidate],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        if proc.returncode == 0:
            return candidate
    return branch


def plan_steps(
    out: Path,
    *,
    parity_models: list[str],
    extra: str = "1,5,20,40",
    reps: int = 5,
    ab_branch: str | None = None,
    typesafe_data: Path = TYPESAFE_JSONL,
    quality: str = QUALITY_ALIAS,
    probe: bool = False,
) -> list[Step]:
    """The ordered step list. Pure apart from writing ``models-rest.txt``.

    ``parity_models`` drives steps 2 (pytest per model) and 6 (bench the
    remaining ones, i.e. every model except the quality alias's target).
    A/B steps appear only when ``ab_branch`` is given.
    """
    from jevmlx.models import resolve_model

    out = Path(out)
    steps: list[Step] = []
    steps.append(
        Step(
            id="doctor",
            title="doctor (gate)",
            argv=(_venv_bin("jevmlx"), "doctor", "--json"),
            outputs=(out / "doctor.done", out / "doctor.json"),
            gate=True,
            capture=True,
            capture_path=out / "doctor.json",
        )
    )
    for model in parity_models:
        steps.append(
            Step(
                id=f"parity-{_slug(model)}",
                title=f"pytest -m slow ({model})",
                argv=(_venv_bin("pytest"), "-m", "slow", "-q"),
                outputs=(out / f"parity-{_slug(model)}.done",),
                env={"MODEL_ID": model},
            )
        )
    steps.append(
        Step(
            id="bench-quality",
            title="jevmlx bench --model quality",
            argv=(
                _venv_bin("jevmlx"),
                "bench",
                "--model",
                quality,
                "--out",
                str(out / "bench-quality"),
            ),
            outputs=(out / "bench-quality.done",),
        )
    )
    steps.append(
        Step(
            id="invariance",
            title="invariance on quality (TypeSafe cases, extra 1/5/20/40)",
            argv=(
                sys.executable,
                "-m",
                "benchmarks.invariance",
                "--model",
                quality,
                "--data",
                str(typesafe_data),
                "--out",
                str(out / "invariance"),
                "--extra",
                extra,
            ),
            outputs=(out / "invariance.done", out / "invariance" / "invariance.json"),
            pre_argv=(
                sys.executable,
                "-m",
                "benchmarks.typesafe.fetch",
                "--out",
                str(typesafe_data),
            ),
            pre_target=typesafe_data,
        )
    )
    steps.append(
        Step(
            id="timing",
            title=f"timing on quality ({reps} reps)",
            argv=(
                sys.executable,
                "-m",
                "benchmarks.timing",
                "--model",
                quality,
                "--reps",
                str(reps),
                "--out",
                str(out),
            ),
            outputs=(
                out / "timing.done",
                out / f"timing-{resolve_model(quality).replace('/', '_')}.json",
            ),
        )
    )
    rest = [m for m in parity_models if m != resolve_model(quality)]
    if rest:
        rest_file = out / "models-rest.txt"
        rest_file.write_text("\n".join(rest) + "\n", encoding="utf-8")
        steps.append(
            Step(
                id="bench-rest",
                title="bench the remaining models",
                argv=(
                    _venv_bin("jevmlx"),
                    "bench",
                    "--models-file",
                    str(rest_file),
                    "--out",
                    str(out / "bench-rest"),
                ),
                outputs=(out / "bench-rest.done",),
            )
        )
    if ab_branch is not None:
        worktree = out / "ab-worktree"
        venv_python = worktree / ".venv" / "bin" / "python"
        # B2: resolve the A/B branch ref. The M5 clone often has the branch
        # only as a remote-tracking ref (origin/<branch>), not a local
        # branch. Try the local branch first, then origin/<branch>; the
        # resolved ref is what ``git worktree add`` gets. One resolver,
        # no dual code paths.
        ab_ref = _resolve_ab_ref(ab_branch)
        steps.append(
            Step(
                id="ab-setup",
                title=f"A/B setup: worktree + venv for {ab_branch}",
                argv=("git", "worktree", "add", "--detach", str(worktree), ab_ref),
                extra_argv=(
                    (
                        shutil.which("uv") or "uv",
                        "venv",
                        "--python-preference",
                        "only-managed",
                        "--python",
                        "3.12",
                        str(worktree / ".venv"),
                    ),
                    (
                        shutil.which("uv") or "uv",
                        "pip",
                        "install",
                        "--python",
                        str(venv_python),
                        "-e",
                        ".[dev]",
                    ),
                ),
                outputs=(out / "ab-setup.done",),
                cwd=REPO_ROOT,
            )
        )
        steps.append(
            Step(
                id="ab-bench",
                title=f"A/B bench quality on {ab_branch}",
                argv=(
                    str(worktree / ".venv" / "bin" / "jevmlx"),
                    "bench",
                    "--model",
                    quality,
                    "--out",
                    str(out / "ab" / "bench-quality"),
                ),
                outputs=(out / "ab-bench.done",),
                cwd=worktree,
            )
        )
        steps.append(
            Step(
                id="ab-invariance",
                title=f"A/B invariance on quality ({extra}) on {ab_branch}",
                argv=(
                    str(venv_python),
                    "-m",
                    "benchmarks.invariance",
                    "--model",
                    quality,
                    "--data",
                    str(typesafe_data),
                    "--out",
                    str(out / "ab" / "invariance"),
                    "--extra",
                    extra,
                ),
                outputs=(out / "ab-invariance.done", out / "ab" / "invariance" / "invariance.json"),
                cwd=worktree,
            )
        )
    steps.append(
        Step(
            id="summary",
            title="SUMMARY.md (main vs A/B)",
            argv=(),
            outputs=(out / "SUMMARY.md",),
            in_process=True,
        )
    )
    if probe:
        # W6-2 prep (optional, non-gating): memory slope probe + adapter
        # parity probe on the quality model. Runs AFTER the core runbook;
        # outputs are standalone JSON.
        steps.append(
            Step(
                id="probe-slope",
                title="benchmarks.probe --command slope",
                argv=(
                    sys.executable,
                    "-m",
                    "benchmarks.probe",
                    "--model",
                    quality,
                    "--command",
                    "slope",
                    "--out",
                    str(out / "probe-slope.json"),
                ),
                outputs=(out / "probe-slope.json",),
            )
        )
        steps.append(
            Step(
                id="probe-adapters",
                title="benchmarks.probe --command adapters",
                argv=(
                    sys.executable,
                    "-m",
                    "benchmarks.probe",
                    "--model",
                    quality,
                    "--command",
                    "adapters",
                    "--out",
                    str(out / "probe-adapters.json"),
                ),
                outputs=(out / "probe-adapters.json",),
            )
        )
    return steps


def step_done(step: Step, fresh: bool) -> bool:
    """Idempotency rule: skip when every output exists, unless --fresh."""
    return not fresh and all(p.exists() for p in step.outputs)


def _step_env(step: Step) -> dict[str, str]:
    return os.environ | step.env


def _now() -> str:
    return datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def runbook_append(
    out: Path, index: int, step: Step, *, rc: int | None, secs: float | None, skipped: bool = False
) -> None:
    """One RUNBOOK.md section per step, appended as the step completes."""
    lines = [f"## {index}. {step.title}"]
    if skipped:
        lines[0] += " — skipped: A/B setup failed"
    elif rc is None:
        lines[0] += " — skip (outputs exist)"
    else:
        lines[0] += f" — exit {rc} — {secs:.1f}s — {_now()}"
        lines.append(f"cmd: {' '.join(step.argv)}")
        if step.env:
            lines.append(f"env: MODEL_ID={step.env['MODEL_ID']}")
        lines.append(f"log: {step.id}.log")
    lines.append("")
    with (out / "RUNBOOK.md").open("a", encoding="utf-8") as f:
        f.write("\n".join(lines))


def execute_step(step: Step) -> int:
    """Run one step; returns the exit code. Streams into <step.id>.log.

    B3: a step whose declared cwd is missing is recorded as failed
    (exit -1) and never raised — the runbook continues.
    """
    if not step.in_process and not step.cwd.exists():
        log_path = step.outputs[0].parent / f"{step.id}.log"
        log_path.write_text(f"FAILED: cwd does not exist: {step.cwd}\n", encoding="utf-8")
        return -1
    log_path = step.outputs[0].parent / f"{step.id}.log"
    with log_path.open("w", encoding="utf-8") as log:
        if step.pre_argv and (step.pre_target is None or not step.pre_target.exists()):
            log.write(f"$ {' '.join(step.pre_argv)}\n")
            log.flush()
            pre = subprocess.run(
                step.pre_argv,
                cwd=step.cwd,
                env=_step_env(step),
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            if pre.returncode != 0:
                return pre.returncode
        if step.in_process:
            # B3(b): if A/B setup failed, the summary shows 'A/B: not run'
            # instead of an all-dashes A/B table.
            ab_failed_marker = step.outputs[0].parent / "ab-failed.txt"
            if ab_failed_marker.exists():
                ab_block = None
            else:
                ab_block = collect_side(step.outputs[0].parent / "ab")
            text = build_summary_text(
                collect_side(step.outputs[0].parent),
                ab_block,
                parity_models=_meta_parity_models(step.outputs[0].parent),
                ab_failed=ab_failed_marker.exists(),
            )
            log.write(text)
            (step.outputs[0].parent / "SUMMARY.md").write_text(text, encoding="utf-8")
            return 0
        if step.capture:
            proc = subprocess.run(
                step.argv, cwd=step.cwd, env=_step_env(step), capture_output=True, text=True
            )
            log.write(proc.stdout)
            log.write(proc.stderr)
            if step.capture_path is not None and proc.returncode == 0:
                step.capture_path.write_text(proc.stdout, encoding="utf-8")
            return proc.returncode
        log.write(f"$ {' '.join(step.argv)}\n")
        log.flush()
        proc = subprocess.run(
            step.argv, cwd=step.cwd, env=_step_env(step), stdout=log, stderr=subprocess.STDOUT
        )
        rc = proc.returncode
        for extra in step.extra_argv:
            if rc != 0:
                break
            log.write(f"$ {' '.join(extra)}\n")
            log.flush()
            rc = subprocess.run(
                extra, cwd=step.cwd, env=_step_env(step), stdout=log, stderr=subprocess.STDOUT
            ).returncode
        return rc


def _meta_parity_models(out: Path) -> list[str]:
    """Parity models recorded by the planner side effect (models-rest.txt holds
    only the rest, so read the RUNBOOK header instead)."""
    for line in (out / "RUNBOOK.md").read_text(encoding="utf-8").splitlines():
        if line.startswith("parity models:"):
            return [m.strip() for m in line.split(":", 1)[1].split(",") if m.strip()]
    return []


# ------------------------------------------------------------- summary build


def _load_json(path: Path) -> dict | None:
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def combo_row(combo_name: str, report: dict | None, timing: dict | None) -> dict:
    """One bench combo -> summary row (agreement-or-accuracy, total_ms, peak)."""
    metrics = (report or {}).get("metrics") or {}
    median = (timing or {}).get("median") or {}
    peak = median.get("peak_active_bytes")
    return {
        "combo": combo_name,
        "agreement": metrics.get("agreement"),
        "accuracy": metrics.get("accuracy"),
        "total_ms": median.get("total_ms"),
        "peak_gb": round(peak / 2**30, 2) if isinstance(peak, (int, float)) else None,
    }


def invariance_rollup(invariance: dict | None) -> dict:
    """invariance.json -> mean flip rate, mean |log-odds drift|, per-field @40.

    Rung keys are ints in-process but strings after the JSON round-trip;
    both are accepted.
    """
    targets = (invariance or {}).get("targets") or []
    flips: list[float] = []
    drifts: list[float] = []
    fields: dict[str, dict] = {}
    for row in targets:
        rungs = {str(k): v for k, v in (row.get("rungs") or {}).items()}
        for entry in rungs.values():
            if isinstance(entry.get("flip_rate"), (int, float)):
                flips.append(entry["flip_rate"])
            drift = entry.get("winner_logodds_drift_mean")
            if isinstance(drift, (int, float)) and math.isfinite(drift):
                drifts.append(drift)
        rung40 = rungs.get("40") or {}
        fields[row.get("field", "?")] = {
            "flip40": rung40.get("flip_rate"),
            "drift40": rung40.get("winner_logodds_drift_mean"),
        }
    return {
        "mean_flip_rate": sum(flips) / len(flips) if flips else None,
        "mean_drift": sum(drifts) / len(drifts) if drifts else None,
        "fields": fields,
    }


def collect_side(side_dir: Path) -> dict:
    """Bench + invariance artifacts for one side (main or ab) -> summary block."""
    side_dir = Path(side_dir)
    combos = [
        combo_row(
            report_path.parent.name,
            _load_json(report_path),
            _load_json(report_path.parent / "timing.json"),
        )
        for report_path in sorted(side_dir.glob("bench-quality/**/report.json"))
    ]
    block: dict = {
        "combos": combos,
        "invariance": invariance_rollup(_load_json(side_dir / "invariance" / "invariance.json")),
    }
    # P4/I7: parity status word (PASS / DRIFT / FAIL) from parity.json.
    parity_path = side_dir / "parity.json"
    if parity_path.exists():
        parity = _load_json(parity_path) or {}
        block["parity_status"] = parity.get(
            "status", "FAIL" if not parity.get("passed") else "PASS"
        )
        block["parity_passed"] = parity.get("passed", False)
    timing_reports = sorted(side_dir.glob("timing-*.json"))
    if timing_reports:
        report = _load_json(timing_reports[0]) or {}
        presets = report.get("presets") or {}
        timing_block: dict = {}
        for preset_id, preset in sorted(presets.items()):
            agg = preset.get("aggregate") or {}
            total = (agg.get("total_ms") or {}).get("median")
            peak = (agg.get("peak_active_bytes") or {}).get("median")
            timing_block[preset_id] = {
                "total_ms_median": total,
                "peak_gb": round(peak / 2**30, 2) if isinstance(peak, (int, float)) else None,
            }
        if timing_block:
            block["timing_report"] = timing_block
    return block


def _fmt(value: float | int | None, pct: bool = False) -> str:
    if value is None:
        return "-"
    # Dict-safe: a metrics field that is unexpectedly a dict (e.g. typesafe's
    # agreement {overall, by_workflow, ...}) should never reach here, but if
    # it does, extract 'overall' or show '-' rather than crash on __format__.
    if isinstance(value, dict):
        value = value.get("overall")
        if value is None:
            return "-"
    return f"{value:.1%}" if pct else f"{value:.4g}"


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _combo_agreement(row: dict) -> float | None:
    """Extract a float agreement/accuracy from a row's metrics.

    Typesafe report.json stores ``agreement`` as a dict
    (``{overall, by_workflow, ...}``); bundled/other report.json stores a
    bare float. This always returns a float: the dict's ``overall`` key when
    ``agreement`` is a dict, the float when it is a number, else falls back
    to ``accuracy``. Never returns a dict (``_fmt`` would crash on it).
    """
    agreement = row.get("agreement")
    if isinstance(agreement, dict):
        agreement = agreement.get("overall")
    if agreement is not None:
        return agreement
    return row.get("accuracy")


def build_summary_text(
    main: dict, ab: dict | None, *, parity_models: list[str], ab_failed: bool = False
) -> str:
    """SUMMARY.md: main table, timing report, invariance rollup, A/B deltas.

    Pure: takes the collected blocks (see collect_side), never touches disk.
    """
    lines = [
        "# M5 runbook summary",
        "",
        f"- parity models: {', '.join(parity_models) or '-'}",
        f"- parity status: {main.get('parity_status', '-')}",
        "",
        "## Main — bench combos",
        "",
        "| combo | agreement/accuracy | total_ms | peak_gb |",
        "|---|---|---|---|",
    ]
    for row in main.get("combos") or []:
        lines.append(
            f"| {row['combo']} | {_fmt(_combo_agreement(row), pct=True)} "
            f"| {_fmt(row['total_ms'])} | {_fmt(row['peak_gb'])} |"
        )
    timing_report = main.get("timing_report") or {}
    if timing_report:
        lines += [
            "",
            "## Main — timing report (quality, decide() presets)",
            "",
            "| preset | total_ms median | peak_gb |",
            "|---|---|---|",
        ]
        for preset, entry in timing_report.items():
            lines.append(
                f"| {preset} | {_fmt(entry.get('total_ms_median'))} "
                f"| {_fmt(entry.get('peak_gb'))} |"
            )
    inv = main.get("invariance") or {}
    if inv.get("fields"):
        lines += [
            "",
            "## Main — invariance (TypeSafe, extra 1/5/20/40; @40 vs extra=1 baseline)",
            "",
            f"- mean flip rate: {_fmt(inv.get('mean_flip_rate'), pct=True)}",
            f"- mean |winner log-odds drift|: {_fmt(inv.get('mean_drift'))}",
            "",
            "| field | flip@40 | drift@40 |",
            "|---|---|---|",
        ]
        for field, entry in inv["fields"].items():
            lines.append(
                f"| {field} | {_fmt(entry.get('flip40'), pct=True)} "
                f"| {_fmt(entry.get('drift40'))} |"
            )
    if ab is not None:

        def _agg(block: dict) -> dict:
            rows = block.get("combos") or []
            agreements = [v for r in rows if (v := _combo_agreement(r)) is not None]
            total_ms = [v for r in rows if (v := r["total_ms"]) is not None]
            peaks = [v for r in rows if (v := r["peak_gb"]) is not None]
            block_inv = block.get("invariance") or {}
            return {
                "agreement": _mean(agreements),
                "flip_rate": block_inv.get("mean_flip_rate"),
                "drift": block_inv.get("mean_drift"),
                "time_per_case_ms": _mean(total_ms),
                "peak_gb": max(peaks) if peaks else None,
            }

        m, a = _agg(main), _agg(ab)
        labels = [
            ("agreement", "agreement/accuracy (mean)", True),
            ("flip_rate", "flip rate (invariance mean)", True),
            ("drift", "|log-odds drift| (mean)", False),
            ("time_per_case_ms", "time per case (ms)", False),
            ("peak_gb", "peak memory (GB, max)", False),
        ]
        lines += [
            "",
            "## A/B comparison — main vs A/B (bench quality + invariance)",
            "",
            "| metric | main | A/B | delta (A/B - main) |",
            "|---|---|---|---|",
        ]
        for key, label, pct in labels:
            mv, av = m[key], a[key]
            delta = None if mv is None or av is None else av - mv
            lines.append(
                f"| {label} | {_fmt(mv, pct=pct)} | {_fmt(av, pct=pct)} | {_fmt(delta, pct=pct)} |"
            )
    else:
        lines += [
            "",
            "## A/B comparison",
            "",
            "A/B: not run (setup failed)." if ab_failed else "A/B not run (no --ab-branch).",
        ]
    lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------- main


def _maybe_caffeinate(allow_sleep: bool, argv_tail: list[str]) -> None:
    """Block macOS idle sleep for the M5 run (W5c-15).

    During the M5 7B smoke macOS idle-slept mid-run; Metal parks, the process
    stays alive, wall time becomes a lie. Re-exec under ``caffeinate -dimsu``
    (display idle / system idle / disk idle / user assertion) exactly once,
    then set a guard env var so the re-execed child does not re-exec. Opt out
    with ``--allow-sleep``. No-op on non-darwin. Best-effort: a missing
    caffeinate prints a warning, POPS the guard env var (so the RUNBOOK
    header truthfully reports ``sleep_blocked: False``), and continues.
    Returns None.
    """
    if sys.platform != "darwin":
        return
    if allow_sleep:
        print("[sleep] idle sleep NOT blocked (--allow-sleep)", flush=True)
        return
    if os.environ.get("JEVMLX_M5_CAFFEINATED") == "1":
        return
    os.environ["JEVMLX_M5_CAFFEINATED"] = "1"
    print(
        "[sleep] blocking macOS idle sleep via caffeinate -dimsu (opt out: --allow-sleep)",
        flush=True,
    )
    try:
        os.execvp(
            "caffeinate",
            ["caffeinate", "-dimsu", sys.executable, "-m", "benchmarks.m5", *argv_tail],
        )
    except FileNotFoundError:
        # Pop the guard so the RUNBOOK header reports sleep_blocked: False —
        # the re-exec never happened, so sleep is NOT blocked.
        os.environ.pop("JEVMLX_M5_CAFFEINATED", None)
        print("[sleep] caffeinate not found; idle sleep NOT blocked", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m benchmarks.m5",
        description=(
            "One-command M5 runbook (W5b-4): doctor, slow parity suite, bench,"
            " invariance, timing, A/B, summary."
        ),
    )
    parser.add_argument("--out", required=True, help="runbook output directory, e.g. m5-2026-09-18")
    parser.add_argument(
        "--parity-models",
        default=None,
        help="comma-separated parity model ids (default: Qwen3-8B, Llama-3.1-8B, Gemma-3-12B)",
    )
    parser.add_argument(
        "--models-file",
        default=None,
        help="file with one parity model id per line ('#' comments allowed);"
        " overrides --parity-models",
    )
    parser.add_argument(
        "--ab-branch", default=None, help="git branch/ref to bench + invariance as A/B"
    )
    parser.add_argument("--reps", type=int, default=5, help="timing reps (step 5)")
    parser.add_argument(
        "--extra", default="1,5,20,40", help="invariance extra-field ladder (step 4)"
    )
    parser.add_argument(
        "--fresh", action="store_true", help="rerun steps whose outputs already exist"
    )
    parser.add_argument(
        "--probe",
        action="store_true",
        help="append the W6-2 prep probes (memory slope + adapter parity) "
        "as optional steps (python -m benchmarks.probe)",
    )
    parser.add_argument(
        "--allow-sleep",
        action="store_true",
        help="do NOT block macOS idle sleep via caffeinate (default: block it)",
    )
    args = parser.parse_args(argv)

    # W5c-15: block macOS idle sleep (caffeinate re-exec). See
    # _maybe_caffeinate for the rationale and the env-var guard.
    _maybe_caffeinate(args.allow_sleep, list(sys.argv[1:]))

    if args.parity_models:
        parity_models = [m.strip() for m in args.parity_models.split(",") if m.strip()]
    elif args.models_file:
        parity_models = read_models_file(args.models_file)
    else:
        parity_models = list(DEFAULT_PARITY_MODELS)

    out = Path(args.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    steps = plan_steps(
        out,
        parity_models=parity_models,
        extra=args.extra,
        reps=args.reps,
        ab_branch=args.ab_branch,
        probe=args.probe,
    )

    if not (out / "RUNBOOK.md").exists():
        header = [
            f"# M5 runbook — {_now()}",
            f"cmd: python -m benchmarks.m5 --out {args.out}"
            + (f" --ab-branch {args.ab_branch}" if args.ab_branch else ""),
            f"parity models: {', '.join(parity_models)}",
            f"sleep_blocked: {os.environ.get('JEVMLX_M5_CAFFEINATED') == '1'}",
            "",
        ]
        (out / "RUNBOOK.md").write_text("\n".join(header), encoding="utf-8")

    failures: list[str] = []
    worktree = out / "ab-worktree"
    # B3: when the A/B setup step fails, every later A/B step is SKIPPED
    # with a RUNBOOK line. The summary step still runs for the main side
    # with an 'A/B: not run' note.
    ab_setup_failed = False
    try:
        for index, step in enumerate(steps, 1):
            if step_done(step, args.fresh):
                runbook_append(out, index, step, rc=None, secs=None)
                continue
            # B3(b): skip later A/B steps when ab-setup failed.
            if ab_setup_failed and step.id.startswith("ab-"):
                runbook_append(out, index, step, rc=None, secs=None, skipped=True)
                continue
            secs = time.perf_counter()
            # B3(c): catch any unexpected exception inside a step, record
            # its type+message, and continue to the next non-dependent step.
            try:
                rc = execute_step(step)
            except Exception as exc:  # noqa: BLE001
                rc = -1
                log_path = step.outputs[0].parent / f"{step.id}.log"
                log_path.write_text(f"EXCEPTION ({type(exc).__name__}): {exc}\n", encoding="utf-8")
            secs = time.perf_counter() - secs
            runbook_append(out, index, step, rc=rc, secs=secs)
            if rc == 0:
                for marker in step.outputs:
                    if marker.name == f"{step.id}.done":
                        marker.touch()
            else:
                failures.append(step.id)
                # B3(b): mark A/B setup as failed so later ab-* steps skip.
                if step.id == "ab-setup":
                    ab_setup_failed = True
                    (out / "ab-failed.txt").write_text(
                        f"A/B setup failed at {_now()}\n", encoding="utf-8"
                    )
                if step.gate:
                    with (out / "RUNBOOK.md").open("a", encoding="utf-8") as f:
                        f.write(
                            f"**ABORT: {step.id} failed (exit {rc}); remaining steps skipped.**\n\n"
                        )
                    return 1
    finally:
        if worktree.exists() and shutil.which("git"):
            subprocess.run(
                ["git", "worktree", "remove", "--force", str(worktree)],
                cwd=REPO_ROOT,
                capture_output=True,
            )
            shutil.rmtree(worktree, ignore_errors=True)

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
