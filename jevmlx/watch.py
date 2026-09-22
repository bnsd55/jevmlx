"""W6-UI: read-only live dashboard for a bench/m5 output directory.

Renders a live dashboard from the files the run already writes
(RUNBOOK.md, run.json, heartbeat.jsonl, completed_cases.jsonl,
predictions.jsonl). The run process is never touched — this module only
READS files. A half-written last line in any file is truncated to the last
complete JSON line (same rule as ``jevmlx.resume``); a missing file shows
"—"; a parse error logs once to stderr and never raises.
"""

from __future__ import annotations

import json
import os
import statistics
import sys
import time
from pathlib import Path

from rich.console import Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

__all__ = [
    "read_jsonl_safe",
    "read_jsonl_or_gz",
    "parse_runbook",
    "parse_run_json",
    "parse_heartbeat",
    "count_lines",
    "live_table_row",
    "eta_string",
    "build_dashboard_renderable",
    "build_dashboard",
    "build_questions",
    "render_html",
    "_dashboard_html",
    "_handle_sse",
    "_watched_files",
    "_mtimes_signature",
    "run_watch",
]

# The README leaderboard columns (benchmarks/leaderboard.py:82-92).
LEADERBOARD_COLUMNS = (
    "Model",
    "Source",
    "Scorer",
    "Machine",
    "Accuracy",
    "Customer service",
    "Agent trace",
    "Security",
    "Invoices",
    "Time per case",
    "Cost per case",
    "Cases",
)

_WORKFLOW_KEYS = (
    "customer_service",
    "agent_trace_observability",
    "security_incidents",
    "invoice_processing",
)

_CACHE_RED_GB = 10.0  # cache memory turns red above this


def _stderr_once(msg: str, _seen: set[str] | None = None) -> None:
    """Log a parse error to stderr once per message (never raise)."""
    seen = _seen if _seen is not None else _STDERR_SEEN
    if msg in seen:
        return
    seen.add(msg)
    print(f"[watch] {msg}", file=sys.stderr, flush=True)


_STDERR_SEEN: set[str] = set()


def read_jsonl_safe(path: Path, *, max_lines: int = 0) -> list[dict]:
    """Read a .jsonl file, truncating to the last complete JSON line.

    A half-written trailing line (the run is mid-write) is silently
    discarded — same rule as ``jevmlx.resume``. A missing file returns [].
    A JSONDecodeError on any line stops reading (later lines are
    unreliable); the error logs once to stderr.
    """
    if not path.exists():
        return []
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return []
    lines = text.splitlines()
    if max_lines:
        lines = lines[-max_lines:]
    out: list[dict] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            # Half-written or corrupt line — stop (later lines unreliable).
            _stderr_once(f"truncated {path.name} at a malformed line")
            break
        if isinstance(obj, dict):
            out.append(obj)
    return out


def read_jsonl_or_gz(path: Path, *, max_lines: int = 0) -> list[dict]:
    """Read a .jsonl or .jsonl.gz file (finished combos are gzipped).

    Tries ``path`` (plain .jsonl) first; if absent, tries ``path + '.gz'``.
    Same truncation / half-line rules as ``read_jsonl_safe``. Used for
    predictions.jsonl (the bench finalizer gzips finished combos) and
    any other artifact that may be gzipped.
    """
    if path.exists():
        return read_jsonl_safe(path, max_lines=max_lines)
    gz = Path(str(path) + ".gz")
    if gz.exists():
        import gzip

        try:
            text = gzip.decompress(gz.read_bytes()).decode("utf-8", errors="replace")
        except (OSError, EOFError):
            return []
        lines = text.splitlines()
        if max_lines:
            lines = lines[-max_lines:]
        out: list[dict] = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                _stderr_once(f"truncated {gz.name} at a malformed line")
                break
            if isinstance(obj, dict):
                out.append(obj)
        return out
    return []


def parse_runbook(out_dir: Path) -> list[dict]:
    """Parse RUNBOOK.md into a list of step dicts: {title, state, wall, cmd}.

    state: "done" (✓), "running" (▶), "skipped" (○), "aborted" (✗).
    """
    path = out_dir / "RUNBOOK.md"
    if not path.exists():
        return []
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return []
    steps: list[dict] = []
    for block in text.split("\n## "):
        block = block.strip()
        if not block or block.startswith("# "):
            continue
        first_line = block.split("\n")[0]
        # "1. Doctor — exit 0 — 3.2s — 2026-..."  or  "1. Doctor — skip ..."
        # or "**ABORT: ..." or "Main — bench combos"
        state = "pending"
        if "— skip" in first_line:
            state = "skipped"
        elif "ABORT" in first_line:
            state = "aborted"
        elif "— exit" in first_line:
            # exit 0 = done, exit != 0 = aborted
            if "exit 0" in first_line:
                state = "done"
            else:
                state = "aborted"
        wall = ""
        if "—" in first_line:
            parts = first_line.split("—")
            for p in parts:
                p = p.strip()
                if p.endswith("s") and p.replace(".", "").replace("s", "").isdigit():
                    wall = p
                    break
        cmd = ""
        for line in block.split("\n"):
            if line.startswith("cmd: "):
                cmd = line[5:].strip()
                break
        steps.append({"title": first_line, "state": state, "wall": wall, "cmd": cmd})
    return steps


def parse_run_json(out_dir: Path) -> dict:
    """Read the combo's run.json (best-effort)."""
    path = out_dir / "run.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def parse_heartbeat(out_dir: Path) -> dict | None:
    """The last heartbeat record from heartbeat.jsonl (None when absent)."""
    records = read_jsonl_safe(out_dir / "heartbeat.jsonl")
    return records[-1] if records else None


def count_lines(path: Path) -> int:
    """Count lines in a file (for dataset total). 0 when absent."""
    if not path.exists():
        return 0
    try:
        return sum(1 for _ in open(path, encoding="utf-8"))
    except OSError:
        return 0


def _fmt_pct(value) -> str:
    if value is None:
        return "—"
    if isinstance(value, (int, float)):
        return f"{value * 100:.1f}%"
    return str(value)


def _fmt_seconds(value) -> str:
    if value is None:
        return "—"
    if isinstance(value, (int, float)):
        if value < 1:
            return f"{value * 1000:.0f}ms"
        return f"{value:.1f}s"
    return str(value)


def _fmt_cost(value) -> str:
    if value is None:
        return "—"
    if isinstance(value, (int, float)):
        return f"${value:.4f}"
    return str(value)


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float] | None:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return None
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def live_table_row(
    records: list[dict],
    *,
    model: str = "—",
    source: str = "—",
    scorer: str = "—",
    machine: str = "—",
) -> dict:
    """Build one leaderboard-equivalent row dict from prediction records.

    Uses jevmlx.evalmetrics for accuracy (field_accuracy) and per-workflow
    accuracy (macro_by workflow). Time per case = median per-case latency.
    Cost per case = the same formula leaderboard.py uses (_fmt_cost on
    cost_per_case_usd, which the evalreport computes from latency × $/hr).
    Cases = len(records).
    """
    from jevmlx.evalmetrics import field_accuracy, macro_by

    cases = len(records)
    acc = field_accuracy(records)
    # Wilson CI.
    if acc is not None and cases:
        correct = sum(1 for r in records if r.get("correct") is True)
        ci = _wilson(correct, cases)
    else:
        ci = None
    # Per-workflow accuracy (the four TypeSafe workflow columns).
    labelled = [r for r in records if r.get("label") is not None and r.get("workflow")]
    wf = macro_by(labelled, "workflow") if labelled else {}
    wf_cells = {key: wf.get(key) for key in _WORKFLOW_KEYS}
    # Time per case: median per-case latency (ms -> s).
    latencies = [
        r.get("latency_ms") for r in records if isinstance(r.get("latency_ms"), (int, float))
    ]
    time_per_case = statistics.median(latencies) / 1000.0 if latencies else None
    # Cost per case: leaderboard uses cost_per_case_usd from the evalreport;
    # the watch dashboard does not recompute the full formula (no model $/hr
    # here) — show — when the report's cost is unavailable.
    cost = None
    return {
        "model": model,
        "source": source,
        "scorer": scorer,
        "machine": machine,
        "accuracy": acc,
        "accuracy_ci": ci,
        "by_workflow": wf_cells,
        "time_per_case_s": time_per_case,
        "cost_per_case_usd": cost,
        "cases": cases,
    }


def eta_string(done: int, total: int, elapsed_s: float) -> str:
    """ETA string from done/total and elapsed time."""
    if total <= 0 or done <= 0 or elapsed_s <= 0:
        return "—"
    rate = done / elapsed_s
    remaining = total - done
    if remaining <= 0:
        return "done"
    eta_s = remaining / rate
    if eta_s < 60:
        return f"{eta_s:.0f}s"
    if eta_s < 3600:
        return f"{eta_s / 60:.0f}m"
    return f"{eta_s / 3600:.1f}h"


def _state_icon(state: str) -> str:
    return {"done": "✓", "running": "▶", "skipped": "○", "aborted": "✗", "pending": "○"}.get(
        state, "○"
    )


def _combo_dirs(out_dir: Path) -> list[Path]:
    """All combo dirs under <out> — discovered by directory, not by run.json.

    The real bench/m5 tree is:
      <out>/bench-quality/<machine>-<model-slug>/<track>-<scorer>-<dataset>/...
      <out>/bench-rest/<machine>-<model-slug>/<track>-<scorer>-<dataset>/...
      <out>/ab/bench-quality/<machine>-quality/<combo>/...
      <out>/invariance/extraNN/...

    A LIVE combo dir has heartbeat.jsonl and predictions.jsonl but NO run.json
    yet (run.json is written on completion). A QUEUED combo dir (never-run)
    has no marker files — we discover it as a sibling of a known combo in the
    same model folder. So we discover combos by ANY of heartbeat.jsonl /
    predictions.jsonl / run.json present, PLUS sibling dirs in model folders
    that already contain at least one combo.
    """
    out_dir = Path(out_dir)
    combos: list[Path] = []
    if not out_dir.exists():
        return combos
    seen: set[Path] = set()
    # The out dir itself can be a single combo.
    if _is_combo_dir(out_dir):
        combos.append(out_dir)
        seen.add(out_dir)
    # rglob for combo-marker files at any depth.
    for marker in ("run.json", "heartbeat.jsonl", "predictions.jsonl"):
        for p in out_dir.rglob(marker):
            combo_dir = p.parent
            if combo_dir in seen or combo_dir == out_dir:
                continue
            # W6-UI round 4b: skip ab-worktree/ (a git worktree, not results)
            # and skip invariance/ (own section, not model rows).
            if "ab-worktree" in combo_dir.parts:
                continue
            seen.add(combo_dir)
            combos.append(combo_dir)
    # W6-UI-3g: discover QUEUED sibling dirs (never-run, no markers) in model
    # folders that already have at least one combo. The model folder must be a
    # proper descendant of <out> so we never walk above out_dir.
    model_dirs: set[Path] = {
        c.parent for c in combos if c.parent != out_dir and out_dir in c.parent.parents
    }
    for model_dir in model_dirs:
        for sibling in sorted(model_dir.iterdir()) if model_dir.is_dir() else []:
            if not sibling.is_dir() or sibling in seen:
                continue
            # A queued combo dir has a name like <track>-<scorer>-<dataset>.
            if "-" in sibling.name and not sibling.name.startswith("."):
                seen.add(sibling)
                combos.append(sibling)
    return combos


def build_dashboard_renderable(out_dir: Path, *, width: int = 120) -> Group:
    """Build the full dashboard renderable for one refresh."""
    out_dir = Path(out_dir)
    panels: list = []

    # --- HEADER ---
    run = parse_run_json(out_dir)
    env = run.get("environment", {}) if run else {}
    cfg = run.get("config", {}) if run else {}
    runbook_steps = parse_runbook(out_dir)
    # Elapsed: from the first heartbeat or run.json timestamp.
    hb = parse_heartbeat(out_dir)
    elapsed_s = 0
    if hb and isinstance(hb.get("elapsed_s"), (int, float)):
        elapsed_s = hb["elapsed_s"]
    header = Table.grid(expand=True)
    header.add_column(justify="left")
    header.add_column(justify="right")
    header.add_row(
        f"[bold]{out_dir}[/bold]",
        f"elapsed {elapsed_s / 60:.1f}m" if elapsed_s else "—",
    )
    machine = env.get("chip") or env.get("machine_model") or "—"
    mlx_ver = env.get("mlx_version") or "—"
    model = cfg.get("model") or run.get("model") or "—"
    freeze = env.get("git_sha") or "—"
    header.add_row(
        f"machine: {machine}  mlx: {mlx_ver}  model: {model}",
        f"freeze: {freeze[:12] if freeze else '—'}",
    )
    panels.append(Panel(header, title="jevmlx watch", border_style="blue"))

    # --- RUNBOOK (m5 steps) ---
    if runbook_steps:
        rb = Table.grid(expand=True)
        rb.add_column(justify="left", ratio=1)
        rb.add_column(justify="right")
        for step in runbook_steps:
            icon = _state_icon(step["state"])
            color = {"done": "green", "aborted": "red", "skipped": "dim", "running": "yellow"}.get(
                step["state"], "white"
            )
            rb.add_row(f"[{color}]{icon}[/{color}] {step['title']}", step.get("wall") or "")
        panels.append(Panel(rb, title="RUNBOOK", border_style="cyan"))

    # --- COMBO (the current/last combo) ---
    combos = _combo_dirs(out_dir)
    if combos:
        combo_dir = combos[-1]
        combo_run = parse_run_json(combo_dir)
        combo_hb = parse_heartbeat(combo_dir)
        # Progress.
        total = count_lines(combo_dir / "dataset.jsonl") or count_lines(out_dir / "dataset.jsonl")
        done = len(read_jsonl_safe(combo_dir / "completed_cases.jsonl"))
        if combo_hb and isinstance(combo_hb.get("cases_done"), int):
            done = max(done, combo_hb["cases_done"])
        done_pct = (done / total * 100) if total else 0
        # Cases/h over the last heartbeat window.
        cases_h = "—"
        if (
            combo_hb
            and isinstance(combo_hb.get("elapsed_s"), (int, float))
            and combo_hb["elapsed_s"] > 0
        ):
            rate = done / (combo_hb["elapsed_s"] / 3600) if done else 0
            cases_h = f"{rate:.0f}/h" if rate else "—"
        eta = eta_string(done, total, combo_hb.get("elapsed_s", 0) if combo_hb else 0)
        # Heartbeat memory.
        mem_str = "—"
        if combo_hb:
            peak = combo_hb.get("peak_memory_bytes", -1)
            active = combo_hb.get("active_memory_bytes", -1)
            cache = combo_hb.get("cache_memory_bytes", -1)

            def _gb(v):
                return f"{v / 2**30:.2f} GB" if v >= 0 else "n/a"

            mem_str = f"peak={_gb(peak)} active={_gb(active)} cache={_gb(cache)}"
            if cache >= 0 and cache / 2**30 >= _CACHE_RED_GB:
                mem_str = f"[red]{mem_str}[/red]"
        combo_table = Table.grid(expand=True)
        combo_table.add_column(justify="left")
        combo_table.add_column(justify="right")
        combo_table.add_row(
            f"[bold]{combo_dir.name}[/bold]  {done}/{total} cases  {done_pct:.0f}%",
            f"{cases_h}  ETA {eta}",
        )
        combo_table.add_row(
            f"last heartbeat: {mem_str}",
            f"cap: {combo_run.get('memory', {}).get('metal_cache_limit_bytes', '—')}",
        )
        panels.append(Panel(combo_table, title="COMBO", border_style="magenta"))

    # --- RESULTS TABLE (README columns) ---
    results = Table(title="Results", expand=True)
    for col in LEADERBOARD_COLUMNS:
        results.add_column(col)
    machine_name = env.get("chip") or "—"
    for combo_dir in combos:
        records = read_jsonl_or_gz(combo_dir / "predictions.jsonl")
        if not records:
            continue
        combo_run = parse_run_json(combo_dir)
        cfg = combo_run.get("config", {}) if combo_run else {}
        row = live_table_row(
            records,
            model=cfg.get("model") or "—",
            source="live",
            scorer=_scorer_name(cfg) or "—",
            machine=machine_name,
        )
        wf = row.get("by_workflow", {})
        ci = row.get("accuracy_ci")
        acc_str = _fmt_pct(row["accuracy"])
        if ci and row["accuracy"] is not None:
            acc_str += f" ±{(ci[1] - ci[0]) / 2 * 100:.1f}%"
        results.add_row(
            row["model"],
            row["source"],
            row["scorer"],
            row["machine"],
            acc_str,
            *[_fmt_pct(wf.get(k)) for k in _WORKFLOW_KEYS],
            _fmt_seconds(row["time_per_case_s"]),
            _fmt_cost(row["cost_per_case_usd"]),
            str(row["cases"]),
        )
    if results.row_count:
        panels.append(results)

    # --- LAST QUESTIONS (tail of predictions.jsonl) ---
    if combos:
        combo_dir = combos[-1]
        tail = read_jsonl_or_gz(combo_dir / "predictions.jsonl", max_lines=8)
        if tail:
            q = Table(title="Last questions", expand=True)
            for col in ("case_id", "field", "pred", "label", "ok", "margin", "ms"):
                q.add_column(col)
            for r in tail:
                ok = "✓" if r.get("correct") else "✗"
                prob = r.get("probability")
                margin = f"{prob:.2f}" if isinstance(prob, (int, float)) else "—"
                q.add_row(
                    str(r.get("case_id", "—"))[:24],
                    str(r.get("field", "—")),
                    str(r.get("prediction", "—"))[:16],
                    str(r.get("label", "—"))[:16],
                    ok,
                    margin,
                    str(r.get("latency_ms", "—")),
                )
            panels.append(q)

    return Group(*panels)


def render_html(out_dir: str | Path, *, refresh: float = 2.0, width: int = 120) -> str:
    """Render the dashboard to a self-contained HTML document.

    Uses rich's ``Console(record=True)`` + ``export_html(inline_styles=True)``
    so the same renderable the terminal sees is exported as HTML. A
    ``<meta http-equiv=refresh>`` tag makes Chrome reload by itself.
    """
    from rich.console import Console

    console = Console(
        record=True,
        width=width,
        force_terminal=True,
        color_system="256",
        file=open(os.devnull, "w"),
    )
    console.print(build_dashboard_renderable(Path(out_dir), width=width))
    body = console.export_html(inline_styles=True)
    # Inject the auto-refresh meta tag (rich's HTML lacks it).
    meta = f'<meta http-equiv="refresh" content="{int(max(1, refresh))}">'
    body = body.replace("<head>", f"<head>{meta}", 1)
    return body


def _read_sleep_blocked(out_dir: Path) -> bool:
    """One source for the sleep-guard flag: the RUNBOOK 'sleep_blocked:' line.

    sleep_blocked=True means caffeinate is running (macOS idle sleep is
    blocked, the guard is ON). Both run.sleep_blocked and the sleep_windows
    health rule read this helper so they can never disagree.
    """
    rb = out_dir / "RUNBOOK.md"
    if not rb.exists():
        return False
    try:
        for line in rb.read_text(encoding="utf-8").splitlines():
            if line.startswith("sleep_blocked:"):
                return "True" in line
    except OSError:
        pass
    return False


def _read_step_log_tail(out_dir: Path, step_id: str, *, n: int = 40) -> str | None:
    """Last n lines of <step.id>.log, or None when the log is absent."""
    log_path = out_dir / f"{step_id}.log"
    if not log_path.exists():
        return None
    try:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return None
    return "\n".join(lines[-n:]) if lines else None


def _extract_step_error(out_dir: Path, step_id: str) -> str | None:
    """The last 'Error|FAILED|Traceback' block from <step.id>.log, or None.

    The failure diagnosis panel shows the error text; m5 tees each step's
    stdout+stderr to <step.id>.log, so the error is there.
    """
    log_path = out_dir / f"{step_id}.log"
    if not log_path.exists():
        return None
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    # Find the last Traceback / FAILED / Error line and grab the block.
    import re

    matches = list(re.finditer(r"(?im)^(Traceback|FAILED|Error|EXCEPTION|assert )", text))
    if not matches:
        return None
    start = matches[-1].start()
    # Grab from that line to the end (capped at 40 lines).
    block = text[start:].splitlines()[:40]
    return "\n".join(block) if block else None


def _parse_pipeline(out_dir: Path) -> dict:
    """Pipeline panel: steps grouped by the LAST attempt.

    Parses RUNBOOK.md for attempt headers of two forms:
    - W6 (#119+): '## attempt <n> <ISO> <hash> <argv>'
    - Legacy (pre-#119): '# M5 runbook — <ISO>' (one per attempt, append-only)
    Both split attempts; attempt_n = count; attempt_started = the ISO.
    Only the last attempt's steps are returned (the dashboard shows the
    current attempt). Steps are deduped by id: last occurrence wins.
    """
    rb = out_dir / "RUNBOOK.md"
    attempt_n = 0
    attempt_started = None
    steps_raw: list[dict] = []
    if rb.exists():
        try:
            text = rb.read_text(encoding="utf-8")
        except OSError:
            text = ""
        # Find all attempt-start lines (both forms).
        import re

        attempt_starts = list(
            re.finditer(r"^(?:## attempt \d+ \S+ \S+|# M5 runbook — \S+)", text, re.MULTILINE)
        )
        if attempt_starts:
            # W6-UI round 4: pick the LIVE attempt — the one that has a
            # 'started <id> <ISO>' line with no matching completion (still
            # running). If none is running, pick the newest by started time
            # (the last block). Previously this took the last block by
            # position, which picked a finished probe/A/B attempt when a
            # later one was still running.
            blocks = []
            for idx, m in enumerate(attempt_starts):
                start = m.start()
                end = (
                    attempt_starts[idx + 1].start() if idx + 1 < len(attempt_starts) else len(text)
                )
                blocks.append(text[start:end])

            # Classify each block: running if it has an unmatched 'started' line.
            def _block_is_running(blk: str) -> bool:
                started_ids = set()
                completed_ids = set()
                for line in blk.splitlines()[1:]:
                    if line.startswith("started "):
                        sp = line.split(None, 2)
                        if len(sp) >= 3:
                            started_ids.add(sp[1])
                    elif line.startswith("## ") and not line.startswith("## attempt "):
                        sp = line.split(" — ", 1)
                        raw_title = sp[0][3:].strip()
                        sid = (
                            (
                                raw_title.split(" ", 1)[1].split(" ", 1)[0]
                                if raw_title.split(" ")[0].rstrip(".").isdigit()
                                else raw_title.split(" ", 1)[0]
                            )
                            if raw_title
                            else ""
                        )
                        tail = sp[1] if len(sp) > 1 else ""
                        if "exit" in tail or "skip" in tail or "ABORT" in tail:
                            completed_ids.add(sid)
                return bool(started_ids - completed_ids)

            running_blocks = [b for b in blocks if _block_is_running(b)]
            block = running_blocks[0] if running_blocks else blocks[-1]
            first = block.split("\n", 1)[0]
            # Parse the attempt header for n + started.
            if first.startswith("## attempt "):
                parts = first.split(None, 4)
                if len(parts) >= 4:
                    attempt_n = int(parts[2]) if parts[2].isdigit() else 0
                    attempt_started = parts[3]
            else:
                # Legacy: '# M5 runbook — <ISO>'.
                attempt_n = len(attempt_starts)
                parts = first.split("—", 1)
                if len(parts) > 1:
                    attempt_started = parts[1].strip()
            # Walk lines: 'started <id> <ISO>' and '## <idx>. <title> — ...'.
            for line in block.splitlines()[1:]:
                if line.startswith("started "):
                    sp = line.split(None, 2)
                    if len(sp) >= 3:
                        steps_raw.append({"id": sp[1], "started": sp[2], "state": "running"})
                elif line.startswith("# M5 runbook — ") or line.startswith("## attempt "):
                    # A new attempt starts within this block — stop (we only want
                    # the last attempt's steps).
                    break
                elif line.startswith("## ") and not line.startswith("## attempt "):
                    sp = line.split(" — ", 1)
                    raw_title = sp[0][3:].strip()  # '1. doctor (gate)'
                    # Strip the leading 'N. ' index prefix -> 'doctor (gate)'.
                    title = (
                        raw_title.split(" ", 1)[1]
                        if raw_title.split(" ")[0].rstrip(".").isdigit()
                        else raw_title
                    )
                    # The step id is the first word of the title (doctor,
                    # parity-..., bench-quality, ...).
                    sid = title.split(" ", 1)[0] if title else raw_title
                    tail = sp[1] if len(sp) > 1 else ""
                    state = "running"
                    exit_code = None
                    wall_s = None
                    if "skip" in tail:
                        state = "skipped"
                    elif "ABORT" in tail:
                        state = "failed"
                    elif "exit 0" in tail:
                        state = "ok"
                    elif "exit" in tail:
                        state = "failed"
                        for tok in tail.split():
                            if tok.startswith("exit"):
                                pass
                            elif tok.lstrip("-").isdigit() and "exit" in tail:
                                try:
                                    exit_code = int(tok)
                                except ValueError:
                                    pass
                    for tok in tail.replace("—", " ").split():
                        if tok.endswith("s") and tok[:-1].replace(".", "", 1).isdigit():
                            try:
                                wall_s = float(tok[:-1])
                            except ValueError:
                                pass
                            break
                    if exit_code is None and "exit 0" in tail:
                        exit_code = 0
                    elif exit_code is None and "exit" in tail:
                        m = tail.split("exit")
                        if len(m) > 1:
                            for tok in m[1].split():
                                if tok.lstrip("-").isdigit():
                                    exit_code = int(tok)
                                    break
                    # Match this completion line to the last 'started' with the
                    # same id (or create a new entry).
                    matched = False
                    for s in reversed(steps_raw):
                        if s.get("id") == sid and s.get("state") == "running" and "title" not in s:
                            s.update(
                                {
                                    "title": title,
                                    "state": state,
                                    "wall_s": wall_s,
                                    "exit": exit_code,
                                }
                            )
                            matched = True
                            break
                    if not matched:
                        steps_raw.append(
                            {
                                "id": sid,
                                "title": title,
                                "state": state,
                                "wall_s": wall_s,
                                "exit": exit_code,
                                "started": None,
                                "argv": [],
                            }
                        )
                elif line.startswith("step: "):
                    # W6-UI-3c: the authoritative step id (matches 'started <id>').
                    # The '## ' line above created an entry with a placeholder
                    # sid (title's first word); replace it with the real id and
                    # try to merge with a prior 'started' entry that has the
                    # same id.
                    real_sid = line[6:].strip()
                    if steps_raw:
                        steps_raw[-1]["id"] = real_sid
                        # Merge into a prior 'started' entry if one exists.
                        for s in reversed(steps_raw[:-1]):
                            if (
                                s.get("id") == real_sid
                                and s.get("state") == "running"
                                and "title" not in s
                            ):
                                s.update(
                                    {
                                        "title": steps_raw[-1].get("title"),
                                        "state": steps_raw[-1].get("state"),
                                        "wall_s": steps_raw[-1].get("wall_s"),
                                        "exit": steps_raw[-1].get("exit"),
                                        "argv": steps_raw[-1].get("argv", []),
                                    }
                                )
                                steps_raw.pop()
                                break
                elif line.startswith("cmd: "):
                    # The RUNBOOK 'cmd: <argv>' line (written by runbook_append).
                    # Attach to the last step.
                    if steps_raw:
                        steps_raw[-1]["argv"] = line[5:].strip().split()
        else:
            # No attempt header (pre-W6 RUNBOOK): fall back to the old parse.
            for s in parse_runbook(out_dir):
                state = {"done": "ok", "aborted": "failed", "skipped": "skipped"}.get(
                    s.get("state", ""), "waiting"
                )
                steps_raw.append(
                    {
                        "id": s.get("title", ""),
                        "title": s.get("title", ""),
                        "state": state,
                        "wall_s": None,
                        "exit": None,
                        "started": None,
                    }
                )
    # W6-UI-3g: dedupe by id — last occurrence wins (legacy append-only
    # RUNBOOKs stack the same step across attempts).
    seen_ids: dict[str, dict] = {}
    for s in steps_raw:
        sid = s.get("id") or ""
        if sid in seen_ids:
            seen_ids[sid].update(s)
        else:
            seen_ids[sid] = s
    steps_raw = list(seen_ids.values())
    # Enrich with stdout_tail + argv (from m5 plan_steps if importable).
    steps_out: list[dict] = []
    try:
        from benchmarks.m5 import Step, plan_steps  # noqa: F401
        # We cannot call plan_steps without args; argv comes from the runbook
        # 'cmd:' line inside each step block instead.
    except Exception:  # noqa: BLE001
        pass
    for s in steps_raw:
        sid = s.get("id") or ""
        step = {
            "id": sid,
            "title": s.get("title") or sid,
            "state": s.get("state", "waiting"),
            "wall_s": s.get("wall_s"),
            "exit": s.get("exit"),
            "started": s.get("started"),
            "argv": s.get("argv", []),
            "stdout_tail": _read_step_log_tail(out_dir, sid) if sid else None,
            "error": _extract_step_error(out_dir, sid) if sid else None,
        }
        steps_out.append(step)
    # W6-UI round 4b: infer the LIVE step from the newest heartbeat's folder
    # path. RUNBOOK writes a step only on exit, so a running combo's step is
    # not in the RUNBOOK yet. Infer the step name from the bench section:
    #   ab/...           -> 'A/B <branch>'
    #   bench-rest/...   -> 'rest bench'
    #   bench-quality/...-> 'quality'
    #   invariance/...   -> 'invariance'
    # Show it as RUNNING above the finished RUNBOOK steps (only if no step
    # with the same id is already in the list as running).
    combos = _combo_dirs(out_dir)
    live_combo, live_hb = _find_live_combo(combos)
    # Only infer a live step if the combo is actually RUNNING (no report.json
    # and no run_failed.txt). A finished combo with a heartbeat is not live.
    if live_combo and live_hb and _combo_status(live_combo, out_dir) == "running":
        live_step_id = _infer_live_step_id(live_combo, out_dir)
        if live_step_id:
            # Only add if no step with this id is already running.
            already = any(
                s.get("id") == live_step_id and s.get("state") == "running" for s in steps_out
            )
            if not already:
                steps_out.append(
                    {
                        "id": live_step_id,
                        "title": live_step_id,
                        "state": "running",
                        "wall_s": None,
                        "exit": None,
                        "started": _iso_ts(live_hb.get("ts")) if live_hb.get("ts") else None,
                        "argv": [],
                        "stdout_tail": None,
                        "error": None,
                    }
                )
    return {"attempt_n": attempt_n, "attempt_started": attempt_started, "steps": steps_out}


def _combo_status(combo_dir: Path, out_dir: Path) -> str:
    """done | running | queued | failed for one combo dir."""
    run_failed = (combo_dir / "run_failed.txt").exists()
    report_exists = (combo_dir / "report.json").exists()
    hb = parse_heartbeat(combo_dir)
    has_preds = (combo_dir / "predictions.jsonl").exists()
    if run_failed:
        return "failed"
    if report_exists:
        return "done"
    if hb or has_preds:
        return "running"
    return "queued"


def _gb(v) -> float | None:
    """Bytes -> GB (2**30), or None for absent/-1 sentinels."""
    if v is None or not isinstance(v, (int, float)) or v < 0:
        return None
    return round(v / 2**30, 2)


def _iso_ts(v) -> str | None:
    """Convert an epoch-seconds ts to ISO-8601 UTC string; pass through None/str.

    The contract requires every ts to be ISO-8601 UTC. Heartbeat records carry
    an epoch float (time.time()); convert at the source so the page never
    renders a raw float like '1789986387.'.
    """
    if v is None:
        return None
    if isinstance(v, str):
        return v
    if isinstance(v, (int, float)):
        from datetime import UTC, datetime

        return datetime.fromtimestamp(v, tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    return None


def _derive_config_from_layout(combo_dir: Path) -> dict:
    """Derive model/track/scorer/dataset when a live combo has no run.json.

    Looks for a sibling run.json in the same model folder (a completed combo
    of the same model), then falls back to the folder name
    <track>-<scorer>-<dataset>.
    """
    # Try a sibling run.json in the same model folder.
    model_dir = combo_dir.parent
    for sibling in sorted(model_dir.iterdir()) if model_dir.is_dir() else []:
        if sibling == combo_dir or not sibling.is_dir():
            continue
        sibling_run = parse_run_json(sibling)
        if sibling_run:
            cfg = (sibling_run.get("config") if sibling_run else {}) or {}
            if cfg.get("model"):
                # Carry the model; the combo folder name gives track/scorer/dataset.
                name = combo_dir.name
                parts = name.split("-", 2)
                return {
                    "model": cfg.get("model"),
                    "track": parts[0] if len(parts) > 0 else None,
                    "scorer": parts[1] if len(parts) > 1 else None,
                    "dataset": parts[2] if len(parts) > 2 else None,
                }
    # Fallback: parse the combo folder name <track>-<scorer>-<dataset>.
    name = combo_dir.name
    parts = name.split("-", 2)
    return {
        "track": parts[0] if len(parts) > 0 else None,
        "scorer": parts[1] if len(parts) > 1 else None,
        "dataset": parts[2] if len(parts) > 2 else None,
    }


def _dataset_name(cfg: dict) -> str | None:
    """Derive the dataset name from run.json config.

    bench writes 'dataset_path' (a full path), not 'dataset'. The name is
    the stem of that path. Falls back to 'dataset' or 'source' if present.
    """
    dp = cfg.get("dataset_path")
    if dp:
        return Path(dp).stem
    return cfg.get("dataset") or cfg.get("source")


def _scorer_name(cfg: dict) -> str | None:
    """bench writes 'scoring' (not 'scorer')."""
    return cfg.get("scorer") or cfg.get("scoring")


def _read_runbook_hash(out_dir: Path) -> str | None:
    """The git sha from the LAST '## attempt <n> <ISO> <hash>' RUNBOOK line.

    A RUNBOOK attempt header overrides the file's environment git_sha (it is
    the run's own hash, not the file's). Returns None when no attempt header
    exists.
    """
    rb = out_dir / "RUNBOOK.md"
    if not rb.exists():
        return None
    try:
        text = rb.read_text(encoding="utf-8")
    except OSError:
        return None
    import re

    headers = list(re.finditer(r"^## attempt \d+ \S+ (\S+)", text, re.MULTILINE))
    if not headers:
        return None
    return headers[-1].group(1)


def _combo_id(combo_dir: Path, out_dir: Path) -> str:
    """The out-relative path '<bench-dir>/<model-folder>/<combo>'.

    This is unique across models (two models can have the same combo folder
    name). For a single-combo bench (run.json at the out root), returns '.'.
    """
    try:
        rel = combo_dir.relative_to(out_dir)
        rel_str = str(rel)
        return rel_str if rel_str else "."
    except ValueError:
        return combo_dir.name


def _model_from_folder(combo_dir: Path) -> str | None:
    """Derive the model id from the parent folder slug <machine>-<model-slug>.

    The slug is '<chip>-<ram>gb-<model-slug>' where the model slug uses '--'
    for '/': 'm5max-128gb-mlx-community--qwen3-8b-4bit' ->
    'mlx-community/Qwen3-8B-4bit'. Returns None if the folder doesn't match
    (e.g. alias folders like 'm5max-128gb-quality' with no '--').
    """
    parent = combo_dir.parent
    slug = parent.name
    # The model slug starts after the machine prefix (<chip>-<ram>gb-).
    # Split on '-' and skip the first 2 parts (chip + ramgb).
    parts = slug.split("-")
    if len(parts) < 3:
        return None
    model_slug = "-".join(parts[2:])
    # The model slug uses '--' for '/'. If there's no '--', it's not a model
    # folder (e.g. 'quality' alias).
    if "--" not in model_slug:
        return None
    return model_slug.replace("--", "/")


def _model_from_sibling(combo_dir: Path) -> str | None:
    """Find config.model from a sibling combo's run.json in the same model folder.

    For alias/stub folders (e.g. m5max-128gb-quality) whose own run.json has
    no config.model, a sibling combo in the same model folder may carry the
    real model id. Returns None if no sibling has one.
    """
    model_dir = combo_dir.parent
    if not model_dir.is_dir():
        return None
    for sibling in sorted(model_dir.iterdir()):
        if sibling == combo_dir or not sibling.is_dir():
            continue
        sibling_run = parse_run_json(sibling)
        if sibling_run:
            sib_cfg = (sibling_run.get("config") if sibling_run else {}) or {}
            if sib_cfg.get("model"):
                return sib_cfg["model"]
    return None


def _infer_live_step_id(live_combo: Path, out_dir: Path) -> str | None:
    """Infer the pipeline step name from the live combo's folder path.

    RUNBOOK writes a step only on exit, so a running combo's step is not in
    the RUNBOOK yet. Infer from the bench section:
      ab/...           -> 'A/B bench'
      bench-rest/...   -> 'rest bench'
      bench-quality/...-> 'quality'
      invariance/...   -> 'invariance'
    Returns None if the path doesn't match a known section.
    """
    try:
        rel = live_combo.relative_to(out_dir)
    except ValueError:
        return None
    parts = rel.parts
    if not parts:
        return None
    top = parts[0]
    if top == "ab":
        return "A/B bench"
    if top == "bench-rest":
        return "rest bench"
    if top == "bench-quality":
        return "quality"
    if top == "invariance":
        return "invariance"
    return None


def _part_from_folder(combo_dir: Path, idx: int) -> str | None:
    """Derive track/scorer/dataset from the combo folder name <track>-<scorer>-<dataset>.

    idx 0=track, 1=scorer, 2=dataset. Returns None if not enough parts.
    """
    parts = combo_dir.name.split("-", 2)
    if idx < len(parts):
        return parts[idx]
    return None


def _mx_device_memory_gb() -> float | None:
    """Machine total memory from mlx's device_info (Apple Silicon unified mem).

    Falls back to None (caller uses env.ram_gb or 128).
    """
    try:
        import mlx.core as mx

        return round(mx.device_info("metal").get("memory_size", 0) / 2**30, 1) or None
    except Exception:  # noqa: BLE001
        return None


_DATASET_COUNT_CACHE: dict[str, tuple[float, int]] = {}


def _count_dataset_cases(ds_path: Path) -> int:
    """Count lines in the dataset jsonl, cached by (path, mtime).

    The dataset_path is the cached jsonl bench writes into run.json config.
    Counting 576 lines on every refresh is wasteful; cache by mtime so a
    growing dataset (invariance) is recounted only when it changes.
    """
    try:
        mtime = ds_path.stat().st_mtime
    except OSError:
        return 0
    key = str(ds_path)
    cached = _DATASET_COUNT_CACHE.get(key)
    if cached and cached[0] == mtime:
        return cached[1]
    n = count_lines(ds_path)
    _DATASET_COUNT_CACHE[key] = (mtime, n)
    return n


def _is_combo_dir(d: Path) -> bool:
    """A combo dir has any of run.json / heartbeat.jsonl / predictions.jsonl."""
    return any((d / m).exists() for m in ("run.json", "heartbeat.jsonl", "predictions.jsonl"))


def _heartbeat_is_recent(hb_path: Path | None, *, max_age_s: float = 6.0) -> bool:
    """True when the heartbeat.jsonl mtime is younger than max_age_s (3x refresh).

    Heartbeat records carry NO timestamp (keys: active_memory_bytes,
    cache_memory_bytes, cases_done, combo, elapsed_s, peak_memory_bytes,
    pred_lines). So heartbeat age = the heartbeat.jsonl file mtime.
    """
    if not hb_path or not hb_path.exists():
        return False
    try:
        return (time.time() - hb_path.stat().st_mtime) < max_age_s
    except OSError:
        return False


def _find_live_combo(combos: list[Path]) -> tuple[Path | None, dict | None]:
    """The live combo: the one with the newest heartbeat.jsonl mtime.

    Falls back to the last combo (sorted) when no heartbeat exists.
    """
    best_combo: Path | None = None
    best_mtime: float = -1.0
    best_hb: dict | None = None
    for c in combos:
        hb_path = c / "heartbeat.jsonl"
        if not hb_path.exists():
            continue
        try:
            mtime = hb_path.stat().st_mtime
        except OSError:
            continue
        if mtime > best_mtime:
            best_mtime = mtime
            best_combo = c
            best_hb = parse_heartbeat(c)
    if best_combo is None and combos:
        best_combo = combos[-1]
        best_hb = parse_heartbeat(best_combo)
    return best_combo, best_hb


def build_dashboard(out_dir: str | Path) -> dict:
    """The full /dashboard.json payload (frozen W6-UI-3a contract).

    Pure read-only function: reads files the run already writes, truncates
    half-written trailing lines to the last complete JSON line (same rule as
    ``jevmlx.resume``), and returns nulls (never raises) for missing files.

    Top-level keys: run, now, memory, health, pipeline, aggregates, results,
    events, history.
    """
    out_dir = Path(out_dir)
    run = parse_run_json(out_dir)
    env = (run.get("environment") if run else {}) or {}
    # W6-UI round 4b: 'memory' is a TOP-LEVEL key in run.json (the real M5
    # shape), not under config. Fall back to config.memory for old fixtures.
    mem_cfg = (run.get("memory") if run else {}) or {}
    if not mem_cfg:
        cfg_tmp = (run.get("config") if run else {}) or {}
        mem_cfg = (cfg_tmp.get("memory") if cfg_tmp else {}) or {}
    combos = _combo_dirs(out_dir)
    # W6-UI-3g: take environment/hash from the NEWEST run.json by mtime (not
    # the first found). The top-level <out> may have no run.json (m5 writes
    # one per combo).
    if not env and combos:
        best_run_mtime = -1.0
        best_env = {}
        best_mem_cfg = {}
        for c in combos:
            rj = c / "run.json"
            if not rj.exists():
                continue
            try:
                mtime = rj.stat().st_mtime
            except OSError:
                continue
            combo_run = parse_run_json(c)
            if combo_run and combo_run.get("environment") and mtime > best_run_mtime:
                best_run_mtime = mtime
                best_env = combo_run["environment"]
                # 'memory' is top-level (real M5 shape); fall back to config.memory.
                best_mem_cfg = (combo_run.get("memory") if combo_run else {}) or {}
                if not best_mem_cfg:
                    combo_cfg_tmp = (combo_run.get("config") if combo_run else {}) or {}
                    best_mem_cfg = (combo_cfg_tmp.get("memory") if combo_cfg_tmp else {}) or {}
        if best_env:
            env = best_env
            if not mem_cfg:
                mem_cfg = best_mem_cfg
    # W6-UI-3g: a RUNBOOK attempt header (## attempt <n> <ISO> <hash>) overrides
    # the git sha when present (it is the run's own hash, not the file's).
    pipeline = _parse_pipeline(out_dir)
    rb_hash = _read_runbook_hash(out_dir)
    if rb_hash:
        env = {**env, "git_sha": rb_hash} if env else {"git_sha": rb_hash}
    # Find the live combo: the one with the newest heartbeat.jsonl mtime
    # anywhere under <out>. Falls back to the last combo.
    now_combo, now_hb = _find_live_combo(combos)
    now_combo_run = parse_run_json(now_combo) if now_combo else {}
    now_cfg = (now_combo_run.get("config") if now_combo_run else {}) or {}
    # A LIVE combo may have no run.json yet — derive config from a sibling
    # run.json (same model folder) or from the folder name
    # <track>-<scorer>-<dataset>.
    if not now_cfg and now_combo:
        now_cfg = _derive_config_from_layout(now_combo)
    # --- run ---
    alerts: list[dict] = []
    # Build health first so alerts can reference it.
    health = _build_health(out_dir, combos, now_combo, now_hb, env, mem_cfg, pipeline)
    for h in health:
        if h.get("state") == "fail":
            alerts.append(
                {
                    "level": "fail",
                    "text": h.get("detail") or h.get("rule", ""),
                    "source": h.get("rule"),
                }
            )
    sleep_blocked = _read_sleep_blocked(out_dir)
    # state=running when the newest heartbeat.jsonl mtime is younger than 3x
    # the refresh interval (default 6s), or a RUNBOOK step is 'running'.
    # Heartbeat records carry NO ts — age = file mtime.
    now_hb_path = now_combo / "heartbeat.jsonl" if now_combo else None
    state = "stopped"
    if _heartbeat_is_recent(now_hb_path, max_age_s=6):
        state = "running"
    elif any(s.get("state") == "running" for s in pipeline.get("steps", [])):
        state = "running"
    elif now_hb:
        # Heartbeat exists but is stale — still running (just quiet).
        state = "running"
    elif (out_dir / "SUMMARY.md").exists():
        state = "done"
    run_block = {
        "out_dir": str(out_dir),
        "hash": (env.get("git_sha") or "—")[:12],
        "machine": env.get("chip") or env.get("machine_model") or "—",
        "mlx_version": env.get("mlx_version") or "—",
        "attempt_n": pipeline.get("attempt_n") or 0,
        "attempt_started": pipeline.get("attempt_started"),
        "awake_s": now_hb.get("elapsed_s") if now_hb else None,
        "sleep_blocked": sleep_blocked,
        "state": state,
        "alerts": alerts,
    }
    # --- now ---
    now_block = _build_now(now_combo, now_hb, now_cfg, out_dir)
    # --- memory ---
    # W6-UI round 4b: cap from the newest run.json's top-level 'memory' block
    # (metal_cache_limit_bytes, 8589934592 -> 8.0 GB). stop_gb = the m5 stop
    # rule constant (10 GB).
    newest_run = parse_run_json(now_combo) if now_combo else {}
    newest_mem = (newest_run.get("memory") if newest_run else {}) or {}
    cap_bytes = newest_mem.get("metal_cache_limit_bytes") or mem_cfg.get("metal_cache_limit_bytes")
    # machine_gb from mx.device_info if available (Apple Silicon).
    machine_gb = env.get("ram_gb") or _mx_device_memory_gb() or 128
    memory_block = {
        "cache_gb": _gb(now_hb.get("cache_memory_bytes")) if now_hb else None,
        "active_gb": _gb(now_hb.get("active_memory_bytes")) if now_hb else None,
        "peak_gb": _gb(now_hb.get("peak_memory_bytes")) if now_hb else None,
        "cap_gb": _gb(cap_bytes),
        "stop_gb": 10.0,  # m5 stop rule: halt if cache >= 10 GB
        "machine_gb": machine_gb,
    }
    # --- aggregates ---
    aggregates = _build_aggregates(out_dir, combos)
    # --- results ---
    results = _build_results(out_dir, combos, env)
    # --- events ---
    events = _build_events(out_dir, combos)
    # --- history ---
    history = _build_history(out_dir)
    return {
        "run": run_block,
        "now": now_block,
        "memory": memory_block,
        "health": health,
        "pipeline": pipeline,
        "aggregates": aggregates,
        "results": results,
        "events": events,
        "history": history,
    }


def _build_health(out_dir, combos, now_combo, now_hb, env, mem_cfg, pipeline) -> list[dict]:
    """Six health rules: ok|warn|fail with a detail string."""
    rules: list[dict] = []
    # cache_over_stop
    cache_gb = _gb(now_hb.get("cache_memory_bytes")) if now_hb else None
    stop_gb = _gb(mem_cfg.get("metal_cache_stop_bytes"))
    if cache_gb is not None and stop_gb is not None and cache_gb >= stop_gb:
        rules.append(
            {
                "rule": "cache_over_stop",
                "state": "fail",
                "detail": f"cache {cache_gb} GB >= stop {stop_gb} GB",
            }
        )
    elif cache_gb is not None and stop_gb is not None and cache_gb >= stop_gb * 0.9:
        rules.append(
            {
                "rule": "cache_over_stop",
                "state": "warn",
                "detail": f"cache {cache_gb} GB near stop {stop_gb} GB",
            }
        )
    else:
        rules.append(
            {
                "rule": "cache_over_stop",
                "state": "ok",
                "detail": f"cache {cache_gb} GB / stop {stop_gb} GB"
                if cache_gb is not None
                else "no heartbeat yet",
            }
        )
    # sleep_windows
    sleep_blocked = _read_sleep_blocked(out_dir)
    rules.append(
        {
            "rule": "sleep_windows",
            "state": "ok" if sleep_blocked else "warn",
            "detail": "caffeinated" if sleep_blocked else "off (not caffeinated)",
        }
    )
    # run_failed_combos
    n_failed = sum(1 for c in combos if (c / "run_failed.txt").exists())
    rules.append(
        {
            "rule": "run_failed_combos",
            "state": "fail" if n_failed else "ok",
            "detail": f"{n_failed} combo(s) failed",
        }
    )
    # parity_fail_models: count once per model (the parity.json is at the
    # MODEL folder level, not per combo). Dedupe by parity.json path.
    n_parity_fail = 0
    fail_models: list[str] = []
    seen_parity: set[Path] = set()
    for c in combos:
        model_dir = c.parent if c.parent.name != out_dir.name else c
        parity_path = model_dir / "parity.json"
        if parity_path in seen_parity or not parity_path.exists():
            continue
        seen_parity.add(parity_path)
        try:
            p = json.loads(parity_path.read_text(encoding="utf-8"))
            if p.get("status") == "FAIL":
                n_parity_fail += 1
                model_name = p.get("model") or _model_from_folder(c)
                if model_name:
                    fail_models.append(model_name)
        except (OSError, json.JSONDecodeError):
            pass
    fail_detail = (
        (f"{n_parity_fail} model(s): " + ", ".join(fail_models)) if fail_models else "0 models"
    )
    rules.append(
        {
            "rule": "parity_fail_models",
            "state": "fail" if n_parity_fail else "ok",
            "detail": fail_detail,
        }
    )
    # metal_alloc_retries
    n_retries = 0
    for c in combos:
        for hb in read_jsonl_safe(c / "heartbeat.jsonl"):
            if isinstance(hb.get("alloc_retry"), int):
                n_retries += hb["alloc_retry"]
    rules.append(
        {
            "rule": "metal_alloc_retries",
            "state": "warn" if n_retries else "ok",
            "detail": f"{n_retries} retries",
        }
    )
    # heartbeat_age = heartbeat.jsonl file mtime (records carry NO ts).
    age = None
    if now_combo:
        hb_path = now_combo / "heartbeat.jsonl"
        if hb_path.exists():
            try:
                age = int(time.time() - hb_path.stat().st_mtime)
            except OSError:
                pass
    elif now_hb and isinstance(now_hb.get("elapsed_s"), (int, float)):
        age = None
    if age is None:
        rules.append({"rule": "heartbeat_age", "state": "ok", "detail": "no heartbeat"})
    elif age > 120:
        rules.append({"rule": "heartbeat_age", "state": "fail", "detail": f"{age} s stale"})
    else:
        rules.append({"rule": "heartbeat_age", "state": "ok", "detail": f"{age} s"})
    return rules


def _find_sibling_cases_total(
    out_dir: Path, now_combo: Path | None, dataset_path: str
) -> int | None:
    """Find a FINISHED sibling combo with the same dataset_path.

    Returns its counts.cases (the record count — same unit as a finished
    combo's total). Used to derive cases_total for a running combo whose
    own run.json has no counts yet. Returns None when no sibling exists.
    """
    for c in _combo_dirs(out_dir):
        if c == now_combo:
            continue
        rj = parse_run_json(c)
        if not rj:
            continue
        sib_cfg = (rj.get("config") if rj else {}) or {}
        if sib_cfg.get("dataset_path") != dataset_path:
            continue
        sib_counts = (rj.get("counts") if rj else {}) or {}
        if sib_counts.get("cases"):
            return sib_counts["cases"]
    return None


def _derive_run_index(out_dir: Path, now_combo: Path | None) -> tuple[int | None, int | None]:
    """Derive run_i / run_n from the model folder count under bench-*.

    The m5 bench writes one folder per model under bench-rest/ (or
    bench-quality/). run_n = the number of model folders; run_i = the
    1-based position of the current model's folder. Returns (None, None)
    when not derivable.
    """
    if now_combo is None:
        return None, None
    # Walk up to find the bench-* dir (parent of the model folder).
    # now_combo = <out>/bench-<kind>/<model-folder>/<combo>
    model_folder = now_combo.parent
    bench_dir = model_folder.parent
    if not bench_dir.name.startswith("bench-"):
        return None, None
    # Collect all model folders that have at least one combo with a run.json.
    model_folders = sorted(
        d
        for d in bench_dir.iterdir()
        if d.is_dir()
        and any(
            (d / c / "run.json").exists()
            for c in [combo.name for combo in d.iterdir() if combo.is_dir()]
        )
    )
    if not model_folders:
        return None, None
    run_n = len(model_folders)
    try:
        run_i = model_folders.index(model_folder) + 1
    except ValueError:
        return None, run_n
    return run_i, run_n


def _build_now(now_combo, now_hb, now_cfg, out_dir) -> dict:
    """The 'now' panel: current combo progress + ETA."""
    total = None  # W6-UI round 4b: None (—) when not derivable, never 0
    done = 0
    pred_lines = 0
    now_run = parse_run_json(now_combo) if now_combo else {}
    now_counts = (now_run.get("counts") if now_run else {}) or {}
    if now_combo:
        # W6-UI round 4b: cases_done from the heartbeat is in TICK units
        # (case × rotation variants), not record units. counts.cases (45)
        # is records — a different unit. The total ticks is not directly in
        # run.json until the combo finishes. Strategy:
        #   1. If this combo is finished (has counts.prediction_lines),
        #      total = counts.cases (records, the finished unit).
        #   2. Else, find a finished SIBLING combo with the same
        #      config.dataset_path and use its counts.cases as the total.
        #   3. If no sibling, total stays None (—) — never 0.
        # The progress PERCENT comes from pred_lines vs the sibling's
        # prediction_lines (a ratio, not a count).
        done = len(read_jsonl_safe(now_combo / "completed_cases.jsonl"))
        if now_hb and isinstance(now_hb.get("cases_done"), int):
            done = max(done, now_hb["cases_done"])
        pred_lines = now_hb.get("pred_lines", 0) if now_hb else 0
        if not pred_lines:
            pred_lines = count_lines(now_combo / "predictions.jsonl")
        # Derive total: finished combo has counts; running combo needs a sibling.
        if now_counts.get("cases"):
            total = now_counts["cases"]
        elif now_cfg and now_cfg.get("dataset_path"):
            # Find a finished sibling combo with the same dataset_path.
            sibling_total = _find_sibling_cases_total(out_dir, now_combo, now_cfg["dataset_path"])
            if sibling_total:
                total = sibling_total
        elif now_combo:
            # Last resort: dataset jsonl line count.
            ds_path = now_combo / "dataset.jsonl"
            if not ds_path.exists():
                ds_path = out_dir / "dataset.jsonl"
            if not ds_path.exists():
                dp = now_cfg.get("dataset_path") if now_cfg else None
                if dp:
                    ds_path = Path(dp)
            if ds_path.exists():
                total = _count_dataset_cases(ds_path)
    # cases_per_h from the last two heartbeat records: (cases_done delta)
    # / (elapsed_s delta). Heartbeat records carry NO ts — elapsed_s is the
    # wall-clock seconds since the combo started (written by evalrun).
    cases_per_h = None
    if now_combo:
        hbs = read_jsonl_safe(now_combo / "heartbeat.jsonl")
        if len(hbs) >= 2:
            first = hbs[0]
            last = hbs[-1]
            d_cases = (last.get("cases_done", 0) or 0) - (first.get("cases_done", 0) or 0)
            d_s = (last.get("elapsed_s", 0) or 0) - (first.get("elapsed_s", 0) or 0)
            if d_s > 0:
                cases_per_h = round(d_cases / d_s * 3600, 1)
    eta_s = None
    elapsed = now_hb.get("elapsed_s", 0) if now_hb else 0
    if total and done and elapsed and done < total:
        eta_s = int((total - done) / (done / elapsed))
    # running accuracy from predictions so far.
    running_acc = None
    running_maj = None
    if now_combo:
        records = read_jsonl_or_gz(now_combo / "predictions.jsonl")
        if records:
            from jevmlx.evalmetrics import field_accuracy, majority_class_baseline

            labelled = [r for r in records if r.get("label") is not None]
            if labelled:
                running_acc = field_accuracy(labelled)
                maj = majority_class_baseline(labelled) if labelled else None
                running_maj = maj.get("overall") if isinstance(maj, dict) else maj
    # heartbeat age = the heartbeat.jsonl file mtime (records carry NO ts).
    hb_age = None
    if now_combo:
        hb_path = now_combo / "heartbeat.jsonl"
        if hb_path.exists():
            try:
                hb_age = int(time.time() - hb_path.stat().st_mtime)
            except OSError:
                pass
    # W6-UI round 4b: run_i/run_n not in config; derive from model folders.
    _run_i, _run_n = _derive_run_index(out_dir, now_combo)
    return {
        "model": now_cfg.get("model") or "—",
        "track": now_cfg.get("track") or "—",
        "scorer": _scorer_name(now_cfg) or "—",
        "dataset": _dataset_name(now_cfg) or "—",
        "run_i": now_cfg.get("run_i") or _run_i,
        "run_n": now_cfg.get("run_n") or _run_n,
        "cases_done": done,
        "cases_total": total,
        "pred_lines": pred_lines,
        "cases_per_h": cases_per_h,
        "eta_s": eta_s,
        "combo_elapsed_s": elapsed,
        "heartbeat_age_s": hb_age,
        "running_accuracy": running_acc,
        "running_majority": running_maj,
    }


def _build_aggregates(out_dir, combos) -> dict:
    """Aggregate metrics across all done/running combos."""
    from jevmlx.evalmetrics import field_accuracy

    par_acc: list[float] = []
    naive_acc: list[float] = []
    exact: list[float] = []
    par_time: list[float] = []
    naive_time: list[float] = []
    parity_counts = {"pass": 0, "drift": 0, "fail": 0}
    seen_parity_paths: set[Path] = set()
    cases_scored = 0
    pred_lines = 0
    wall_s = 0
    for c in combos:
        run = parse_run_json(c)
        metrics = (run.get("metrics") if run else {}) or {}
        cfg = (run.get("config") if run else {}) or {}
        track = cfg.get("track") or "parallel"
        records = read_jsonl_or_gz(c / "predictions.jsonl")
        labelled = [r for r in records if r.get("label") is not None]
        if labelled:
            acc = field_accuracy(labelled)
            if acc is not None:
                if track == "naive_local":
                    naive_acc.append(acc)
                else:
                    par_acc.append(acc)
            er = metrics.get("exact_record_accuracy")
            if er is not None:
                exact.append(er)
        # time per case from timing.json median.
        timing = _read_timing_json(c)
        t = (timing.get("median") or {}).get("per_item_end_to_end_ms")
        if t is not None:
            t_s = t / 1000.0
            if track == "naive_local":
                naive_time.append(t_s)
            else:
                par_time.append(t_s)
        # parity (model-level; dedupe by parity.json path so it's counted once).
        model_dir = c.parent if c.parent.name != out_dir.name else c
        parity_path = model_dir / "parity.json"
        if parity_path.exists() and parity_path not in seen_parity_paths:
            seen_parity_paths.add(parity_path)
            try:
                p = json.loads(parity_path.read_text(encoding="utf-8"))
                st = (p.get("status", "PASS") or "PASS").lower()
                if st in parity_counts:
                    parity_counts[st] += 1
            except (OSError, json.JSONDecodeError):
                pass
        cases_scored += ((run.get("counts") if run else {}) or {}).get("cases", 0) or 0
        pred_lines += count_lines(c / "predictions.jsonl")
        hb = parse_heartbeat(c)
        if hb and isinstance(hb.get("elapsed_s"), (int, float)):
            wall_s += hb["elapsed_s"]
    return {
        "field_accuracy_parallel_labels": round(sum(par_acc) / len(par_acc), 4)
        if par_acc
        else None,
        "field_accuracy_naive": round(sum(naive_acc) / len(naive_acc), 4) if naive_acc else None,
        "exact_record": round(sum(exact) / len(exact), 4) if exact else None,
        "time_per_case_parallel_s": round(sum(par_time) / len(par_time), 3) if par_time else None,
        "time_per_case_naive_s": round(sum(naive_time) / len(naive_time), 3)
        if naive_time
        else None,
        "parity_counts": parity_counts,
        "cases_scored": cases_scored,
        "pred_lines": pred_lines,
        "wall_s": wall_s or None,
    }


def _read_timing_json(combo_dir: Path) -> dict:
    p = combo_dir / "timing.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _build_results(out_dir, combos, env) -> list[dict]:
    """One row per combo, README-leaderboard-shaped."""
    from jevmlx.evalmetrics import field_accuracy, majority_class_baseline, wilson_interval

    rows: list[dict] = []
    for c in combos:
        run = parse_run_json(c)
        cfg = (run.get("config") if run else {}) or {}
        metrics = (run.get("metrics") if run else {}) or {}
        records = read_jsonl_or_gz(c / "predictions.jsonl")
        labelled = [r for r in records if r.get("label") is not None]
        accuracy = field_accuracy(labelled) if labelled else metrics.get("accuracy")
        ci_low = None
        ci_high = None
        if accuracy is not None and labelled:
            correct = sum(1 for r in labelled if r.get("correct") is True)
            ci = wilson_interval(correct, len(labelled))
            if ci:
                ci_low, ci_high = ci.get("ci_low"), ci.get("ci_high")
        maj = None
        if labelled:
            m = majority_class_baseline(labelled)
            maj = m.get("overall") if isinstance(m, dict) else m
        # parity for this model.
        model_dir = c.parent if c.parent.name != out_dir.name else c
        parity_path = model_dir / "parity.json"
        parity_status = None
        parity_drift = None
        if parity_path.exists():
            try:
                p = json.loads(parity_path.read_text(encoding="utf-8"))
                parity_status = p.get("status")
                parity_drift = p.get("max_abs_drift_nats") or p.get("max_drift_nats")
            except (OSError, json.JSONDecodeError):
                pass
        timing = _read_timing_json(c)
        t = (timing.get("median") or {}).get("per_item_end_to_end_ms")
        ab_delta = _ab_delta(out_dir, c, accuracy)
        # W6-UI round 4b: model is never None. Use config.model, else
        # reconstruct the Hub id from the folder slug (-- -> /), else
        # 'alias folder (no run)' for stub folders like m5max-128gb-quality.
        model = cfg.get("model") or _model_from_folder(c)
        if model is None:
            # Check if ANY sibling combo in the same model folder has a
            # run.json with config.model (the real model id).
            model = _model_from_sibling(c) or "alias folder (no run)"
        rows.append(
            {
                "combo_id": _combo_id(c, out_dir),
                "display_name": c.name,
                "model": model,
                "dataset": _dataset_name(cfg) or _part_from_folder(c, 2),
                "scorer": _scorer_name(cfg) or _part_from_folder(c, 1),
                "track": cfg.get("track") or _part_from_folder(c, 0) or "parallel",
                "status": _combo_status(c, out_dir),
                "accuracy": accuracy,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "majority": maj,
                "exact_record": metrics.get("exact_record_accuracy"),
                "parity_status": parity_status,
                "parity_drift": parity_drift,
                "time_per_case_s": round(t / 1000.0, 3) if t is not None else None,
                "calls": timing.get("calls"),
                "cases": ((run.get("counts") if run else {}) or {}).get("cases") if run else None,
                "ab_delta": ab_delta,
            }
        )
    return rows


def _ab_delta(out_dir: Path, main_combo: Path, main_accuracy) -> float | None:
    """main accuracy - ab accuracy for the same model, else None."""
    if main_accuracy is None:
        return None
    ab_dir = out_dir / "ab"
    if not ab_dir.exists():
        return None
    run = parse_run_json(main_combo)
    model = ((run.get("config") if run else {}) or {}).get("model")
    if not model:
        return None
    for c in _combo_dirs(ab_dir):
        ab_run = parse_run_json(c)
        ab_model = ((ab_run.get("config") if ab_run else {}) or {}).get("model")
        if ab_model == model:
            ab_records = read_jsonl_or_gz(c / "predictions.jsonl")
            ab_labelled = [r for r in ab_records if r.get("label") is not None]
            if ab_labelled:
                from jevmlx.evalmetrics import field_accuracy

                ab_acc = field_accuracy(ab_labelled)
                if ab_acc is not None:
                    return round(main_accuracy - ab_acc, 4)
    return None


def _build_events(out_dir, combos) -> list[dict]:
    """Heartbeats, combo_done, alloc_retry, step_done, attempt, newest first."""
    events: list[dict] = []
    # Heartbeats from all combos. Heartbeat records carry NO ts — use the
    # heartbeat.jsonl file mtime as the event ts.
    for c in combos:
        hb_path = c / "heartbeat.jsonl"
        file_ts = None
        if hb_path.exists():
            try:
                file_ts = _iso_ts(hb_path.stat().st_mtime)
            except OSError:
                file_ts = None
        for hb in read_jsonl_safe(hb_path):
            text = (
                f"{hb.get('combo', c.name)} {hb.get('cases_done', '?')}/?"
                f" cache {_gb(hb.get('cache_memory_bytes'))} GB"
                f" peak {_gb(hb.get('peak_memory_bytes'))} GB"
            )
            kind = "heartbeat"
            if isinstance(hb.get("alloc_retry"), int) and hb["alloc_retry"]:
                kind = "alloc_retry"
                text = f"Metal allocation failed, retried (x{hb['alloc_retry']})"
            events.append({"ts": file_ts, "kind": kind, "text": text})
    # combo_done: report.json exists -> done.
    for c in combos:
        if (c / "report.json").exists():
            events.append({"ts": None, "kind": "combo_done", "text": f"{c.name} done"})
    # step_done + attempt from RUNBOOK.
    rb = out_dir / "RUNBOOK.md"
    if rb.exists():
        try:
            for line in rb.read_text(encoding="utf-8").splitlines():
                if line.startswith("## attempt "):
                    parts = line.split(None, 4)
                    events.append(
                        {
                            "ts": parts[3] if len(parts) > 3 else None,
                            "kind": "attempt",
                            "text": line,
                        }
                    )
                elif line.startswith("## ") and "exit" in line:
                    events.append({"ts": None, "kind": "step_done", "text": line[3:]})
        except OSError:
            pass

    # Sort newest-first by ISO-8601 ts string; None ts go last.
    def _sort_key(e):
        ts = e.get("ts")
        if isinstance(ts, str) and ts:
            return (0, ts)
        return (1, "")

    events.sort(key=_sort_key, reverse=True)
    return events


def _build_history(out_dir) -> list[dict]:
    """Previous attempts (all but the last) from RUNBOOK.md attempt headers."""
    rb = out_dir / "RUNBOOK.md"
    if not rb.exists():
        return []
    try:
        text = rb.read_text(encoding="utf-8")
    except OSError:
        return []
    # Find all '## attempt <n> <ISO> <hash> <argv>' headers and the steps
    # that failed between each attempt and the next.
    import re

    headers = list(re.finditer(r"^## attempt (\d+) (\S+) (\S+)", text, re.MULTILINE))
    if len(headers) <= 1:
        return []
    history: list[dict] = []
    for i, m in enumerate(headers[:-1]):  # all but the last (current) attempt
        start = m.end()
        end = headers[i + 1].start() if i + 1 < len(headers) else len(text)
        block = text[start:end]
        failed: list[str] = []
        for line in block.splitlines():
            if line.startswith("## ") and ("exit" in line and "exit 0" not in line):
                failed.append(line[3:].split(" — ")[0])
        outcome = "failed" if failed else "stopped"
        history.append(
            {
                "attempt_n": int(m.group(1)),
                "started": m.group(2),
                "hash": m.group(3),
                "outcome": outcome,
                "steps_failed": failed,
            }
        )
    return history


def build_questions(out_dir: str | Path, combo_id: str) -> list[dict]:
    """The /questions.json?combo=<id> payload: every decision of one combo.

    Each record carries the prediction-line fields plus context_text joined
    from the cached dataset jsonl by row id (never stored in results),
    options (name + p), margin, acc_so_far, rotations_same_field, rescored,
    and drift.
    """
    out_dir = Path(out_dir)
    # Resolve the combo dir by id (the out-relative path).
    if combo_id == ".":
        combo_dir = out_dir
    else:
        candidate = out_dir / combo_id
        combo_dir = candidate if candidate.exists() else None
    if combo_dir is None:
        # Fallback: match by folder name (backward compat).
        for c in _combo_dirs(out_dir):
            if c.name == combo_id:
                combo_dir = c
                break
    if combo_dir is None:
        return []
    records = read_jsonl_or_gz(combo_dir / "predictions.jsonl")
    # Build the dataset context lookup by case_id. The dataset jsonl may be
    # in the combo dir, the out dir, or at config.dataset_path (the absolute
    # path bench writes into run.json).
    ctx: dict[str, str] = {}
    ds_path = combo_dir / "dataset.jsonl"
    if not ds_path.exists():
        ds_path = out_dir / "dataset.jsonl"
    if not ds_path.exists():
        # W6-UI-3e: read from run.json config.dataset_path.
        combo_run = parse_run_json(combo_dir)
        dp = ((combo_run.get("config") if combo_run else {}) or {}).get("dataset_path")
        if dp:
            ds_path = Path(dp)
    if ds_path.exists():
        for line in ds_path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                break  # half-written line
            cid = obj.get("id")
            if cid and isinstance(obj.get("context"), str):
                ctx[cid] = obj["context"]
    # Accumulate accuracy as we walk.
    seen = 0
    correct = 0
    out: list[dict] = []
    # Group by case_id+field for rotations_same_field.
    by_case_field: dict[tuple, list[str]] = {}
    for r in records:
        key = (r.get("case_id"), r.get("field"))
        by_case_field.setdefault(key, []).append(r.get("prediction"))
    for r in records:
        ok = r.get("correct")
        if ok is True:
            correct += 1
        if ok is not None:
            seen += 1
        acc_so_far = round(correct / seen, 4) if seen else None
        prob = r.get("probability")
        per_option = r.get("per_option") or {}
        options = [{"name": k, "p": v} for k, v in per_option.items()] if per_option else []
        # margin = top1 - top2 from options or probability.
        margin = None
        if options:
            ps = sorted((o["p"] for o in options), reverse=True)
            margin = round(ps[0] - ps[1], 4) if len(ps) >= 2 else round(ps[0], 4) if ps else None
        elif isinstance(prob, (int, float)):
            margin = round(prob, 4)
        rotations = by_case_field.get((r.get("case_id"), r.get("field")), [])
        call_ms = r.get("per_item_end_to_end_ms") or r.get("latency_ms")
        out.append(
            {
                "ts": _iso_ts(r.get("ts")),
                "case_id": r.get("case_id"),
                "rotation": r.get("permutation") or r.get("rotation"),
                "field": r.get("field"),
                "predicted": r.get("prediction"),
                "label": r.get("label"),
                "ok": ok,
                "p_pred": prob,
                "margin": margin,
                "call_ms": call_ms,
                "acc_so_far": acc_so_far,
                "options": options,
                "context_text": ctx.get(r.get("case_id")),
                "field_question": r.get("field_question"),
                "rotations_same_field": rotations,
                "rescored": r.get("rescored"),
                "drift": r.get("drift"),
            }
        )
    return out


def render_json(out_dir: str | Path) -> dict:
    """Backward-compat wrapper: the /dashboard.json payload (now build_dashboard)."""
    return build_dashboard(out_dir)


def _dashboard_html(refresh: float) -> str:
    """Load the static control-room page and inject the refresh interval.

    The page is a single self-contained HTML file (inline CSS + vanilla JS,
    no framework, no CDN) shipped at ``jevmlx/web/dashboard.html``. The
    ``__REFRESH__`` placeholder becomes ``const REFRESH`` — the SSE scan
    interval, not a meta-refresh tag (the page is live via SSE + DOM
    patching, no full-page reload).
    """
    html_path = Path(__file__).parent / "web" / "dashboard.html"
    html = html_path.read_text(encoding="utf-8")
    return html.replace("__REFRESH__", str(int(max(1, refresh))))


def _watched_files(out_dir: Path) -> list[Path]:
    """Files whose mtime change signals a dashboard update.

    RUNBOOK.md + every heartbeat.jsonl / run.json under <out> (including
    combo subdirectories). Cheap os.stat scan — no file reads.
    """
    files = [out_dir / "RUNBOOK.md", out_dir / "run.json"]
    for c in _combo_dirs(out_dir):
        files.append(c / "heartbeat.jsonl")
        files.append(c / "run.json")
        files.append(c / "predictions.jsonl")
    return [f for f in files if f.exists()]


def _mtimes_signature(paths: list[Path]) -> tuple:
    """A tuple of (path, mtime_ns) pairs — changes when any file changes."""
    sig = []
    for p in paths:
        try:
            sig.append((str(p), p.stat().st_mtime_ns))
        except OSError:
            sig.append((str(p), 0))
    return tuple(sig)


def _handle_sse(
    out_dir: Path,
    wfile,
    refresh: float,
    *,
    mtime_source=None,
    max_iterations: int | None = None,
) -> None:
    """Server-Sent Events: emit 'event: dashboard' when watched files change.

    One thread per connection. Watches mtimes of RUNBOOK.md and every
    heartbeat.jsonl / run.json / predictions.jsonl under <out> (cheap
    os.stat scan every ``refresh`` seconds). On change, writes the full
    build_dashboard JSON as a ``dashboard`` event. Every 15 s without
    change, writes a ``: keepalive`` comment. The browser's EventSource
    reconnects automatically on disconnect.

    Test hooks (production ignores them):
      * ``mtime_source``: a callable returning a signature tuple; defaults
        to ``_mtimes_signature(_watched_files(out_dir))``. Inject a fake
        to drive deterministic change/timeout scenarios without threads.
      * ``max_iterations``: stop after N loop iterations (tests only).
    """
    import time

    if mtime_source is None:

        def mtime_source():
            return _mtimes_signature(_watched_files(out_dir))

    keepalive_s = 15.0
    last_keepalive = time.monotonic()
    last_sig = mtime_source()
    i = 0
    while True:
        time.sleep(refresh)
        sig = mtime_source()
        now = time.monotonic()
        if sig != last_sig:
            last_sig = sig
            payload = json.dumps(build_dashboard(out_dir))
            msg = f"event: dashboard\ndata: {payload}\n\n"
            wfile.write(msg.encode("utf-8"))
            wfile.flush()
            last_keepalive = now
        elif now - last_keepalive >= keepalive_s:
            wfile.write(b": keepalive\n\n")
            wfile.flush()
            last_keepalive = now
        i += 1
        if max_iterations is not None and i >= max_iterations:
            break
            last_keepalive = now


def _serve_web(out_dir: Path, *, port: int, refresh: float) -> None:
    """Serve the static dashboard page + JSON + SSE over stdlib http.server.

    Routes: ``/`` (HTML page), ``/dashboard.json`` (the 9-key contract),
    ``/questions.json?combo=<id>`` (the flat question list), ``/events``
    (Server-Sent Events: pushes a new dashboard payload when watched files
    change). No new dependency, no JS framework.
    """
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from urllib.parse import parse_qs, urlparse

    page_html = _dashboard_html(refresh)

    class _Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # silence default logging
            pass

        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path == "/dashboard.json":
                payload = json.dumps(build_dashboard(out_dir)).encode("utf-8")
                self._send(200, "application/json", payload)
            elif parsed.path == "/questions.json":
                qs = parse_qs(parsed.query)
                combo = qs.get("combo", [""])[0]
                payload = json.dumps(build_questions(out_dir, combo)).encode("utf-8")
                self._send(200, "application/json", payload)
            elif parsed.path == "/events":
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "keep-alive")
                self.end_headers()
                _handle_sse(out_dir, self.wfile, refresh)
            elif parsed.path == "/":
                self._send(200, "text/html; charset=utf-8", page_html.encode("utf-8"))
            else:
                self._send(404, "text/plain", b"not found")

        def _send(self, code, ctype, payload):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    server = ThreadingHTTPServer(("127.0.0.1", port), _Handler)
    print(f"[watch] web dashboard at http://127.0.0.1:{port}/ (SSE, refresh {int(refresh)}s)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()


def run_watch(
    out_dir: str | Path,
    *,
    refresh: float = 2.0,
    web: bool = False,
    port: int = 8765,
    tty: bool = True,
) -> None:
    """Attach to a bench/m5 output dir and redraw the dashboard in place.

    Terminal mode (default, ``tty=True``): rich ``Live`` redraw, keys q/p.
    Web mode (``web=True``): serve the same dashboard as HTML at
    http://127.0.0.1:PORT/ with auto-refresh, plus /dashboard.json.
    Both can run together; ``tty=False`` runs web only.
    """
    out_dir = Path(out_dir)
    if web and not tty:
        _serve_web(out_dir, port=port, refresh=refresh)
        return
    if web and tty:
        import threading as _th

        _th.Thread(
            target=_serve_web,
            args=(out_dir,),
            kwargs={"port": port, "refresh": refresh},
            daemon=True,
        ).start()
    # Terminal mode.
    paused = False

    def _render():
        try:
            cols = os.get_terminal_size().columns
        except OSError:
            cols = 120
        return build_dashboard_renderable(out_dir, width=cols)

    try:
        with Live(_render(), refresh_per_second=1 / refresh, screen=True) as live:
            while True:
                time.sleep(refresh)
                if not paused:
                    live.update(_render())
    except KeyboardInterrupt:
        pass
