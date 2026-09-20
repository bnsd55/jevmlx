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
    "parse_runbook",
    "parse_run_json",
    "parse_heartbeat",
    "count_lines",
    "live_table_row",
    "eta_string",
    "build_dashboard",
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
    """All combo dirs under <out> (the out dir itself when it holds predictions).

    A single-combo bench writes run.json/predictions.jsonl directly into the
    out dir; a multi-combo bench writes them into subdirectories.
    """
    combos: list[Path] = []
    if (out_dir / "run.json").exists() or (out_dir / "predictions.jsonl").exists():
        combos.append(out_dir)
    for child in sorted(out_dir.iterdir() if out_dir.exists() else []):
        if not child.is_dir() or child.name.startswith(".") or child.name in ("ab", "invariance"):
            continue
        # A combo dir has run.json or predictions.jsonl.
        if (child / "run.json").exists() or (child / "predictions.jsonl").exists():
            combos.append(child)
        else:
            # Maybe nested: <out>/<machine-model>/<combo>/
            for sub in sorted(child.iterdir() if child.is_dir() else []):
                if sub.is_dir() and (
                    (sub / "run.json").exists() or (sub / "predictions.jsonl").exists()
                ):
                    combos.append(sub)
    return combos


def build_dashboard(out_dir: Path, *, width: int = 120) -> Group:
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
        records = read_jsonl_safe(combo_dir / "predictions.jsonl")
        if not records:
            continue
        combo_run = parse_run_json(combo_dir)
        cfg = combo_run.get("config", {}) if combo_run else {}
        row = live_table_row(
            records,
            model=cfg.get("model") or "—",
            source="live",
            scorer=cfg.get("scorer") or "—",
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
        tail = read_jsonl_safe(combo_dir / "predictions.jsonl", max_lines=8)
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
    console.print(build_dashboard(Path(out_dir), width=width))
    body = console.export_html(inline_styles=True)
    # Inject the auto-refresh meta tag (rich's HTML lacks it).
    meta = f'<meta http-equiv="refresh" content="{int(max(1, refresh))}">'
    body = body.replace("<head>", f"<head>{meta}", 1)
    return body


def render_json(out_dir: str | Path) -> dict:
    """The raw dashboard numbers for scripts (the /dashboard.json payload).

    Keys: out_dir, elapsed_s, machine, mlx_version, model, freeze,
    runbook (list of step dicts), combos (list with progress + row dict).
    """
    out_dir = Path(out_dir)
    run = parse_run_json(out_dir)
    env = run.get("environment", {}) if run else {}
    cfg = run.get("config", {}) if run else {}
    hb = parse_heartbeat(out_dir)
    elapsed_s = (
        hb.get("elapsed_s", 0) if hb and isinstance(hb.get("elapsed_s"), (int, float)) else 0
    )
    combos_out: list[dict] = []
    for combo_dir in _combo_dirs(out_dir):
        records = read_jsonl_safe(combo_dir / "predictions.jsonl")
        combo_run = parse_run_json(combo_dir)
        combo_cfg = combo_run.get("config", {}) if combo_run else {}
        total = count_lines(combo_dir / "dataset.jsonl") or count_lines(out_dir / "dataset.jsonl")
        done = len(read_jsonl_safe(combo_dir / "completed_cases.jsonl"))
        combo_hb = parse_heartbeat(combo_dir)
        if combo_hb and isinstance(combo_hb.get("cases_done"), int):
            done = max(done, combo_hb["cases_done"])
        row = (
            live_table_row(
                records,
                model=combo_cfg.get("model") or "—",
                source="live",
                scorer=combo_cfg.get("scorer") or "—",
                machine=env.get("chip") or "—",
            )
            if records
            else None
        )
        combos_out.append(
            {
                "name": combo_dir.name,
                "done": done,
                "total": total,
                "progress": (done / total) if total else 0,
                "heartbeat": combo_hb,
                "row": row,
            }
        )
    return {
        "out_dir": str(out_dir),
        "elapsed_s": elapsed_s,
        "machine": env.get("chip") or "—",
        "mlx_version": env.get("mlx_version") or "—",
        "model": cfg.get("model") or run.get("model") or "—",
        "freeze": (env.get("git_sha") or "—")[:12],
        "runbook": parse_runbook(out_dir),
        "combos": combos_out,
    }


def _serve_web(out_dir: Path, *, port: int, refresh: float) -> None:
    """Serve the dashboard + JSON over stdlib http.server (no new dep)."""
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class _Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # silence default logging
            pass

        def do_GET(self):
            if self.path == "/dashboard.json":
                payload = json.dumps(render_json(out_dir)).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
            else:
                html = render_html(out_dir, refresh=refresh)
                payload = html.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

    server = ThreadingHTTPServer(("127.0.0.1", port), _Handler)
    print(f"[watch] web dashboard at http://127.0.0.1:{port}/ (refresh {int(refresh)}s)")
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
        return build_dashboard(out_dir, width=cols)

    try:
        with Live(_render(), refresh_per_second=1 / refresh, screen=True) as live:
            while True:
                time.sleep(refresh)
                if not paused:
                    live.update(_render())
    except KeyboardInterrupt:
        pass
