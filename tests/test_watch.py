"""W6-UI: tests for the read-only live watcher (no model, fake files)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from jevmlx.watch import (
    count_lines,
    eta_string,
    live_table_row,
    parse_heartbeat,
    parse_run_json,
    parse_runbook,
    read_jsonl_safe,
    render_html,
    render_json,
)


class TestReadJsonlSafe:
    def test_half_written_last_line_truncated(self, tmp_path):
        """A half-written trailing JSON line is discarded (same rule as resume)."""
        p = tmp_path / "heartbeat.jsonl"
        p.write_text(
            json.dumps({"a": 1}) + "\n" + json.dumps({"a": 2}) + "\n" + '{"a": 3, "broken',
            encoding="utf-8",
        )
        records = read_jsonl_safe(p)
        assert len(records) == 2
        assert records[-1]["a"] == 2

    def test_missing_file_returns_empty(self, tmp_path):
        assert read_jsonl_safe(tmp_path / "nope.jsonl") == []

    def test_max_lines_tails(self, tmp_path):
        p = tmp_path / "f.jsonl"
        p.write_text("\n".join(json.dumps({"i": i}) for i in range(10)) + "\n", encoding="utf-8")
        records = read_jsonl_safe(p, max_lines=3)
        assert len(records) == 3
        assert records[-1]["i"] == 9


class TestParseRunbook:
    def test_steps_parsed_with_state(self, tmp_path):
        (tmp_path / "RUNBOOK.md").write_text(
            "# M5 runbook\n\n"
            "## 1. Doctor — exit 0 — 3.2s — 2026-09-20\n"
            "cmd: jevmlx doctor --json\n\n"
            "## 2. Parity — skip (outputs exist)\n\n"
            "## 3. Bench — exit 1 — 10.0s — 2026-09-20\n"
            "**ABORT: bench failed (exit 1); remaining steps skipped.**\n",
            encoding="utf-8",
        )
        steps = parse_runbook(tmp_path)
        assert len(steps) == 3
        assert steps[0]["state"] == "done"
        assert steps[0]["wall"] == "3.2s"
        assert steps[0]["cmd"] == "jevmlx doctor --json"
        assert steps[1]["state"] == "skipped"
        assert steps[2]["state"] == "aborted"

    def test_no_runbook_returns_empty(self, tmp_path):
        assert parse_runbook(tmp_path) == []


class TestParseHeartbeat:
    def test_last_record_returned(self, tmp_path):
        p = tmp_path / "heartbeat.jsonl"
        p.write_text(
            json.dumps({"cases_done": 2, "elapsed_s": 10})
            + "\n"
            + json.dumps({"cases_done": 4, "elapsed_s": 20})
            + "\n",
            encoding="utf-8",
        )
        hb = parse_heartbeat(tmp_path)
        assert hb is not None
        assert hb["cases_done"] == 4

    def test_missing_returns_none(self, tmp_path):
        assert parse_heartbeat(tmp_path) is None


class TestParseRunJson:
    def test_reads_run_json(self, tmp_path):
        (tmp_path / "run.json").write_text(
            json.dumps({"run_id": "x", "environment": {"chip": "M2"}, "config": {"model": "m"}}),
            encoding="utf-8",
        )
        run = parse_run_json(tmp_path)
        assert run["environment"]["chip"] == "M2"

    def test_missing_returns_empty(self, tmp_path):
        assert parse_run_json(tmp_path) == {}


class TestLiveTableRow:
    def test_yields_exactly_readme_columns(self):
        """The row dict has exactly the keys the README leaderboard table has."""
        records = [
            {
                "case_id": "c1",
                "field": "f",
                "prediction": "A",
                "label": "A",
                "correct": True,
                "latency_ms": 100,
                "workflow": "customer_service",
                "probability": 0.9,
            },
            {
                "case_id": "c2",
                "field": "f",
                "prediction": "B",
                "label": "A",
                "correct": False,
                "latency_ms": 200,
                "workflow": "customer_service",
                "probability": 0.4,
            },
        ]
        row = live_table_row(records, model="m", source="live", scorer="trie", machine="M2")
        # The row has the leaderboard column keys.
        expected_keys = {
            "model",
            "source",
            "scorer",
            "machine",
            "accuracy",
            "accuracy_ci",
            "by_workflow",
            "time_per_case_s",
            "cost_per_case_usd",
            "cases",
        }
        assert set(row.keys()) == expected_keys
        # Accuracy = 1/2 = 0.5.
        assert row["accuracy"] == pytest.approx(0.5)
        # Cases = 2.
        assert row["cases"] == 2
        # Time per case = median(100, 200) / 1000 = 0.15s.
        assert row["time_per_case_s"] == pytest.approx(0.15)
        # Wilson CI is a tuple.
        assert row["accuracy_ci"] is not None
        lo, hi = row["accuracy_ci"]
        assert 0 <= lo <= 0.5 <= hi <= 1

    def test_empty_records(self):
        """No records -> accuracy None, cases 0."""
        row = live_table_row([], model="m")
        assert row["accuracy"] is None
        assert row["cases"] == 0
        assert row["time_per_case_s"] is None

    def test_workflow_columns(self):
        """Per-workflow accuracy fills the four TypeSafe columns."""
        records = [
            {
                "field": "f",
                "prediction": "A",
                "label": "A",
                "correct": True,
                "workflow": "customer_service",
                "probability": 0.9,
            },
            {
                "field": "f",
                "prediction": "A",
                "label": "A",
                "correct": True,
                "workflow": "security_incidents",
                "probability": 0.9,
            },
        ]
        row = live_table_row(records)
        wf = row["by_workflow"]
        assert wf["customer_service"] == pytest.approx(1.0)
        assert wf["security_incidents"] == pytest.approx(1.0)
        # Workflows with no records -> None.
        assert wf["agent_trace_observability"] is None


class TestEtaString:
    def test_basic_eta(self):
        # 10 done, 100 total, 100s elapsed -> rate 0.1/s, 900 left -> 900s = 15m.
        assert eta_string(10, 100, 100) == "15m"

    def test_zero_done(self):
        assert eta_string(0, 100, 0) == "—"

    def test_zero_total(self):
        assert eta_string(10, 0, 100) == "—"

    def test_complete(self):
        assert eta_string(100, 100, 1000) == "done"

    def test_hours(self):
        # 10 done, 1000 total, 100s -> 990 left at 0.1/s = 9900s = 2.75h.
        assert eta_string(10, 1000, 100) == "2.8h"


class TestCountLines:
    def test_counts_lines(self, tmp_path):
        p = tmp_path / "data.jsonl"
        p.write_text('{"a":1}\n{"a":2}\n{"a":3}\n', encoding="utf-8")
        assert count_lines(p) == 3

    def test_missing_returns_zero(self, tmp_path):
        assert count_lines(tmp_path / "nope.jsonl") == 0


class TestUiWiring:
    def test_bench_ui_spawns_watcher_with_out_dir(self, monkeypatch, tmp_path):
        """jevmlx bench --ui spawns the bench subprocess and calls the watcher
        on the out dir. Injects a fake watch_fn so the test NEVER enters the
        live loop or binds a port. No real process is spawned (Popen patched)."""
        import jevmlx.cli as cli

        spawned = []
        watch_called = []

        class _FakeProc:
            def wait(self):
                return 0

        def _fake_popen(*args, **kwargs):
            spawned.append(args[0])
            return _FakeProc()

        def _fake_watch(out_dir, refresh=2.0, web=False, port=8765, tty=True):
            watch_called.append(
                {
                    "out_dir": str(out_dir),
                    "refresh": refresh,
                    "web": web,
                    "port": port,
                    "tty": tty,
                }
            )

        import subprocess as _sp

        monkeypatch.setattr(_sp, "Popen", _fake_popen)
        out_dir = tmp_path / "run"
        cli._run_bench_ui(out_dir, ["--model", "m"], watch_fn=_fake_watch)

        # The bench subprocess was spawned (python -m jevmlx bench ...).
        assert spawned, "bench subprocess was not spawned"
        assert "-m" in spawned[0] and "jevmlx" in spawned[0]
        # The watcher was called with the out dir.
        assert watch_called, "watch_fn was not called"
        assert watch_called[0]["out_dir"] == str(out_dir)
        assert watch_called[0]["refresh"] == 2.0
        # bench.log was created in the out dir.
        assert (out_dir / "bench.log").exists()

    def test_bench_ui_web_passthrough(self, monkeypatch, tmp_path):
        """--ui --web passes web/port/tty through to the watcher."""
        import jevmlx.cli as cli

        watch_called = []

        class _FakeProc:
            def wait(self):
                return 0

        monkeypatch.setattr("subprocess.Popen", lambda *a, **k: _FakeProc())

        def _fake_watch(out_dir, refresh=2.0, web=False, port=8765, tty=True):
            watch_called.append({"web": web, "port": port, "tty": tty})

        cli._run_bench_ui(
            tmp_path / "run",
            ["--model", "m"],
            web=True,
            port=9999,
            tty=False,
            watch_fn=_fake_watch,
        )
        assert watch_called[0]["web"] is True
        assert watch_called[0]["port"] == 9999
        assert watch_called[0]["tty"] is False


class TestRenderHtml:
    def test_html_has_table_headers_and_refresh_meta(self, tmp_path):
        """render_html returns a document with the leaderboard column headers
        and the auto-refresh meta tag. Renders to a string — no server."""
        # Minimal combo dir with one prediction.
        (tmp_path / "predictions.jsonl").write_text(
            json.dumps(
                {
                    "case_id": "c1",
                    "field": "f",
                    "prediction": "A",
                    "label": "A",
                    "correct": True,
                    "latency_ms": 100,
                    "workflow": "customer_service",
                    "probability": 0.9,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        (tmp_path / "run.json").write_text(
            json.dumps({"config": {"model": "m"}, "environment": {"chip": "M2"}}),
            encoding="utf-8",
        )
        html = render_html(tmp_path, refresh=3)
        # The refresh meta tag is present.
        assert '<meta http-equiv="refresh" content="3">' in html
        # The leaderboard column headers are in the rendered table.
        for col in ("Model", "Accuracy", "Cases"):
            assert col in html

    def test_html_empty_dir(self, tmp_path):
        """An empty dir still renders a valid HTML document."""
        html = render_html(tmp_path)
        assert "<html" in html.lower() or "<!DOCTYPE" in html


class TestRenderJson:
    def test_json_has_expected_keys(self, tmp_path):
        """render_json (= build_dashboard) returns the frozen W6-UI-3a contract."""
        (tmp_path / "predictions.jsonl").write_text(
            json.dumps(
                {
                    "case_id": "c1",
                    "field": "f",
                    "prediction": "A",
                    "label": "A",
                    "correct": True,
                    "latency_ms": 100,
                    "workflow": "customer_service",
                    "probability": 0.9,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        (tmp_path / "run.json").write_text(
            json.dumps({"config": {"model": "m"}, "environment": {"chip": "M2"}}),
            encoding="utf-8",
        )
        data = render_json(tmp_path)
        # The 9 frozen top-level keys.
        assert set(data) == {
            "run",
            "now",
            "memory",
            "health",
            "pipeline",
            "aggregates",
            "results",
            "events",
            "history",
        }
        assert data["run"]["out_dir"] == str(tmp_path)
        assert data["run"]["machine"] == "M2"
        assert isinstance(data["results"], list)
        assert len(data["results"]) == 1
        assert data["results"][0]["model"] == "m"

    def test_json_empty_dir(self, tmp_path):
        data = render_json(tmp_path)
        # Empty dir: all 9 keys present, results empty, never raises.
        assert set(data) == {
            "run",
            "now",
            "memory",
            "health",
            "pipeline",
            "aggregates",
            "results",
            "events",
            "history",
        }
        assert data["results"] == []
        assert data["events"] == []


# --- W6-UI-3a: full dashboard data layer (frozen contract) -----------------


class TestDashboardContract:
    """Fixture tree: two attempts, one failed combo, one running combo with
    heartbeat, and a cached dataset. Asserts every key, attempt grouping,
    questions text join, and ETA math."""

    @staticmethod
    def _build_fixture(out: Path) -> None:
        """Build the fixture tree under <out>."""
        # run.json at the top level (m5 run dir).
        (out / "run.json").write_text(
            json.dumps(
                {
                    "run_id": "r1",
                    "environment": {
                        "chip": "M5 Max",
                        "mlx_version": "0.32.2",
                        "git_sha": "dbb1ff1abcdef",
                        "machine_memory_gb": 128,
                    },
                    "config": {
                        "model": "mlx-community/Qwen3-8B-4bit",
                        "memory": {
                            "metal_cache_limit_bytes": 8 * 2**30,
                            "metal_cache_stop_bytes": 10 * 2**30,
                        },
                    },
                    "counts": {"cases": 576},
                }
            ),
            encoding="utf-8",
        )
        # RUNBOOK.md with two attempt headers.
        (out / "RUNBOOK.md").write_text(
            "\n".join(
                [
                    "# M5 runbook — 2026-09-21T00:00:00Z",
                    "sleep_blocked: False",
                    "",
                    "## attempt 1 2026-09-21T00:00:00Z aaa111 python -m benchmarks.m5 --out .",
                    "started doctor 2026-09-21T00:00:01Z",
                    "## 1. doctor — exit 1 — 0.2s — 2026-09-21T00:00:01Z",
                    "",
                    "## attempt 2 2026-09-21T02:17:00Z dbb1ff1 python -m benchmarks.m5 --out .",
                    "started doctor 2026-09-21T02:17:01Z",
                    "## 1. doctor — exit 0 — 0.2s — 2026-09-21T02:17:01Z",
                    "started parity-mlx-community_Qwen3-8B-4bit 2026-09-21T02:18:00Z",
                    "## 2. pytest -m slow (Qwen3-8B) — exit 1 — 28.8s — 2026-09-21T02:18:29Z",
                    "started bench-quality 2026-09-21T02:19:00Z",
                    "## 3. jevmlx bench --model quality — exit 0 — 15600.0s — 2026-09-21T06:39:00Z",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        # A failed combo (run_failed.txt).
        failed_dir = out / "bench-failed"
        failed_dir.mkdir()
        (failed_dir / "run.json").write_text(
            json.dumps(
                {"config": {"model": "gemma-3-12B", "track": "parallel"}, "counts": {"cases": 24}}
            ),
            encoding="utf-8",
        )
        (failed_dir / "run_failed.txt").write_text("crash\n", encoding="utf-8")
        # A running combo with heartbeat.
        running_dir = out / "bench-running"
        running_dir.mkdir()
        (running_dir / "run.json").write_text(
            json.dumps(
                {
                    "config": {"model": "Qwen3-8B", "track": "parallel", "scorer": "labels"},
                    "counts": {"cases": 576},
                }
            ),
            encoding="utf-8",
        )
        (running_dir / "dataset.jsonl").write_text(
            "\n".join(json.dumps({"id": f"c{i}", "context": f"ctx-{i}"}) for i in range(10)) + "\n",
            encoding="utf-8",
        )
        now_ts = __import__("time").time()
        (running_dir / "heartbeat.jsonl").write_text(
            json.dumps(
                {
                    "combo": "bench-running",
                    "cases_done": 270,
                    "pred_lines": 3673,
                    "elapsed_s": 1884,
                    "peak_memory_bytes": 86 * 2**30,
                    "active_memory_bytes": 4 * 2**30,
                    "cache_memory_bytes": 8 * 2**30,
                    "ts": now_ts - 23,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        # predictions.jsonl with a couple of labelled rows.
        (running_dir / "predictions.jsonl").write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "case_id": "c0",
                            "field": "verdict",
                            "prediction": "yes",
                            "label": "yes",
                            "correct": True,
                            "probability": 0.9,
                            "per_option": {"yes": 0.9, "no": 0.1},
                            "permutation": 0,
                            "per_item_end_to_end_ms": 212,
                        }
                    ),
                    json.dumps(
                        {
                            "case_id": "c1",
                            "field": "verdict",
                            "prediction": "no",
                            "label": "yes",
                            "correct": False,
                            "probability": 0.4,
                            "per_option": {"yes": 0.6, "no": 0.4},
                            "permutation": 1,
                            "per_item_end_to_end_ms": 215,
                        }
                    ),
                ]
            )
            + "\n",
            encoding="utf-8",
        )

    def test_dashboard_has_all_nine_top_level_keys(self, tmp_path):
        from jevmlx.watch import build_dashboard

        out = tmp_path / "run"
        out.mkdir()
        self._build_fixture(out)
        d = build_dashboard(out)
        assert set(d) == {
            "run",
            "now",
            "memory",
            "health",
            "pipeline",
            "aggregates",
            "results",
            "events",
            "history",
        }

    def test_run_block_fields(self, tmp_path):
        from jevmlx.watch import build_dashboard

        out = tmp_path / "run"
        out.mkdir()
        self._build_fixture(out)
        run = build_dashboard(out)["run"]
        assert run["out_dir"] == str(out)
        assert run["hash"] == "dbb1ff1abcde"
        assert run["machine"] == "M5 Max"
        assert run["mlx_version"] == "0.32.2"
        assert run["attempt_n"] == 2  # last attempt header
        assert run["attempt_started"] == "2026-09-21T02:17:00Z"
        assert run["sleep_blocked"] is False
        assert run["state"] == "running"  # heartbeat present

    def test_pipeline_groups_by_last_attempt(self, tmp_path):
        from jevmlx.watch import build_dashboard

        out = tmp_path / "run"
        out.mkdir()
        self._build_fixture(out)
        pipeline = build_dashboard(out)["pipeline"]
        assert pipeline["attempt_n"] == 2
        steps = pipeline["steps"]
        # Only attempt 2's steps (3 steps), not attempt 1's.
        ids = [s["id"] for s in steps]
        assert "doctor" in ids
        # Each step has the frozen keys.
        for s in steps:
            assert {
                "id",
                "title",
                "state",
                "wall_s",
                "exit",
                "started",
                "argv",
                "stdout_tail",
                "error",
            } <= set(s)
        # The doctor step completed ok in attempt 2.
        doc = next(s for s in steps if s["id"] == "doctor")
        assert doc["state"] == "ok"
        assert doc["exit"] == 0

    def test_memory_block(self, tmp_path):
        from jevmlx.watch import build_dashboard

        out = tmp_path / "run"
        out.mkdir()
        self._build_fixture(out)
        mem = build_dashboard(out)["memory"]
        assert mem["cache_gb"] == 8.0
        assert mem["active_gb"] == 4.0
        assert mem["peak_gb"] == 86.0
        assert mem["cap_gb"] == 8.0
        assert mem["stop_gb"] == 10.0
        assert mem["machine_gb"] == 128

    def test_health_rules(self, tmp_path):
        from jevmlx.watch import build_dashboard

        out = tmp_path / "run"
        out.mkdir()
        self._build_fixture(out)
        health = build_dashboard(out)["health"]
        rules = {h["rule"]: h for h in health}
        # All six rules present.
        assert set(rules) == {
            "cache_over_stop",
            "sleep_windows",
            "run_failed_combos",
            "parity_fail_models",
            "metal_alloc_retries",
            "heartbeat_age",
        }
        # cache 8 GB == stop 10 GB -> not over, ok.
        assert rules["cache_over_stop"]["state"] == "ok"
        # One run_failed combo.
        assert rules["run_failed_combos"]["state"] == "fail"

    def test_results_one_row_per_combo(self, tmp_path):
        from jevmlx.watch import build_dashboard

        out = tmp_path / "run"
        out.mkdir()
        self._build_fixture(out)
        results = build_dashboard(out)["results"]
        assert len(results) == 3  # top-level + failed + running
        statuses = {r["status"] for r in results}
        assert "failed" in statuses
        assert "running" in statuses
        for r in results:
            assert {
                "model",
                "dataset",
                "scorer",
                "track",
                "status",
                "accuracy",
                "ci_low",
                "ci_high",
                "majority",
                "exact_record",
                "parity_status",
                "parity_drift",
                "time_per_case_s",
                "calls",
                "cases",
                "ab_delta",
            } <= set(r)

    def test_now_block_eta_math(self, tmp_path):
        from jevmlx.watch import build_dashboard

        out = tmp_path / "run"
        out.mkdir()
        self._build_fixture(out)
        now = build_dashboard(out)["now"]
        assert now["model"] == "Qwen3-8B"
        assert now["cases_done"] == 270
        assert now["cases_total"] == 10  # dataset.jsonl has 10 lines
        # ETA = (total - done) / (done / elapsed). done (270) > total (10)
        # because heartbeat cases_done exceeds dataset lines; ETA is None.
        assert now["eta_s"] is None
        assert now["pred_lines"] == 3673

    def test_history_previous_attempts(self, tmp_path):
        from jevmlx.watch import build_dashboard

        out = tmp_path / "run"
        out.mkdir()
        self._build_fixture(out)
        history = build_dashboard(out)["history"]
        # Attempt 1 is previous (attempt 2 is current).
        assert len(history) == 1
        assert history[0]["attempt_n"] == 1
        assert history[0]["hash"] == "aaa111"

    def test_events_newest_first(self, tmp_path):
        from jevmlx.watch import build_dashboard

        out = tmp_path / "run"
        out.mkdir()
        self._build_fixture(out)
        events = build_dashboard(out)["events"]
        assert len(events) > 0
        # Every event has the frozen keys.
        for e in events:
            assert {"ts", "kind", "text"} <= set(e)
        # Kinds are from the allowed set.
        valid = {
            "heartbeat",
            "combo_done",
            "alloc_retry",
            "step_done",
            "attempt",
            "run_failed",
            "sleep",
        }
        assert all(e["kind"] in valid for e in events)

    def test_questions_text_join_from_cached_dataset(self, tmp_path):
        from jevmlx.watch import build_questions

        out = tmp_path / "run"
        out.mkdir()
        self._build_fixture(out)
        qs = build_questions(out, "bench-running")
        assert len(qs) == 2
        # context_text joined from dataset.jsonl by case_id.
        assert qs[0]["context_text"] == "ctx-0"
        assert qs[1]["context_text"] == "ctx-1"
        # options carried from per_option.
        assert {o["name"] for o in qs[0]["options"]} == {"yes", "no"}
        # margin = top1 - top2.
        assert qs[0]["margin"] == 0.8  # 0.9 - 0.1
        # acc_so_far: first correct -> 1.0.
        assert qs[0]["acc_so_far"] == 1.0
        # rotations_same_field: both rows are field 'verdict' but different
        # cases, so each case has 1 rotation.
        assert len(qs[0]["rotations_same_field"]) == 1

    def test_questions_missing_combo_returns_empty(self, tmp_path):
        from jevmlx.watch import build_questions

        out = tmp_path / "run"
        out.mkdir()
        self._build_fixture(out)
        assert build_questions(out, "no-such-combo") == []

    def test_dashboard_never_raises_on_missing_files(self, tmp_path):
        from jevmlx.watch import build_dashboard

        # Completely empty dir — no run.json, no RUNBOOK, nothing.
        d = build_dashboard(tmp_path)
        assert set(d) == {
            "run",
            "now",
            "memory",
            "health",
            "pipeline",
            "aggregates",
            "results",
            "events",
            "history",
        }
        assert d["results"] == []
        assert d["run"]["state"] == "stopped"

    def test_half_written_heartbeat_truncated(self, tmp_path):
        """A half-written trailing heartbeat line is discarded (same rule as resume)."""
        from jevmlx.watch import build_dashboard

        out = tmp_path / "run"
        out.mkdir()
        (out / "run.json").write_text(
            json.dumps({"config": {"model": "m"}, "environment": {"chip": "M2"}}),
            encoding="utf-8",
        )
        combo = out / "bench-x"
        combo.mkdir()
        (combo / "heartbeat.jsonl").write_text(
            json.dumps({"combo": "x", "cases_done": 5, "elapsed_s": 10, "ts": 0})
            + "\n"
            + "{broken half-line",
            encoding="utf-8",
        )
        # Must not raise; the broken line is truncated.
        d = build_dashboard(out)
        assert isinstance(d, dict)
