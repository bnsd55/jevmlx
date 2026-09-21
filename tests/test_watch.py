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
        """Build a realistic fixture tree under <out>.

        Combo dirs named <track>-<scorer>-<dataset> like bench.py writes;
        two heartbeats 5 min apart so cases_per_h and eta_s are numbers;
        run.json with runs/run index; a parity.json with status; one step log.
        """
        import time as _time

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
        # RUNBOOK.md with two attempt headers + cmd lines.
        (out / "RUNBOOK.md").write_text(
            "\n".join(
                [
                    "# M5 runbook — 2026-09-21T00:00:00Z",
                    "sleep_blocked: True",
                    "",
                    "## attempt 1 2026-09-21T00:00:00Z aaa111 python -m benchmarks.m5 --out .",
                    "started doctor 2026-09-21T00:00:01Z",
                    "## 1. doctor — exit 1 — 0.2s — 2026-09-21T00:00:01Z",
                    "",
                    "## attempt 2 2026-09-21T02:17:00Z dbb1ff1 python -m benchmarks.m5 --out .",
                    "started doctor 2026-09-21T02:17:01Z",
                    "## 1. doctor — exit 0 — 0.2s — 2026-09-21T02:17:01Z",
                    "step: doctor",
                    "cmd: .venv/bin/jevmlx doctor --json",
                    "started parity-mlx-community_Qwen3-8B-4bit 2026-09-21T02:18:00Z",
                    "## 2. pytest -m slow (Qwen3-8B) — exit 1 — 28.8s — 2026-09-21T02:18:29Z",
                    "step: parity-mlx-community_Qwen3-8B-4bit",
                    "cmd: .venv/bin/pytest -m slow -q",
                    "started bench-quality 2026-09-21T02:19:00Z",
                    "## 3. jevmlx bench --model quality — exit 0 — 15600.0s — 2026-09-21T06:39:00Z",
                    "step: bench-quality",
                    "cmd: .venv/bin/jevmlx bench "
                    "--model mlx-community/Qwen2.5-7B-Instruct-4bit --out bench-quality",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        # A step log with an error (the failed slow test).
        (out / "parity-mlx-community_Qwen3-8B-4bit.log").write_text(
            "\n".join(
                [
                    "============================= test session starts "
                    "=============================",
                    "collected 30 items",
                    "tests/test_engine.py::test_chunking_matches_full_batch PASSED",
                    "tests/test_engine.py::test_w1a_scoring_parity FAILED",
                    "",
                    "=================================== FAILURES "
                    "===================================",
                    "FAILED tests/test_engine.py::test_w1a_scoring_parity"
                    "_batch_vs_chunked_real_model",
                    "assert winners identical across chunk sizes",
                    "field counterparty_jurisdiction_risk: chunked='SAN' full='TIER_3'",
                    "========================= 1 failed, 28 passed in 27.5s "
                    "=========================",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        # A failed combo (run_failed.txt), named like bench.py writes.
        failed_dir = out / "parallel-labels-bundled-gemma"
        failed_dir.mkdir()
        (failed_dir / "run.json").write_text(
            json.dumps(
                {
                    "config": {"model": "gemma-3-12B", "track": "parallel", "scorer": "labels"},
                    "counts": {"cases": 24},
                }
            ),
            encoding="utf-8",
        )
        (failed_dir / "run_failed.txt").write_text("crash\n", encoding="utf-8")
        # A parity.json with FAIL status (model-level).
        (out / "parity.json").write_text(
            json.dumps(
                {
                    "model": "Qwen3-8B",
                    "passed": False,
                    "status": "FAIL",
                    "max_abs_drift_nats": 0.08,
                    "winners_identical": False,
                    "atol": 0.05,
                }
            ),
            encoding="utf-8",
        )
        # A running combo with two heartbeats 5 min apart, named like bench.py.
        running_dir = out / "parallel-labels-typesafe-qwen3"
        running_dir.mkdir()
        (running_dir / "run.json").write_text(
            json.dumps(
                {
                    "config": {
                        "model": "Qwen3-8B",
                        "track": "parallel",
                        "scorer": "labels",
                        "dataset": "typesafe",
                        "run_i": 1,
                        "run_n": 2,
                    },
                    "counts": {"cases": 576},
                }
            ),
            encoding="utf-8",
        )
        # dataset.jsonl with 576 lines (matching counts.cases).
        (running_dir / "dataset.jsonl").write_text(
            "\n".join(json.dumps({"id": f"c{i}", "context": f"ctx-{i}"}) for i in range(576))
            + "\n",
            encoding="utf-8",
        )
        now_ts = _time.time()
        # Two heartbeats 5 min (300s) apart: cases_done 200 then 270.
        (running_dir / "heartbeat.jsonl").write_text(
            json.dumps(
                {
                    "combo": "parallel-labels-typesafe-qwen3",
                    "cases_done": 200,
                    "pred_lines": 2700,
                    "elapsed_s": 1584,
                    "peak_memory_bytes": 86 * 2**30,
                    "active_memory_bytes": 4 * 2**30,
                    "cache_memory_bytes": 8 * 2**30,
                    "ts": now_ts - 300,
                }
            )
            + "\n"
            + json.dumps(
                {
                    "combo": "parallel-labels-typesafe-qwen3",
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
        # predictions.jsonl with a couple of labelled rows (same case, two rotations).
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
                            "case_id": "c0",
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
        assert run["sleep_blocked"] is True  # caffeinated (guard on)
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
        assert now["cases_total"] == 576  # manifest counts.cases
        assert now["run_i"] == 1
        assert now["run_n"] == 2
        # ETA = (total - done) / (done / elapsed) = (576-270)/(270/1884).
        assert now["eta_s"] is not None and now["eta_s"] > 0
        assert now["pred_lines"] == 3673
        # cases_per_h from two heartbeats 5 min apart (200->270 in 300s).
        assert now["cases_per_h"] is not None and now["cases_per_h"] > 0

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
        qs = build_questions(out, "parallel-labels-typesafe-qwen3")
        assert len(qs) == 2
        # context_text joined from dataset.jsonl by case_id.
        assert qs[0]["context_text"] == "ctx-0"
        assert qs[1]["context_text"] == "ctx-0"  # same case, rotation 1
        # options carried from per_option.
        assert {o["name"] for o in qs[0]["options"]} == {"yes", "no"}
        # margin = top1 - top2.
        assert qs[0]["margin"] == 0.8  # 0.9 - 0.1
        # acc_so_far: first correct -> 1.0.
        assert qs[0]["acc_so_far"] == 1.0
        # rotations_same_field: both rows are case c0 field 'verdict' (2 rotations).
        assert len(qs[0]["rotations_same_field"]) == 2

    def test_questions_missing_combo_returns_empty(self, tmp_path):
        from jevmlx.watch import build_questions

        out = tmp_path / "run"
        out.mkdir()
        self._build_fixture(out)
        assert build_questions(out, "no-such-combo") == []

    def test_sleep_flag_agrees_between_run_and_health(self, tmp_path):
        """Fix 1: run.sleep_blocked and the sleep_windows health rule read the
        same RUNBOOK line, so they can never disagree."""
        from jevmlx.watch import _read_sleep_blocked, build_dashboard

        out = tmp_path / "run"
        out.mkdir()
        self._build_fixture(out)
        d = build_dashboard(out)
        sleep_flag = _read_sleep_blocked(out)
        assert d["run"]["sleep_blocked"] is sleep_flag
        sleep_rule = next(h for h in d["health"] if h["rule"] == "sleep_windows")
        # sleep_blocked=True (caffeinated) -> ok state, detail caffeinated.
        assert sleep_rule["state"] == "ok"
        assert "caffeinated" in sleep_rule["detail"]

    def test_event_timestamps_are_iso_strings(self, tmp_path):
        """Fix 2: every event ts is an ISO-8601 UTC string, never an epoch float."""
        from jevmlx.watch import build_dashboard

        out = tmp_path / "run"
        out.mkdir()
        self._build_fixture(out)
        events = build_dashboard(out)["events"]
        assert len(events) > 0
        for e in events:
            ts = e.get("ts")
            if ts is not None:
                assert isinstance(ts, str), f"ts is {type(ts).__name__}, not str"
                # ISO-8601 UTC ends with Z.
                assert ts.endswith("Z"), f"ts {ts!r} is not ISO-8601 UTC"

    def test_pipeline_step_titles_are_human_text(self, tmp_path):
        """Fix 4: step titles are the m5 Step.title text ('doctor', 'pytest ...'),
        not raw argv; the leading 'N. ' index is stripped."""
        from jevmlx.watch import build_dashboard

        out = tmp_path / "run"
        out.mkdir()
        self._build_fixture(out)
        steps = build_dashboard(out)["pipeline"]["steps"]
        titles = [s["title"] for s in steps]
        # No leading index prefix like '1. ' or '2. '.
        for t in titles:
            assert not t.split()[0].rstrip(".").isdigit(), f"title {t!r} has index prefix"
        # The doctor step title is 'doctor' (from Step.title='doctor (gate)').
        assert any("doctor" in t for t in titles)

    def test_pipeline_step_has_argv_stdout_tail_and_error(self, tmp_path):
        """Fix 5: the failed step has argv (from cmd:), stdout_tail (last 40
        lines of <step.id>.log), and error (the FAILED block)."""
        from jevmlx.watch import build_dashboard

        out = tmp_path / "run"
        out.mkdir()
        self._build_fixture(out)
        steps = build_dashboard(out)["pipeline"]["steps"]
        parity_step = next(s for s in steps if "parity" in s["id"])
        assert parity_step["state"] == "failed"
        assert parity_step["exit"] == 1
        assert parity_step["argv"]  # cmd: line captured
        assert parity_step["stdout_tail"] is not None
        assert "FAILED" in parity_step["stdout_tail"]
        assert parity_step["error"] is not None
        assert "winners" in parity_step["error"] or "FAILED" in parity_step["error"]

    def test_results_combo_names_match_bench_convention(self, tmp_path):
        """Fix 6: combo dirs are named <track>-<scorer>-<dataset> like bench.py."""
        from jevmlx.watch import build_dashboard

        out = tmp_path / "run"
        out.mkdir()
        self._build_fixture(out)
        # The running combo dir name is parallel-labels-typesafe-qwen3.
        d = build_dashboard(out)
        # results has a row for it.
        assert any(r["track"] == "parallel" for r in d["results"])

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


# --- W6-UI-3d: discover combos from the real bench/m5 layout ---------------


class TestRealBenchLayout:
    """Fixture built by the REAL writers (run_bench_models with a fake engine),
    not hand-made directories. Asserts the data layer finds combos at the real
    depth: <out>/<machine>-<slug>/<track>-<scorer>-<dataset>/run.json."""

    @staticmethod
    def _patch_bench_with_heartbeat(monkeypatch, tmp_path):
        """Patch bench core with a fake _run_one that writes a real run.json
        (config + counts) and a heartbeat.jsonl, so the watcher can discover
        combos, derive model/track/scorer/dataset from config, and find the
        live combo by heartbeat mtime."""
        import time as _time

        from jevmlx import bench

        def fake_run_one(
            model,
            track,
            scorer,
            jsonl,
            combo_dir,
            dataset_lock_path=None,
            resume=False,
            heartbeat_every=0,
            combo="",
            run_i=None,
            run_n=None,
        ):
            combo_dir.mkdir(parents=True, exist_ok=True)
            (combo_dir / "run.json").write_text(
                json.dumps(
                    {
                        "run_id": f"r-{model}-{combo}",
                        "environment": {"chip": "fake-8gb"},
                        "config": {
                            "model": model,
                            "track": track,
                            "scorer": scorer,
                            "dataset": "bundled",
                            "run_i": run_i or 1,
                            "run_n": run_n or 1,
                        },
                        "counts": {"cases": 24},
                    }
                ),
                encoding="utf-8",
            )
            # A heartbeat (the live combo gets a fresh one).
            (combo_dir / "heartbeat.jsonl").write_text(
                json.dumps(
                    {
                        "combo": combo,
                        "cases_done": 12,
                        "pred_lines": 144,
                        "elapsed_s": 60,
                        "peak_memory_bytes": 4 * 2**30,
                        "active_memory_bytes": 2 * 2**30,
                        "cache_memory_bytes": 1 * 2**30,
                        "ts": _time.time(),
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (combo_dir / "predictions.jsonl").write_text(
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
                )
                + "\n",
                encoding="utf-8",
            )
            (combo_dir / "report.json").write_text(
                json.dumps(
                    {
                        "environment": {"chip": "fake-8gb"},
                        "metrics": {"accuracy": 0.9, "n_cases": 24, "latency_ms_p50": 1.0},
                    }
                ),
                encoding="utf-8",
            )
            return {"run": {"model": model}}

        monkeypatch.setattr(bench, "preflight", lambda force, machine_override: "fake-8gb")
        monkeypatch.setattr(
            bench,
            "build_datasets",
            lambda datasets: (
                {"bundled": tmp_path / "b.jsonl"},
                {"bundled": tmp_path / "b.dataset.lock.json"},
            ),
        )
        monkeypatch.setattr(
            bench, "_load_engine_with_timeout", lambda model, timeout: (object(), object())
        )
        monkeypatch.setattr(bench, "_run_one", fake_run_one)
        monkeypatch.setattr(bench, "_print_pr_instructions", lambda folder, last_run: None)
        monkeypatch.setattr(bench, "BENCH_CACHE", tmp_path / "cache")

    def test_dashboard_finds_combos_in_real_bench_layout(self, tmp_path, monkeypatch):
        """The data layer discovers combos by rglob for run.json, finding the
        real depth: <out>/bench-quality/<machine>-<slug>/<combo>/run.json."""
        from jevmlx import bench
        from jevmlx.watch import _combo_dirs

        self._patch_bench_with_heartbeat(monkeypatch, tmp_path)
        out = tmp_path / "m5-run"
        # m5 writes to <out>/bench-quality and <out>/bench-rest.
        bench.run_bench_models(
            models=["org/model-a"],
            datasets=["bundled"],
            scorers=["slots"],
            tracks=["parallel"],
            out=out / "bench-quality",
            runs=1,
        )
        bench.run_bench_models(
            models=["org/model-b"],
            datasets=["bundled"],
            scorers=["labels"],
            tracks=["parallel"],
            out=out / "bench-rest",
            runs=1,
        )
        # The combo dirs are at depth 3 under <out>.
        combos = _combo_dirs(out)
        assert len(combos) == 2
        # run.json exists in each.
        for c in combos:
            assert (c / "run.json").exists()

    def test_results_have_one_row_per_real_combo(self, tmp_path, monkeypatch):
        """results[] has one row per combo, with model/track/scorer derived
        from run.json config (not folder names)."""
        from jevmlx import bench
        from jevmlx.watch import build_dashboard

        self._patch_bench_with_heartbeat(monkeypatch, tmp_path)
        out = tmp_path / "m5-run"
        bench.run_bench_models(
            models=["org/model-a", "org/model-b"],
            datasets=["bundled"],
            scorers=["slots"],
            tracks=["parallel"],
            out=out / "bench-quality",
            runs=1,
        )
        d = build_dashboard(out)
        results = d["results"]
        assert len(results) == 2  # one per model
        models = {r["model"] for r in results}
        assert models == {"org/model-a", "org/model-b"}
        # track/scorer derived from config, not folder names.
        for r in results:
            assert r["track"] == "parallel"
            assert r["scorer"] == "slots"

    def test_now_picks_live_combo_by_heartbeat_mtime(self, tmp_path, monkeypatch):
        """NOW picks the combo with the newest heartbeat.jsonl mtime."""
        from jevmlx import bench
        from jevmlx.watch import build_dashboard

        self._patch_bench_with_heartbeat(monkeypatch, tmp_path)
        out = tmp_path / "m5-run"
        bench.run_bench_models(
            models=["org/model-a", "org/model-b"],
            datasets=["bundled"],
            scorers=["slots"],
            tracks=["parallel"],
            out=out / "bench-quality",
            runs=1,
        )
        # Touch model-b's heartbeat to make it the newest.
        b_hb = list((out / "bench-quality").rglob("heartbeat.jsonl"))
        for hb in b_hb:
            if "model-b" in str(hb):
                hb.touch()
        d = build_dashboard(out)
        assert d["now"]["model"] == "org/model-b"
        assert d["now"]["cases_done"] == 12
        assert d["run"]["state"] == "running"

    def test_questions_returns_rows_for_real_combo(self, tmp_path, monkeypatch):
        """questions.json returns rows for a combo discovered at real depth."""
        from jevmlx import bench
        from jevmlx.watch import _combo_dirs, build_questions

        self._patch_bench_with_heartbeat(monkeypatch, tmp_path)
        out = tmp_path / "m5-run"
        bench.run_bench_models(
            models=["org/model-a"],
            datasets=["bundled"],
            scorers=["slots"],
            tracks=["parallel"],
            out=out / "bench-quality",
            runs=1,
        )
        combos = _combo_dirs(out)
        assert len(combos) == 1
        combo_id = combos[0].name
        qs = build_questions(out, combo_id)
        assert len(qs) == 1
        assert qs[0]["case_id"] == "c0"
        assert qs[0]["predicted"] == "yes"

    def test_live_combo_without_run_json_is_discovered(self, tmp_path, monkeypatch):
        """A LIVE combo dir has heartbeat.jsonl + predictions.jsonl but NO
        run.json yet. _combo_dirs discovers it by any marker file."""
        from jevmlx.watch import _combo_dirs, build_dashboard

        out = tmp_path / "m5-run"
        # The real tree: bench-quality/<machine>-<model>/<combo>/
        combo = out / "bench-quality" / "m5max-128gb-qwen3" / "parallel-labels-typesafe"
        combo.mkdir(parents=True)
        # NO run.json — the combo is still running.
        (combo / "heartbeat.jsonl").write_text(
            json.dumps(
                {
                    "active_memory_bytes": 4 * 2**30,
                    "cache_memory_bytes": 8 * 2**30,
                    "cases_done": 12,
                    "combo": "parallel-labels-typesafe",
                    "elapsed_s": 60,
                    "peak_memory_bytes": 6 * 2**30,
                    "pred_lines": 144,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        (combo / "predictions.jsonl").write_text(
            json.dumps(
                {
                    "case_id": "c0",
                    "field": "v",
                    "prediction": "yes",
                    "label": "yes",
                    "correct": True,
                    "probability": 0.9,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        combos = _combo_dirs(out)
        assert combo in combos
        # build_dashboard does not raise; NOW picks it.
        d = build_dashboard(out)
        assert d["now"]["cases_done"] == 12
        assert d["now"]["track"] == "parallel"  # from folder name

    def test_heartbeat_has_no_ts_age_uses_file_mtime(self, tmp_path, monkeypatch):
        """Heartbeat records carry NO ts key. Age = heartbeat.jsonl file mtime."""
        import time as _time

        from jevmlx.watch import _heartbeat_is_recent, build_dashboard

        out = tmp_path / "m5-run"
        combo = out / "bench-quality" / "m5max-128gb-qwen3" / "parallel-labels-typesafe"
        combo.mkdir(parents=True)
        hb_path = combo / "heartbeat.jsonl"
        hb_path.write_text(
            json.dumps(
                {
                    "cases_done": 5,
                    "elapsed_s": 10,
                    "combo": "x",
                    "cache_memory_bytes": 0,
                    "peak_memory_bytes": 0,
                    "active_memory_bytes": 0,
                    "pred_lines": 50,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        # The heartbeat is fresh (just written).
        assert _heartbeat_is_recent(hb_path, max_age_s=6)
        # Make it stale.
        import os

        old_ts = _time.time() - 300
        os.utime(hb_path, (old_ts, old_ts))
        assert not _heartbeat_is_recent(hb_path, max_age_s=6)
        # build_dashboard: heartbeat_age_s is the file age.
        d = build_dashboard(out)
        assert d["now"]["heartbeat_age_s"] is not None
        assert d["now"]["heartbeat_age_s"] >= 290

    def test_cases_per_h_from_elapsed_s_delta(self, tmp_path, monkeypatch):
        """cases_per_h = (cases_done delta) / (elapsed_s delta) across the last
        two heartbeat lines. No ts key needed."""
        from jevmlx.watch import build_dashboard

        out = tmp_path / "m5-run"
        combo = out / "bench-quality" / "m5max-128gb-qwen3" / "parallel-labels-typesafe"
        combo.mkdir(parents=True)
        (combo / "heartbeat.jsonl").write_text(
            json.dumps(
                {
                    "cases_done": 100,
                    "elapsed_s": 100,
                    "combo": "x",
                    "cache_memory_bytes": 0,
                    "peak_memory_bytes": 0,
                    "active_memory_bytes": 0,
                    "pred_lines": 1200,
                }
            )
            + "\n"
            + json.dumps(
                {
                    "cases_done": 160,
                    "elapsed_s": 200,
                    "combo": "x",
                    "cache_memory_bytes": 0,
                    "peak_memory_bytes": 0,
                    "active_memory_bytes": 0,
                    "pred_lines": 1920,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        (combo / "run.json").write_text(
            json.dumps({"config": {"model": "m", "track": "parallel"}, "counts": {"cases": 576}}),
            encoding="utf-8",
        )
        d = build_dashboard(out)
        # (160-100)/(200-100)*3600 = 60/100*3600 = 2160 cases/h.
        assert d["now"]["cases_per_h"] == 2160.0

    def test_event_ts_is_iso_from_heartbeat_mtime(self, tmp_path, monkeypatch):
        """Event ts is ISO-8601 from the heartbeat.jsonl file mtime (no ts key
        in the record)."""
        from jevmlx.watch import build_dashboard

        out = tmp_path / "m5-run"
        combo = out / "bench-quality" / "m5max-128gb-qwen3" / "parallel-labels-typesafe"
        combo.mkdir(parents=True)
        (combo / "heartbeat.jsonl").write_text(
            json.dumps(
                {
                    "cases_done": 5,
                    "elapsed_s": 10,
                    "combo": "x",
                    "cache_memory_bytes": 0,
                    "peak_memory_bytes": 0,
                    "active_memory_bytes": 0,
                    "pred_lines": 50,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        d = build_dashboard(out)
        hb_events = [e for e in d["events"] if e["kind"] == "heartbeat"]
        assert len(hb_events) >= 1
        for e in hb_events:
            ts = e.get("ts")
            if ts is not None:
                assert isinstance(ts, str)
                assert ts.endswith("Z")
