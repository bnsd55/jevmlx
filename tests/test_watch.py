"""W6-UI: tests for the read-only live watcher (no model, fake files)."""

from __future__ import annotations

import json

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
        """render_json returns the raw numbers for scripts."""
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
        assert data["out_dir"] == str(tmp_path)
        assert data["model"] == "m"
        assert data["machine"] == "M2"
        assert isinstance(data["combos"], list)
        assert data["runbook"] == []  # no RUNBOOK.md

    def test_json_empty_dir(self, tmp_path):
        data = render_json(tmp_path)
        assert data["combos"] == []
        assert data["runbook"] == []
