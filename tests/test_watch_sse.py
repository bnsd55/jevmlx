"""W6-UI-3f: SSE live-update handler tests.

The SSE handler (_handle_sse) watches mtimes of RUNBOOK.md + every
heartbeat.jsonl / run.json / predictions.jsonl under <out>, and emits
'event: dashboard' + the build_dashboard JSON when any change. Every 15 s
without change it writes a ': keepalive' comment. No port bind, no browser.

Tests use the ``mtime_source`` and ``max_iterations`` test hooks (production
ignores them) to drive deterministic change/timeout scenarios without threads.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from jevmlx.watch import _handle_sse, _mtimes_signature, _watched_files


class _FakeWfile:
    """A fake wfile that captures bytes written (no real socket)."""

    def __init__(self):
        self.buf = bytearray()

    def write(self, data: bytes) -> int:
        self.buf.extend(data)
        return len(data)

    def flush(self) -> None:
        pass


def _make_out_dir(tmp_path: Path) -> Path:
    """Create a minimal out_dir with a RUNBOOK.md and one combo dir."""
    out = tmp_path / "run"
    out.mkdir()
    (out / "RUNBOOK.md").write_text("# run\n", encoding="utf-8")
    (out / "run.json").write_text(
        json.dumps(
            {
                "config": {"model": "M", "dataset": "D", "scorer": "S", "track": "parallel"},
                "environment": {"chip": "M2", "machine": "mac"},
            }
        ),
        encoding="utf-8",
    )
    combo = out / "bench-quality" / "mac-M" / "parallel-S-D"
    combo.mkdir(parents=True)
    (combo / "heartbeat.jsonl").write_text(
        json.dumps({"ts": "2026-01-01T00:00:00Z", "phase": "run"}) + "\n", encoding="utf-8"
    )
    (combo / "predictions.jsonl").write_text(
        json.dumps({"case_id": "c1", "field": "f", "prediction": "A", "label": "A"}) + "\n",
        encoding="utf-8",
    )
    return out


class TestWatchedFiles:
    """_watched_files discovers RUNBOOK.md + heartbeat.jsonl + run.json + predictions.jsonl."""

    def test_finds_runbook_and_combo_files(self, tmp_path):
        out = _make_out_dir(tmp_path)
        files = _watched_files(out)
        names = [f.name for f in files]
        assert "RUNBOOK.md" in names
        assert "heartbeat.jsonl" in names
        assert "run.json" in names
        assert "predictions.jsonl" in names

    def test_empty_dir(self, tmp_path):
        assert _watched_files(tmp_path) == []


class TestMtimesSignature:
    """_mtimes_signature returns a tuple that changes when a file's mtime changes."""

    def test_signature_changes_on_mtime_update(self, tmp_path):
        out = _make_out_dir(tmp_path)
        files = _watched_files(out)
        sig1 = _mtimes_signature(files)
        # Bump the mtime of heartbeat.jsonl forward.
        hb = [f for f in files if f.name == "heartbeat.jsonl"][0]
        st = hb.stat()
        os.utime(hb, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))
        sig2 = _mtimes_signature(files)
        assert sig1 != sig2

    def test_signature_stable_when_unchanged(self, tmp_path):
        out = _make_out_dir(tmp_path)
        files = _watched_files(out)
        assert _mtimes_signature(files) == _mtimes_signature(files)


class TestSseHandler:
    """_handle_sse emits 'event: dashboard' on mtime change, ': keepalive' on timeout."""

    def test_emits_dashboard_event_on_change(self, tmp_path, monkeypatch):
        out = _make_out_dir(tmp_path)
        refresh = 0.01
        # Make time.sleep a no-op so the loop spins instantly.
        monkeypatch.setattr(time, "sleep", lambda s: None)

        # Drive mtime changes deterministically: signature changes on call 2.
        real_sig = _mtimes_signature(_watched_files(out))
        calls = [0]

        def mtime_source():
            calls[0] += 1
            return real_sig if calls[0] == 1 else real_sig + (("changed", 999),)

        wfile = _FakeWfile()
        _handle_sse(out, wfile, refresh, mtime_source=mtime_source, max_iterations=1)

        output = wfile.buf.decode("utf-8")
        assert "event: dashboard" in output
        assert "data: " in output
        data_line = [ln for ln in output.splitlines() if ln.startswith("data: ")][0]
        payload = json.loads(data_line[len("data: ") :])
        assert set(payload.keys()) == {
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

    def test_emits_keepalive_on_timeout(self, tmp_path, monkeypatch):
        out = _make_out_dir(tmp_path)
        refresh = 0.01
        # Make time.sleep a no-op, and make monotonic jump forward 16s per
        # call so the 15s keepalive threshold is crossed immediately.
        call_count = [0]

        def fast_sleep(s):
            pass

        def fast_monotonic():
            call_count[0] += 1
            return call_count[0] * 16.0

        monkeypatch.setattr(time, "sleep", fast_sleep)
        monkeypatch.setattr("jevmlx.watch.time.monotonic", fast_monotonic)

        # Signature never changes -> keepalive path.
        real_sig = _mtimes_signature(_watched_files(out))

        def mtime_source():
            return real_sig

        wfile = _FakeWfile()
        _handle_sse(out, wfile, refresh, mtime_source=mtime_source, max_iterations=1)

        output = wfile.buf.decode("utf-8")
        assert ": keepalive" in output
        assert "event: dashboard" not in output

    def test_no_output_when_no_change_and_no_timeout(self, tmp_path, monkeypatch):
        out = _make_out_dir(tmp_path)
        refresh = 0.01
        monkeypatch.setattr(time, "sleep", lambda s: None)
        # monotonic returns 0 always -> never crosses 15s keepalive.
        monkeypatch.setattr("jevmlx.watch.time.monotonic", lambda: 0.0)
        real_sig = _mtimes_signature(_watched_files(out))

        def mtime_source():
            return real_sig

        wfile = _FakeWfile()
        _handle_sse(out, wfile, refresh, mtime_source=mtime_source, max_iterations=3)
        assert len(wfile.buf) == 0
