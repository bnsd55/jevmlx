"""W5b-9: CLI end-to-end smoke tests — every subcommand through `cli.main`.

The engine is the conftest fake: `load_engine` is monkeypatched (via the
cli `_engine()` seam) so no test downloads or loads a real model. The
OpenAI backend runs against a loopback http.server (no network). Error
paths must exit non-zero with ONE line on stderr — no traceback.
"""

from __future__ import annotations

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
from conftest import FakeModel, FakeTokenizer, make_engine, make_engine_result, make_field_telemetry

from jevmlx import cli


@pytest.fixture()
def fake_engine(monkeypatch):
    """cli's lazy engine seam -> (FakeModel, FakeTokenizer) + a
    run_parallel_generation stub returning the full contract shape; records
    the model ids `load_engine` was called with."""
    calls: list[str] = []

    def fake_load_engine(model_id: str):
        calls.append(model_id)
        return make_engine(FakeModel(), FakeTokenizer(), model_id=model_id)

    def fake_rpg(engine, context, schema, **kwargs):
        # The real engine validates the calibration payload before any model
        # work (_load_calibration); the stub mirrors that contract so a bad
        # --calibration FILE fails the same way it does live. (Inline dicts
        # are the Python-API path; the CLI flag is a file path by contract.)
        if kwargs.get("calibration") is not None:
            from jevmlx.engine import _load_calibration

            _load_calibration(kwargs["calibration"])
        return make_engine_result(
            fields={
                "action": make_field_telemetry(
                    value="APPROVE", choices=["APPROVE", "DENY"], probability=0.9
                ),
                "urgent": make_field_telemetry(
                    value=True, type_="boolean", choices=["true", "false"], probability=0.8
                ),
                "is_fraudulent": make_field_telemetry(
                    value=False, type_="boolean", choices=["true", "false"], probability=0.99
                ),
            },
            parsed={
                "action": {"value": "APPROVE", "prob": 0.9},
                "urgent": {"value": True, "prob": 0.8},
                "is_fraudulent": {"value": False, "prob": 0.99},
            },
        )

    monkeypatch.setattr(cli, "_engine", lambda: (fake_load_engine, fake_rpg))
    # serve() imports engine.load_engine directly (F6: no cli dependency);
    # patch the engine module's names so the serve test fakes the same way.
    import jevmlx.engine as engine_mod

    monkeypatch.setattr(engine_mod, "load_engine", fake_load_engine)
    monkeypatch.setattr(engine_mod, "run_parallel_generation", fake_rpg)
    # Some tests purge jevmlx.* submodules from sys.modules
    # (test_imports_without_mlx); pin the patched modules so later
    # `from jevmlx.engine import ...` resolves to IT, not a freshly
    # re-imported module with the real engine.
    monkeypatch.setitem(sys.modules, "jevmlx.cli", cli)
    monkeypatch.setitem(sys.modules, "jevmlx.engine", engine_mod)
    return calls


@pytest.fixture()
def preset_files(tmp_path):
    """A schema + context + constraints + labels JSONL, written once."""
    schema = {
        "action": {
            "type": "enum",
            "description": "What to do",
            "choices": ["APPROVE", "DENY"],
        },
        "urgent": {"type": "boolean", "description": "Is it urgent"},
    }
    schema_path = tmp_path / "schema.json"
    schema_path.write_text(json.dumps(schema), encoding="utf-8")
    context_path = tmp_path / "context.txt"
    context_path.write_text("A routine request.", encoding="utf-8")
    constraints_path = tmp_path / "constraints.json"
    constraints_path.write_text(
        json.dumps(
            [
                {
                    "type": "implies",
                    "parent": "urgent",
                    "child": "action",
                    "mapping": {True: ["APPROVE", "DENY"], False: ["APPROVE", "DENY"]},
                }
            ]
        ),
        encoding="utf-8",
    )
    jsonl_path = tmp_path / "cases.jsonl"
    jsonl_path.write_text(
        json.dumps(
            {
                "id": "c1",
                "source": "quality-eval",
                "workflow": None,
                "schema": schema,
                "context": "A routine request.",
                "labels": {"action": "APPROVE", "urgent": True},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "schema": str(schema_path),
        "context": str(context_path),
        "constraints": str(constraints_path),
        "jsonl": str(jsonl_path),
    }


def _run(argv):
    """Run main(); returns (exit_code, out, err). SystemExit(0) is success."""
    import contextlib
    import io

    out, err = io.StringIO(), io.StringIO()
    code = 0
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            cli.main(argv)
        except SystemExit as exc:
            code = exc.code or 0
    return code, out.getvalue(), err.getvalue()


# ------------------------------------------------------------------ decide --


def test_decide_native_json_shape(fake_engine, preset_files, capsys, tmp_path):
    """`decide --json` prints the parsed_json payload; the engine received
    the alias model id; flags all wire through."""
    calib = tmp_path / "cal.json"
    # W5-C finding 22: the ONE bundle shape (scalar under "scalar").
    calib.write_text(
        json.dumps(
            {
                "model_revision": "test",
                "prompt_version": "jevmlx-parallel-v9",
                "scoring": "slots",
                "prior_mode": "off",
                "scalar": {"temperature": 1.0},
                "multi": {"a": 1.0, "b": 0.0},
            }
        )
    )
    code, out, err = _run(
        [
            "decide",
            "--model",
            "quality",  # alias: must reach load_engine as given (it resolves)
            "--schema",
            preset_files["schema"],
            "--context",
            preset_files["context"],
            "--constraints",
            preset_files["constraints"],
            "--calibration",
            str(calib),
            "--prior-correction",
            "--scoring",
            "labels",
            "--temperature",
            "0.7",
            "--json",
        ]
    )
    assert code == 0, err
    assert fake_engine == ["quality"]
    payload = json.loads(out)
    assert {"action", "urgent"} <= set(payload)
    for name in ("action", "urgent"):
        assert set(payload[name]) == {"value", "prob"}
    assert "Traceback" not in err


def test_decide_table_output(fake_engine, preset_files):
    """Without --json the demo table renders: fields, values, confidences."""
    code, out, err = _run(
        [
            "decide",
            "--schema",
            preset_files["schema"],
            "--context",
            preset_files["context"],
        ]
    )
    assert code == 0, err
    assert "action" in out and "urgent" in out
    assert "APPROVE" in out
    assert "Assembled JSON" in out


def test_decide_context_stdin(fake_engine, preset_files, monkeypatch):
    """`--context -` reads stdin."""
    monkeypatch.setattr("sys.stdin", type("S", (), {"read": lambda self: "stdin ctx"})())
    code, out, err = _run(
        [
            "decide",
            "--schema",
            preset_files["schema"],
            "--context",
            "-",
            "--json",
        ]
    )
    assert code == 0, err
    assert {"action", "urgent"} <= set(json.loads(out))


def test_decide_bundled_preset(fake_engine, capsys):
    """`decide --preset fintech_fraud` runs the bundled preset end to end."""
    code, out, err = _run(["decide", "--preset", "fintech_fraud", "--json"])
    assert code == 0, err
    assert "is_fraudulent" in out


def test_decide_error_bad_schema_file(fake_engine):
    """A missing schema file exits 1 with ONE line — no traceback."""
    code, out, err = _run(["decide", "--schema", "/nope.json", "--context", "-"])
    assert code == 1
    assert err.count("\n") == 1, err
    assert "Traceback" not in err


def test_decide_error_bad_constraints_json(fake_engine, tmp_path):
    """Invalid constraints JSON: one-line exit 1."""
    bad = tmp_path / "bad.json"
    bad.write_text("not json", encoding="utf-8")
    code, out, err = _run(
        [
            "decide",
            "--preset",
            "fintech_fraud",
            "--constraints",
            str(bad),
        ]
    )
    assert code == 1
    assert err.count("\n") == 1, err
    assert "Traceback" not in err


def test_decide_error_bad_calibration_shape(fake_engine, tmp_path):
    """A wrong-shaped calibration payload exits 1 with one line."""
    bad = tmp_path / "cal.json"
    bad.write_text(json.dumps({"multi": {"a": "x"}}), encoding="utf-8")
    code, out, err = _run(
        [
            "decide",
            "--preset",
            "fintech_fraud",
            "--calibration",
            str(bad),
            "--json",
        ]
    )
    assert code == 1
    assert err.count("\n") == 1, err
    assert "Traceback" not in err


# ------------------------------------------------------------- openai backend --


class _StubOpenAI(BaseHTTPRequestHandler):
    """One canned chat-completions response per POST."""

    def do_POST(self):  # noqa: N802 — http.server API
        length = int(self.headers.get("Content-Length", 0))
        json.loads(self.rfile.read(length))
        canned = self.server.canned.pop(0)
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(canned).encode())


def test_decide_openai_backend_against_stub_server(fake_engine, preset_files, monkeypatch):
    """`--backend openai` end to end against a loopback HTTP stub; the
    api-model tokenizer comes from a stubbed transformers (no Hub)."""
    completion = {
        "choices": [
            {
                "message": {"content": ""},
                "logprobs": {
                    "content": [{"top_logprobs": [{"token": "APPROVE", "logprob": -0.1}]}]
                },
            }
        ]
    }
    server = HTTPServer(("127.0.0.1", 0), _StubOpenAI)
    server.canned = [completion, completion]  # one per field (enum + boolean)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", lambda _m: FakeTokenizer())
    try:
        code, out, err = _run(
            [
                "decide",
                "--schema",
                preset_files["schema"],
                "--context",
                preset_files["context"],
                "--backend",
                "openai",
                "--base-url",
                f"http://127.0.0.1:{server.server_port}/v1",
                "--api-model",
                "stub-model",
                "--json",
            ]
        )
    finally:
        server.shutdown()
    assert code == 0, err
    payload = json.loads(out)
    assert payload["action"]["value"] == "APPROVE"
    assert "Traceback" not in err


# --------------------------------------------------------------- calibrate --


def test_calibrate_end_to_end_writes_out(fake_engine, preset_files, tmp_path):
    """`calibrate --data cases.jsonl --out` fits and writes the JSON file."""
    out_path = tmp_path / "calibrators.json"
    code, out, err = _run(["calibrate", "--data", preset_files["jsonl"], "--out", str(out_path)])
    assert code == 0, err
    payload = json.loads(out_path.read_text())
    # W5-C: ONE bundle shape — the scalar temperature lives under
    # "scalar", with provenance alongside.
    assert payload["scalar"]["temperature"] > 0
    assert payload["prompt_version"].startswith("jevmlx-parallel-")
    assert payload["prior_mode"] == "off"
    assert "Traceback" not in err


def test_calibrate_error_missing_data(fake_engine):
    """A missing --data file exits 1 with ONE line."""
    code, out, err = _run(["calibrate", "--data", "/nope.jsonl"])
    assert code == 1
    assert err.count("\n") == 1, err
    assert "Traceback" not in err


# ----------------------------------------------------------------- validate --


def test_validate_ok_schema(fake_engine, preset_files, monkeypatch):
    """`validate` on a clean schema prints OK and exits 0."""
    monkeypatch.setattr(
        "transformers.AutoTokenizer.from_pretrained",
        lambda _m: FakeTokenizer(),
    )
    code, out, err = _run(["validate", preset_files["schema"]])
    assert code == 0, err
    assert "OK: no findings." in out


# ------------------------------------------------------------------- serve --


def test_serve_decides_over_http(fake_engine, preset_files, monkeypatch):
    """`serve` boots through cli._dispatch and answers one POST /decide with
    the fake engine (jevmlx.engine.load_engine monkeypatched); the thread is
    a daemon, so it dies with the test."""
    import urllib.request

    import jevmlx.serve as serve_mod

    captured: dict = {}

    class _CapturingServer(serve_mod.ThreadingHTTPServer):
        """Force port 0 (OS free-port) and capture the bound address."""

        def __init__(self, addr, *a, **k):
            super().__init__((addr[0], 0), *a, **k)
            captured["server"] = self

    monkeypatch.setattr(serve_mod, "ThreadingHTTPServer", _CapturingServer)
    real_serve_forever = _CapturingServer.serve_forever
    started = threading.Event()

    def signal_then_serve(self, *a, **k):
        started.set()
        real_serve_forever(self, *a, **k)

    monkeypatch.setattr(_CapturingServer, "serve_forever", signal_then_serve)

    thread = threading.Thread(
        target=lambda: cli._dispatch(["serve", "--host", "127.0.0.1", "--port", "0"]),
        daemon=True,
    )
    thread.start()
    assert started.wait(timeout=10), "serve did not start"

    port = captured["server"].server_port

    payload = json.dumps(
        {
            "schema": json.loads(open(preset_files["schema"]).read()),
            "context": "A routine request.",
        }
    ).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/decide",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        body = json.loads(resp.read())
    assert body["parsed_json"]["action"]["value"] == "APPROVE"


# -------------------------------------------------------------------- eval --


def test_eval_parallel_track_writes_outputs(fake_engine, preset_files, tmp_path):
    """`eval --track parallel` writes predictions.jsonl + run.json + timing.json."""
    out_dir = tmp_path / "evalout"
    code, out, err = _run(
        [
            "eval",
            "--data",
            preset_files["jsonl"],
            "--track",
            "parallel",
            "--out",
            str(out_dir),
            "--prior-correction",
            "--scoring",
            "labels",
        ]
    )
    assert code == 0, err
    assert (out_dir / "predictions.jsonl").is_file()
    assert (out_dir / "run.json").is_file()
    run = json.loads((out_dir / "run.json").read_text())
    assert run["config"]["track"] == "parallel"
    assert run["config"]["prior_correction"] is True
    lines = [
        json.loads(line)
        for line in (out_dir / "predictions.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert lines, "no prediction lines"
    assert "Traceback" not in err


def test_eval_error_missing_data(fake_engine, tmp_path):
    """A missing --data file exits 1 with ONE line."""
    code, out, err = _run(
        ["eval", "--data", "/nope.jsonl", "--track", "parallel", "--out", str(tmp_path / "o")]
    )
    assert code == 1
    assert err.count("\n") == 1, err
    assert "Traceback" not in err


# ------------------------------------------------------------------ report --


def test_report_from_eval_output(fake_engine, preset_files, tmp_path):
    """`report` consumes the eval run's predictions.jsonl (offline)."""
    out_dir = tmp_path / "evalout"
    code, _, err = _run(
        ["eval", "--data", preset_files["jsonl"], "--track", "parallel", "--out", str(out_dir)]
    )
    assert code == 0, err
    report_path = tmp_path / "report.json"
    code, out, err = _run(
        ["report", "--predictions", str(out_dir / "predictions.jsonl"), "--out", str(report_path)]
    )
    assert code == 0, err
    payload = json.loads(report_path.read_text())
    assert "metrics" in payload
    assert (tmp_path / "report.md").is_file()


# ------------------------------------------------------------------- bench --


def test_bench_dry_run_through_cli(fake_engine, monkeypatch, tmp_path):
    """`bench --dry-run` prints the plan and exits 0 without loading."""
    from jevmlx import bench

    monkeypatch.setattr(bench, "build_datasets", lambda datasets: ({}, {}))
    code, out, err = _run(
        [
            "bench",
            "--model",
            "org/m",
            "--datasets",
            "bundled",
            "--scorers",
            "slots",
            "--tracks",
            "parallel",
            "--out",
            str(tmp_path / "results"),
            "--dry-run",
        ]
    )
    assert code == 0, err
    assert "machine:" in out
    assert "parallel-slots-bundled" in out


# ------------------------------------------------------------------ doctor --


def test_doctor_json_output(fake_engine, monkeypatch):
    """`doctor --json` prints the payload object (environment + checks list),
    exit matches the payload's exit_code; no traceback."""
    code, out, err = _run(["doctor", "--json"])
    assert "Traceback" not in err
    payload = json.loads(out)
    assert "environment" in payload and "checks" in payload
    assert isinstance(payload["checks"], list)
    assert {"name", "status"} <= set(payload["checks"][0])
    assert code == payload["exit_code"]


# --------------------------------------------------------- version + guard --


def test_version_flag():
    code, out, _ = _run(["--version"])
    assert code == 0
    assert out.startswith("jevmlx ")


def test_engine_runtime_error_is_one_line(monkeypatch):
    """A RuntimeError from the engine path (e.g. non-Apple-Silicon) exits 1
    with one line on stderr — never a traceback."""

    def boom(model_id):
        raise RuntimeError("jevmlx requires Apple Silicon")

    monkeypatch.setattr(cli, "_engine", lambda: (boom, lambda *a, **k: None))
    code, out, err = _run(["decide", "--preset", "fintech_fraud"])
    assert code == 1
    assert err.count("\n") == 1, err
    assert "Apple Silicon" in err


def test_verbose_reraises_instead_of_one_line(monkeypatch):
    """F1: with -v the error re-raises (full traceback) instead of being
    swallowed into a one-liner — the debug path for programming errors."""

    def boom(model_id):
        raise RuntimeError("engine blew up internally")

    monkeypatch.setattr(cli, "_engine", lambda: (boom, lambda *a, **k: None))
    with pytest.raises(RuntimeError, match="engine blew up internally"):
        _run(["-v", "decide", "--preset", "fintech_fraud"])


def test_multiline_error_keeps_full_detail(fake_engine, tmp_path):
    """F2: multi-line validation errors keep their detail — the full
    message is printed, not just the first line."""
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"multi": {"a": "x"}}), encoding="utf-8")
    code, out, err = _run(
        ["decide", "--preset", "fintech_fraud", "--calibration", str(bad), "--json"]
    )
    assert code == 1
    assert "could not convert string to float: 'x'" in err
