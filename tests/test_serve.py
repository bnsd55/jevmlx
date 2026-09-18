"""Serve tests: fake decide_fn, HTTPServer on port 0 in a thread. No model."""

import json
import logging
import threading
import urllib.error
import urllib.request
from http.server import HTTPServer

import pytest
from conftest import make_engine_result, make_field_telemetry

from jevmlx.serve import make_handler

FAKE_RESULT = make_engine_result(
    fields={"action": make_field_telemetry(value="APPROVE", choices=["APPROVE"], probability=0.9)},
    elapsed_ms=1.0,
)


def _start_server(decide_fn, model_id="fake"):
    httpd = HTTPServer(("127.0.0.1", 0), make_handler(decide_fn, model_id))
    port = httpd.server_address[1]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, port


def _post(port, payload, raw=None):
    data = raw if raw is not None else json.dumps(payload).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/decide",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


@pytest.fixture()
def server():
    calls = []

    def fake_decide(schema_dict, context, temperature=1.0):
        calls.append((schema_dict, context, temperature))
        return dict(FAKE_RESULT)

    httpd, port = _start_server(fake_decide)
    yield port, calls
    httpd.shutdown()
    httpd.server_close()


def test_decide_valid_returns_fixed_dict_and_args(server):
    port, calls = server
    schema = {"action": {"type": "enum", "choices": ["APPROVE"], "description": "d"}}
    status, body = _post(port, {"schema": schema, "context": "ctx here", "temperature": 0.7})

    assert status == 200
    assert body == FAKE_RESULT
    assert calls == [(schema, "ctx here", 0.7)]


def test_decide_invalid_json_is_400(server):
    port, _ = server
    status, body = _post(port, None, raw=b"{not json")
    assert status == 400
    assert "error" in body


def test_decide_missing_context_is_400(server):
    port, _ = server
    status, body = _post(port, {"schema": {"a": {"type": "boolean", "description": "d"}}})
    assert status == 400
    assert "error" in body


def test_health(server):
    port, _ = server
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as resp:
        assert resp.status == 200
        assert json.loads(resp.read()) == {
            "ok": True,
            "model": "fake",
            "busy": False,
            "requests_served": 0,
        }


def test_health_after_decide_shows_served_and_not_busy(server):
    port, _ = server
    schema = {"action": {"type": "enum", "choices": ["APPROVE"], "description": "d"}}
    _post(port, {"schema": schema, "context": "ctx"})
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as resp:
        health = json.loads(resp.read())
    assert health["requests_served"] == 1
    assert health["busy"] is False


def test_decide_internal_error_is_500():
    def boom(schema_dict, context, temperature=1.0):
        raise RuntimeError("gpu said no")

    httpd, port = _start_server(boom)
    try:
        status, body = _post(port, {"schema": {}, "context": "x"})
        assert status == 500
        assert body["error"] == "gpu said no"
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_decide_internal_error_with_empty_message_is_500():
    def silent_boom(schema_dict, context, temperature=1.0):
        raise RuntimeError()  # str(e) is "" -> error must be the exception type name

    httpd, port = _start_server(silent_boom)
    try:
        status, body = _post(port, {"schema": {}, "context": "x"})
        assert status == 500
        assert body["error"] == "RuntimeError"
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_serve_does_not_print(server, capsys):
    """T8a: the library logs, never print()s — stdout must stay clean."""
    port, _ = server
    schema = {"action": {"type": "enum", "choices": ["APPROVE"], "description": "d"}}
    _post(port, {"schema": schema, "context": "ctx"})
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5):
        pass
    assert capsys.readouterr().out == ""


def test_decide_emits_log_record(server, caplog):
    """T8b: a decide emits a log record through the logging module."""
    port, _ = server
    schema = {"action": {"type": "enum", "choices": ["APPROVE"], "description": "d"}}
    with caplog.at_level(logging.INFO, logger="jevmlx.serve"):
        _post(port, {"schema": schema, "context": "ctx"})
    assert any(r.levelno >= logging.INFO for r in caplog.records)
