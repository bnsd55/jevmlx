"""Tests for the OpenAI-compatible slots backend: loopback http.server only,
no network."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
from conftest import FakeTokenizer as _FakeTokenizer

from jevmlx.http import ChatCompletionsError
from jevmlx.openai_slots import decide_openai
from jevmlx.schema import StructuredSchema


def _tok() -> _FakeTokenizer:
    return _FakeTokenizer()


def _schema() -> StructuredSchema:
    return StructuredSchema(
        {
            "risk_tier": {
                "type": "enum",
                "description": "Risk tier",
                "choices": ["LOW", "MEDIUM", "HIGH"],
            }
        }
    )


class _FakeOpenAI(BaseHTTPRequestHandler):
    """Canned chat-completions responses; records request payloads."""

    server_version = "FakeOpenAI/1"

    def do_POST(self):  # noqa: N802 — http.server API
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length))
        self.server.requests.append(body)
        canned = self.server.canned.pop(0)
        self.send_response(canned.get("status", 200))
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(canned["body"]).encode())


def _make_server(canned: list[dict]):
    server = HTTPServer(("127.0.0.1", 0), _FakeOpenAI)
    server.requests = []
    server.canned = list(canned)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def _completion(top_logprobs: list[dict]) -> dict:
    return {
        "choices": [
            {
                "message": {"content": ""},
                "logprobs": {"content": [{"top_logprobs": top_logprobs}]},
            }
        ]
    }


def test_scalar_field_reads_top_logprobs_and_renormalises():
    """Canned top_logprobs -> correct value, probabilities sum to 1."""
    server, thread = _make_server(
        [
            {
                "body": _completion(
                    [
                        {"token": '"A"', "logprob": -0.5},
                        {"token": '"B"', "logprob": -1.0},
                        {"token": '"C"', "logprob": -2.0},
                    ]
                )
            }
        ]
    )
    try:
        result = decide_openai(
            f"http://127.0.0.1:{server.server_port}",
            "fake-model",
            None,
            _schema(),
            "ctx",
            _tok(),
        )
    finally:
        server.shutdown()
        thread.join()

    assert result["parsed_json"]["risk_tier"]["value"] == "LOW"
    probs = [
        result["field_telemetry"]["risk_tier"]["log_scores"][c] for c in ("LOW", "MEDIUM", "HIGH")
    ]
    # renormalised over exactly the three aliases; exponentiate and sum
    import math

    assert abs(sum(math.exp(lp) for lp in probs) - 1.0) < 1e-9
    assert (
        result["field_telemetry"]["risk_tier"]["log_scores"]["LOW"]
        > result["field_telemetry"]["risk_tier"]["log_scores"]["HIGH"]
    )
    assert result["confidence_model"] == "openai_slots"
    assert result["prompt_version"] == "jevmlx-openai-slots-v1"
    assert len(result["prompt_sha256"]) == 64
    telemetry = result["field_telemetry"]["risk_tier"]
    assert telemetry["rows"] == 1
    assert telemetry["truncated"] is False
    assert telemetry["alternatives"][0] == ("LOW", telemetry["probability"])


def test_missing_alias_gets_floor_and_truncated_flag():
    server, thread = _make_server(
        [
            {
                "body": _completion(
                    [
                        {"token": '"A"', "logprob": -0.1},
                        {"token": '"B"', "logprob": -3.0},
                        # '"C"' missing from top-k
                    ]
                )
            }
        ]
    )
    try:
        result = decide_openai(
            f"http://127.0.0.1:{server.server_port}", "m", None, _schema(), "ctx", _tok()
        )
    finally:
        server.shutdown()
        thread.join()
    telemetry = result["field_telemetry"]["risk_tier"]
    assert telemetry["truncated"] is True
    # floor = exp(min found logprob) = exp(-3.0); renormalised it stays the
    # smallest of the three
    log_scores = telemetry["log_scores"]
    assert log_scores["HIGH"] == min(log_scores.values())
    import math

    assert (
        abs(
            math.exp(log_scores["HIGH"])
            - math.exp(-3.0) / (math.exp(-0.1) + math.exp(-3.0) + math.exp(-3.0))
        )
        < 1e-9
    )


def test_multi_field_one_request_per_option():
    server, thread = _make_server(
        [
            {
                "body": _completion(
                    [{"token": '"yes"', "logprob": -0.2}, {"token": '"no"', "logprob": -4.0}]
                )
            },
            {
                "body": _completion(
                    [{"token": '"yes"', "logprob": -4.0}, {"token": '"no"', "logprob": -0.3}]
                )
            },
        ]
    )
    schema = StructuredSchema(
        {
            "tags": {
                "type": "multi",
                "description": "tags",
                "choices": ["a", "b"],
            }
        }
    )
    try:
        result = decide_openai(
            f"http://127.0.0.1:{server.server_port}", "m", None, schema, "ctx", _tok()
        )
    finally:
        server.shutdown()
        thread.join()

    assert len(server.requests) == 2  # one yes/no request per option
    telemetry = result["field_telemetry"]["tags"]
    assert telemetry["rows"] == 2
    assert set(telemetry["per_option"]) == {"a", "b"}
    assert telemetry["per_option"]["a"] > 0.5 > telemetry["per_option"]["b"]
    assert result["parsed_json"]["tags"]["value"] == ["a"]


def test_request_payload_shape():
    server, thread = _make_server([{"body": _completion([{"token": '"A"', "logprob": -0.1}])}])
    try:
        decide_openai(f"http://127.0.0.1:{server.server_port}", "m", None, _schema(), "ctx", _tok())
    finally:
        server.shutdown()
        thread.join()
    payload = server.requests[0]
    assert payload["model"] == "m"
    assert payload["max_tokens"] == 1
    assert payload["logprobs"] is True
    assert payload["top_logprobs"] == 20
    assert payload["temperature"] == 0
    roles = [m["role"] for m in payload["messages"]]
    assert roles == ["system", "user", "assistant"]
    # the decision row is the assistant prefix
    assert payload["messages"][2]["content"] == '{\n  "risk_tier": '
    # the user turn carries the alias schema block and delimited context
    assert '"risk_tier": A)' in payload["messages"][1]["content"]
    assert "<<<CONTEXT" in payload["messages"][1]["content"]


def test_http_500_raises_baseline_error():
    server, thread = _make_server([{"status": 500, "body": {"error": {"message": "kaboom"}}}])
    try:
        with pytest.raises(ChatCompletionsError, match="500"):
            decide_openai(
                f"http://127.0.0.1:{server.server_port}", "m", None, _schema(), "ctx", _tok()
            )
    finally:
        server.shutdown()
        thread.join()
