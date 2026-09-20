"""B12: dump sample /decide and /v1/systemone responses to JSON fixtures.

The TypeScript client tests parse these EXACT bytes (one source of truth:
the server's real output, not a hand-written shape). Run with:

    python -m tests.dump_serve_fixtures

Writes:
  js/tests/fixtures/decide-200.json
  js/tests/fixtures/systemone-200.json
  js/tests/fixtures/health-200.json
  js/tests/fixtures/models-200.json
  js/tests/fixtures/decide-429.json
  js/tests/fixtures/decide-413.json
  js/tests/fixtures/ready-503.json
"""

from __future__ import annotations

import json
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib import error, request

from conftest import make_engine_result, make_field_telemetry

from jevmlx.serve import _AdmissionQueue, _ServerStats, make_handler

FIXTURES_DIR = Path(__file__).resolve().parents[1] / "js" / "tests" / "fixtures"

_FAKE_RESULT = make_engine_result(
    fields={
        "action": make_field_telemetry(
            value="APPROVE", choices=["APPROVE", "DENY"], probability=0.9
        )
    },
    elapsed_ms=1.0,
)

_SYSTEMONE_QUESTIONS = {
    "q1": {
        "type": "choice",
        "instructions": "Pick a category",
        "criteria": {"A": "Option A", "B": "Option B"},
    },
    "q2": {"type": "noul", "instructions": "Is it true?"},
    "q3": {
        "type": "score",
        "instructions": "Rate 0-3",
        "criteria": {"none": "No harm", "low": "Minor", "med": "Moderate", "high": "Severe"},
    },
}


def _make_systemone_result(questions):
    fields = {}
    for qid, qspec in questions.items():
        qtype = qspec.get("type")
        if qtype == "choice":
            criteria = qspec.get("criteria", {})
            choices = list(criteria.keys()) if isinstance(criteria, dict) else list(criteria)
            fields[qid] = make_field_telemetry(value=choices[0], choices=choices, probability=0.7)
            fields[qid]["top_choices"] = [
                {"choice": choices[0], "probability": 0.7},
                {"choice": choices[1] if len(choices) > 1 else "other", "probability": 0.3},
            ]
        elif qtype == "noul":
            fields[qid] = make_field_telemetry(
                value=True, type_="boolean", choices=["true", "false"], probability=0.8
            )
            fields[qid]["top_choices"] = [
                {"choice": "true", "probability": 0.8},
                {"choice": "false", "probability": 0.2},
            ]
        elif qtype == "score":
            criteria = qspec.get("criteria", {})
            choices = list(criteria.keys()) if isinstance(criteria, dict) else list(criteria)
            fields[qid] = make_field_telemetry(value=choices[1], choices=choices, probability=0.5)
            probs = [0.1, 0.5, 0.3, 0.1] if len(choices) >= 4 else [0.2, 0.5, 0.3][: len(choices)]
            fields[qid]["top_choices"] = [
                {"choice": c, "probability": p} for c, p in zip(choices, probs, strict=False)
            ]
    return make_engine_result(fields=fields)


def _start_server(decide_fn, *, stats=None, aq=None, **kw):
    if stats is None:
        stats = _ServerStats()
        stats.mark_ready()
    if aq is None:
        aq = _AdmissionQueue(maxsize=16, stats=stats)
    kw.setdefault("tokenize_fn", lambda t: t.split())
    kw.setdefault("count_rows_fn", lambda s: 1)
    handler = make_handler(decide_fn, "fake", stats, admission_queue=aq, **kw)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    port = httpd.server_address[1]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, port, stats, aq


def _post(port, path, payload):
    data = json.dumps(payload).encode()
    req = request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=5) as resp:
            return resp.status, json.loads(resp.read()), dict(resp.headers)
    except error.HTTPError as e:
        return e.code, json.loads(e.read()), dict(e.headers)


def _get(port, path):
    req = request.Request(f"http://127.0.0.1:{port}{path}")
    try:
        with request.urlopen(req, timeout=5) as resp:
            return resp.status, json.loads(resp.read()), dict(resp.headers)
    except error.HTTPError as e:
        return e.code, json.loads(e.read()), dict(e.headers)


def _write(name: str, status: int, body: dict, headers: dict) -> None:
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    (FIXTURES_DIR / name).write_text(
        json.dumps(
            {"status": status, "headers": headers, "body": body},
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    # --- happy-path /decide + /health + /v1/models ---
    httpd, port, stats, aq = _start_server(lambda sd, ctx, t=1.0: dict(_FAKE_RESULT))
    try:
        # /decide
        schema = {"action": {"type": "enum", "choices": ["APPROVE", "DENY"], "description": "d"}}
        status, body, headers = _post(
            port, "/decide", {"schema": schema, "context": "ctx", "temperature": 0.7}
        )
        assert status == 200, status
        _write("decide-200.json", status, body, headers)

        # /health
        status, body, headers = _get(port, "/health")
        assert status == 200, status
        _write("health-200.json", status, body, headers)

        # /v1/models
        status, body, headers = _get(port, "/v1/models")
        assert status == 200, status
        _write("models-200.json", status, body, headers)
    finally:
        httpd.shutdown()
        httpd.server_close()
        aq.shutdown()

    # --- /v1/systemone (needs a decide_fn that returns the right fields) ---
    httpd, port, stats, aq = _start_server(
        lambda sd, ctx, t=1.0: _make_systemone_result(_SYSTEMONE_QUESTIONS)
    )
    try:
        status, body, headers = _post(
            port, "/v1/systemone", {"state": {"user": "alice"}, "questions": _SYSTEMONE_QUESTIONS}
        )
        assert status == 200, status
        _write("systemone-200.json", status, body, headers)
    finally:
        httpd.shutdown()
        httpd.server_close()
        aq.shutdown()

    # --- /ready 503 (server not ready) ---
    # Do NOT call mark_ready() so /ready returns 503.
    httpd, port, stats, aq = _start_server(
        lambda sd, ctx, t=1.0: dict(_FAKE_RESULT),
        stats=_ServerStats(),  # default: not ready
    )
    try:
        status, body, headers = _get(port, "/ready")
        assert status == 503, status
        _write("ready-503.json", status, body, headers)
    finally:
        httpd.shutdown()
        httpd.server_close()
        aq.shutdown()

    # --- /decide 413 (too many rows) ---
    httpd, port, stats, aq = _start_server(lambda sd, ctx, t=1.0: dict(_FAKE_RESULT), max_rows=0)
    try:
        schema = {"action": {"type": "enum", "choices": ["A", "B"], "description": "d"}}
        status, body, headers = _post(port, "/decide", {"schema": schema, "context": "x"})
        assert status == 413, status
        _write("decide-413.json", status, body, headers)
    finally:
        httpd.shutdown()
        httpd.server_close()
        aq.shutdown()

    # --- /decide 429 (queue full) ---
    # A blocking decide_fn + queue capacity 1: fire 3 concurrent requests;
    # the worker holds the first (queued), so the 2nd fills the queue and
    # the 3rd is rejected with 429.
    block_event = threading.Event()
    results: list[dict] = []

    def blocking_decide(sd, ctx, t=1.0):
        block_event.wait(timeout=5)
        return dict(_FAKE_RESULT)

    httpd, port, stats, aq = _start_server(blocking_decide)
    aq._max = 1  # capacity 1: one queued + one in-flight
    try:
        schema = {"action": {"type": "enum", "choices": ["A", "B"], "description": "d"}}
        payload = {"schema": schema, "context": "x"}

        def _fire():
            results.append(_post(port, "/decide", payload))

        threads = [threading.Thread(target=_fire) for _ in range(3)]
        for t in threads:
            t.start()
        # Let the queue settle (worker holds req 1, req 2 queued, req 3 overflow).
        threading.Event().wait(timeout=0.5)
        block_event.set()
        for t in threads:
            t.join(timeout=3)
        # Find the 429.
        for status, body, headers in results:
            if status == 429:
                _write("decide-429.json", status, body, headers)
                break
        else:
            raise AssertionError(f"no 429 in {[r[0] for r in results]}")
    finally:
        httpd.shutdown()
        httpd.server_close()
        aq.shutdown()

    print(f"wrote {len(list(FIXTURES_DIR.glob('*.json')))} fixtures to {FIXTURES_DIR}")


if __name__ == "__main__":
    main()


# ---- pytest wrapper: verify committed fixtures match the live server ----


def test_serve_fixtures_match_committed(tmp_path):
    """The committed js/tests/fixtures/*.json must match a fresh server dump.

    One source of truth: if the server response shape changes, this test
    fails until the fixtures are re-dumped (run `python tests/test_dump_serve_fixtures.py`).
    """
    import importlib

    mod = importlib.import_module("tests.test_dump_serve_fixtures")
    # Redirect FIXTURES_DIR to a temp dir, dump, compare to committed.
    orig = mod.FIXTURES_DIR
    mod.FIXTURES_DIR = tmp_path
    try:
        mod.main()
    finally:
        mod.FIXTURES_DIR = orig

    committed = orig
    for fixture_file in sorted(committed.glob("*.json")):
        committed_text = fixture_file.read_text(encoding="utf-8")
        fresh_file = tmp_path / fixture_file.name
        assert fresh_file.exists(), f"{fixture_file.name} missing from fresh dump"
        fresh_text = fresh_file.read_text(encoding="utf-8")
        # Compare parsed JSON (ignores key order, whitespace).
        committed_json = json.loads(committed_text)
        fresh_json = json.loads(fresh_text)
        # The Date and X-Request-Id headers differ per-run; strip them.
        for d in (committed_json, fresh_json):
            d["headers"].pop("Date", None)
            d["headers"].pop("X-Request-Id", None)
            d["body"].pop("request_id", None)
            d["body"].pop("queue_wait_ms", None)
            d["body"].pop("queue_depth", None)
        assert committed_json == fresh_json, (
            f"{fixture_file.name} drifted from the server output; "
            "re-dump with: python tests/test_dump_serve_fixtures.py"
        )
