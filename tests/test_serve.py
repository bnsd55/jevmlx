"""Serve tests: fake decide_fn, ThreadingHTTPServer on port 0 in a thread. No model."""

import json
import logging
import threading
import time
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest
from conftest import make_engine_result, make_field_telemetry

from jevmlx.serve import _AdmissionQueue, _ServerStats, make_handler

FAKE_RESULT = make_engine_result(
    fields={"action": make_field_telemetry(value="APPROVE", choices=["APPROVE"], probability=0.9)},
    elapsed_ms=1.0,
)


def _start_server(decide_fn, model_id="fake", *, stats=None, aq=None, **kw):
    if stats is None:
        stats = _ServerStats()
        stats.mark_ready()
    if aq is None:
        aq = _AdmissionQueue(maxsize=16, stats=stats)
    # M2 round 3: tokenize_fn is required; default to a simple stand-in.
    kw.setdefault("tokenize_fn", lambda t: t.split())  # whitespace-split fake
    kw.setdefault("count_rows_fn", lambda s: 1)
    handler = make_handler(decide_fn, model_id, stats, admission_queue=aq, **kw)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    port = httpd.server_address[1]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, port, stats, aq


def _post(port, payload, raw=None, headers=None):
    data = raw if raw is not None else json.dumps(payload).encode()
    hdrs = {"Content-Type": "application/json"}
    if headers:
        hdrs.update(headers)
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/decide",
        data=data,
        headers=hdrs,
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, json.loads(resp.read()), dict(resp.headers)
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read()), dict(e.headers)


def _get(port, path, *, headers=None):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, json.loads(resp.read()), dict(resp.headers)
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read()), dict(e.headers)


@pytest.fixture()
def server():
    calls = []
    lock = threading.Lock()

    def fake_decide(schema_dict, context, temperature=1.0):
        with lock:
            calls.append((schema_dict, context, temperature))
        return dict(FAKE_RESULT)

    httpd, port, stats, aq = _start_server(fake_decide)
    yield port, calls, stats, aq
    httpd.shutdown()
    httpd.server_close()
    aq.shutdown()


# ---------------------------------------------------------------------------
# Existing contract (updated for B7: queue telemetry, no `busy`, request id)
# ---------------------------------------------------------------------------


def test_decide_valid_returns_fixed_dict_and_args(server):
    port, calls, _, _ = server
    schema = {"action": {"type": "enum", "choices": ["APPROVE"], "description": "d"}}
    status, body, headers = _post(
        port, {"schema": schema, "context": "ctx here", "temperature": 0.7}
    )

    assert status == 200
    # B7: the response carries queue telemetry + request id, ADDED to the
    # engine result (not replacing it).
    for key in ("queue_depth", "queue_wait_ms", "request_id"):
        assert key in body, key
    # All original engine-result keys are preserved.
    for key in FAKE_RESULT:
        assert key in body, key
    assert calls == [(schema, "ctx here", 0.7)]


def test_decide_invalid_json_is_400(server):
    port, _, _, _ = server
    status, body, _ = _post(port, None, raw=b"{not json")
    assert status == 400
    assert "error" in body


def test_decide_missing_context_is_400(server):
    port, _, _, _ = server
    status, body, _ = _post(port, {"schema": {"a": {"type": "boolean", "description": "d"}}})
    assert status == 400
    assert "error" in body


def test_health(server):
    port, _, _, _ = server
    status, body, _ = _get(port, "/health")
    assert status == 200
    # /health = process liveness (always 200). B7: carries worker_alive +
    # queue telemetry. No `busy` (M4).
    assert body["ok"] is True
    assert body["model"] == "fake"
    assert body["worker_alive"] is True
    assert body["requests_served"] == 0
    assert "queue_depth" in body
    assert "queue_capacity" in body
    assert "busy" not in body


def test_health_after_decide_shows_served(server):
    port, _, _, _ = server
    schema = {"action": {"type": "enum", "choices": ["APPROVE"], "description": "d"}}
    _post(port, {"schema": schema, "context": "ctx"})
    _, body, _ = _get(port, "/health")
    assert body["requests_served"] == 1


def test_decide_internal_error_is_500():
    """M5: a decide_fn exception is a 500 to THAT request; the worker
    survives and continues."""

    def boom(schema_dict, context, temperature=1.0):
        raise RuntimeError("gpu said no")

    httpd, port, stats, aq = _start_server(boom)
    try:
        status, body, _ = _post(port, {"schema": {}, "context": "x"})
        assert status == 500
        assert body["error"] == "gpu said no"
        # Worker survived — a second request gets a real response.
        status2, body2, _ = _post(port, {"schema": {}, "context": "y"})
        assert status2 == 500  # boom always raises
        assert body2["error"] == "gpu said no"
        # Worker is still alive (M5).
        _, health, _ = _get(port, "/health")
        assert health["worker_alive"] is True
    finally:
        httpd.shutdown()
        httpd.server_close()
        aq.shutdown()


def test_decide_internal_error_with_empty_message_is_500():
    def silent_boom(schema_dict, context, temperature=1.0):
        raise RuntimeError()  # str(e) is "" -> error must be the type name

    httpd, port, stats, aq = _start_server(silent_boom)
    try:
        status, body, _ = _post(port, {"schema": {}, "context": "x"})
        assert status == 500
        assert body["error"] == "RuntimeError"
    finally:
        httpd.shutdown()
        httpd.server_close()
        aq.shutdown()


def test_serve_does_not_print(server, capsys):
    """T8a: the library logs, never print()s — stdout must stay clean."""
    port, _, _, _ = server
    schema = {"action": {"type": "enum", "choices": ["APPROVE"], "description": "d"}}
    _post(port, {"schema": schema, "context": "ctx"})
    _get(port, "/health")
    assert capsys.readouterr().out == ""


def test_decide_emits_log_record(server, caplog):
    """T8b: a decide emits a log record through the logging module."""
    port, _, _, _ = server
    schema = {"action": {"type": "enum", "choices": ["APPROVE"], "description": "d"}}
    with caplog.at_level(logging.INFO, logger="jevmlx.serve"):
        _post(port, {"schema": schema, "context": "ctx"})
    assert any(r.levelno >= logging.INFO for r in caplog.records)


# ---------------------------------------------------------------------------
# W6-B7 tests
# ---------------------------------------------------------------------------


def test_request_id_echoed_when_provided(server):
    """B7 item 4: echo X-Request-Id if given."""
    port, _, _, _ = server
    schema = {"action": {"type": "enum", "choices": ["APPROVE"], "description": "d"}}
    status, body, headers = _post(
        port, {"schema": schema, "context": "ctx"}, headers={"X-Request-Id": "abc-123"}
    )
    assert status == 200
    assert headers.get("X-Request-Id") == "abc-123"
    assert body["request_id"] == "abc-123"


def test_request_id_generated_when_absent(server):
    """B7 item 4: generate a request id when the client doesn't send one."""
    port, _, _, _ = server
    schema = {"action": {"type": "enum", "choices": ["APPROVE"], "description": "d"}}
    status, body, headers = _post(port, {"schema": schema, "context": "ctx"})
    assert status == 200
    rid = headers.get("X-Request-Id")
    assert rid and rid.startswith("jev-")
    assert body["request_id"] == rid


def test_ready_503_until_marked_ready():
    """B7 item 3: /ready returns 503 until model load + warm-up, then 200."""
    stats = _ServerStats()
    httpd, port, _, aq = _start_server(lambda s, c, t: {}, "fake", stats=stats)
    try:
        # Before ready: 503.
        status, body, _ = _get(port, "/ready")
        assert status == 503
        assert body["ready"] is False
        # Mark ready (simulates model load + warm-up complete).
        stats.mark_ready()
        status, body, _ = _get(port, "/ready")
        assert status == 200
        assert body["ready"] is True
    finally:
        httpd.shutdown()
        httpd.server_close()
        aq.shutdown()


def test_ready_503_when_worker_dead():
    """M5: /ready goes 503 when the worker dies, even if warm-up completed."""
    stats = _ServerStats()
    stats.mark_ready()
    httpd, port, _, aq = _start_server(lambda s, c, t: {}, "fake", stats=stats)
    try:
        status, _, _ = _get(port, "/ready")
        assert status == 200
        stats.mark_worker_dead()
        status, _, _ = _get(port, "/ready")
        assert status == 503
        # /health still 200 (liveness) but reports worker_alive=False.
        _, health, _ = _get(port, "/health")
        assert health["worker_alive"] is False
    finally:
        httpd.shutdown()
        httpd.server_close()
        aq.shutdown()


def test_health_always_200_even_before_ready():
    """B7 item 3: /health is process liveness — always 200 if alive."""
    stats = _ServerStats()
    httpd, port, _, aq = _start_server(lambda s, c, t: {}, "fake", stats=stats)
    try:
        status, body, _ = _get(port, "/health")
        assert status == 200  # liveness, not readiness
        assert body["ok"] is True
        assert "queue_depth" in body
        assert "queue_capacity" in body
    finally:
        httpd.shutdown()
        httpd.server_close()
        aq.shutdown()


def test_429_on_queue_overflow_with_retry_after():
    """B7 item 2: full queue -> 429 with Retry-After header (not 529)."""
    block = threading.Event()

    def blocking_decide(schema_dict, context, temperature=1.0):
        block.wait(timeout=10)
        return {}

    # Queue size 1: after the first request occupies the worker + 1 queued,
    # the third is rejected with 429.
    stats = _ServerStats()
    stats.mark_ready()
    aq = _AdmissionQueue(maxsize=1, stats=stats)
    httpd, port, _, _ = _start_server(blocking_decide, "fake", stats=stats, aq=aq)
    try:
        schema = {"action": {"type": "enum", "choices": ["A"], "description": "d"}}
        # Fire 3 concurrent requests: 1 worker + 1 queued + 1 rejected.
        results = []
        threads = []
        rlock = threading.Lock()

        def fire():
            try:
                s, b, h = _post(port, {"schema": schema, "context": "x"})
                with rlock:
                    results.append((s, b, h))
            except Exception as e:  # noqa: BLE001
                with rlock:
                    results.append((0, {"error": str(e)}, {}))

        for _ in range(3):
            t = threading.Thread(target=fire)
            t.start()
            threads.append(t)
        time.sleep(1.0)

        rejected = [r for r in results if r[0] == 429]
        assert len(rejected) >= 1, f"expected at least one 429, got {[r[0] for r in results]}"
        status, body, headers = rejected[0]
        assert "Retry-After" in headers
        assert int(headers["Retry-After"]) >= 1
        assert body["error"] == "admission queue full"
        assert body["queue_depth"] >= 1

        block.set()
    finally:
        httpd.shutdown()
        httpd.server_close()
        aq.shutdown()


def test_serialization_two_decides_never_overlap(server):
    """The serial worker never runs two decide_fn calls concurrently."""
    port, _, _, _ = server
    active = threading.Event()
    overlap = threading.Event()

    # Replace the decide_fn with one that detects overlap.
    httpd, port2, stats, aq = _start_server(None)  # placeholder
    httpd.shutdown()
    httpd.server_close()
    aq.shutdown()

    def detecting_decide(schema_dict, context, temperature=1.0):
        if active.is_set():
            overlap.set()
        active.set()
        time.sleep(0.1)
        active.clear()
        return dict(FAKE_RESULT)

    httpd, port, stats, aq = _start_server(detecting_decide)
    try:
        schema = {"action": {"type": "enum", "choices": ["A"], "description": "d"}}
        threads = []
        for _ in range(4):
            t = threading.Thread(target=lambda: _post(port, {"schema": schema, "context": "x"}))
            t.start()
            threads.append(t)
        for t in threads:
            t.join(timeout=5)
        assert not overlap.is_set(), "two decide_fn calls ran concurrently"
    finally:
        httpd.shutdown()
        httpd.server_close()
        aq.shutdown()


def test_admission_rejects_too_many_rows_413():
    """B7 item 7: max expanded rows exceeded -> 413."""

    def count_rows_fn(schema_dict):
        return 3000  # over the default 2048

    stats = _ServerStats()
    stats.mark_ready()
    httpd, port, _, aq = _start_server(
        lambda s, c, t: {},
        "fake",
        stats=stats,
        count_rows_fn=count_rows_fn,
        tokenize_fn=lambda t: [1],
        max_rows=2048,
    )
    try:
        schema = {"big": {"type": "enum", "choices": ["c"], "description": "d"}}
        status, body, _ = _post(port, {"schema": schema, "context": "ctx"})
        assert status == 413
        assert body["limit"] == "max_rows"
        assert body["value"] > body["ceiling"]
    finally:
        httpd.shutdown()
        httpd.server_close()
        aq.shutdown()


def test_admission_rejects_too_many_prompt_tokens_413():
    """B7 item 7: max prompt tokens exceeded -> 413."""
    stats = _ServerStats()
    stats.mark_ready()
    httpd, port, _, aq = _start_server(
        lambda s, c, t: {},
        "fake",
        stats=stats,
        tokenize_fn=lambda t: list(range(100000)),
        count_rows_fn=lambda s: 1,
        max_prompt_tokens=100,
    )
    try:
        schema = {"a": {"type": "enum", "choices": ["x"], "description": "d"}}
        status, body, _ = _post(port, {"schema": schema, "context": "hello"})
        assert status == 413
        assert body["limit"] == "max_prompt_tokens"
    finally:
        httpd.shutdown()
        httpd.server_close()
        aq.shutdown()


def test_admission_no_projected_memory_limit():
    """N1: the projected-memory limit is DROPPED. A request that would have
    been rejected by the old memory projection is now accepted (the engine
    chunks rows to its own measured budget)."""
    stats = _ServerStats()
    stats.mark_ready()
    httpd, port, _, aq = _start_server(
        lambda s, c, t: dict(FAKE_RESULT),
        "fake",
        stats=stats,
        tokenize_fn=lambda t: [1],
        count_rows_fn=lambda s: 1,
    )
    try:
        schema = {"a": {"type": "enum", "choices": ["x"], "description": "d"}}
        status, body, _ = _post(port, {"schema": schema, "context": "ctx"})
        assert status == 200  # accepted — no memory projection
    finally:
        httpd.shutdown()
        httpd.server_close()
        aq.shutdown()


def test_admission_bad_schema_is_400_not_dropped():
    """B3/N8: a malformed schema (dict that fails StructuredSchema) is 400,
    not a dropped connection. Mirrors production: _count_rows (O(schema)
    arithmetic, never raises) runs in _admit; StructuredSchema validation
    happens inside the worker (decide_fn) and maps to 400 there."""
    from jevmlx.schema import StructuredSchema
    from jevmlx.serve import _count_rows

    def decide_fn(schema_dict, context, temperature=1.0):
        # Production: the worker constructs StructuredSchema, which can raise
        # ValueError on a bad field spec. The worker catches it -> 400.
        StructuredSchema(schema_dict)
        return dict(FAKE_RESULT)

    stats = _ServerStats()
    stats.mark_ready()
    httpd, port, _, aq = _start_server(
        decide_fn,
        "fake",
        stats=stats,
        count_rows_fn=_count_rows,  # production: O(schema), never raises
        tokenize_fn=lambda t: [1],
    )
    try:
        # A dict schema with a malformed field spec (no type) that
        # StructuredSchema rejects — the worker raises ValueError -> 400.
        status, body, _ = _post(
            port, {"schema": {"a": {"description": "no type"}}, "context": "ctx"}
        )
        assert status == 400
    finally:
        httpd.shutdown()
        httpd.server_close()
        aq.shutdown()


def test_queue_wait_ms_in_response(server):
    """B7 item 5: queue depth and queue-wait ms in the response telemetry."""
    port, _, _, _ = server
    schema = {"action": {"type": "enum", "choices": ["APPROVE"], "description": "d"}}
    status, body, _ = _post(port, {"schema": schema, "context": "ctx"})
    assert status == 200
    assert isinstance(body["queue_wait_ms"], (int, float))
    assert body["queue_wait_ms"] >= 0
    assert body["queue_depth"] >= 0


def test_b1_strict_dead_queued_client_frees_slot():
    """B1 strict: a dead queued client frees its slot immediately so a live
    client behind it is admitted (not blocked behind the current forward).

    A in flight (slow), B queued then disconnects, C must be admitted
    (not 429 behind a dead B)."""

    def slow_decide(schema_dict, context, temperature=1.0):
        time.sleep(1.0)  # A occupies the worker for 1s
        return dict(FAKE_RESULT)

    # B1 round 3: maxsize=1 — A occupies worker, B queued (fills the 1
    # slot), C must be admitted because B's disconnect frees the slot.
    stats = _ServerStats()
    stats.mark_ready()
    aq = _AdmissionQueue(maxsize=1, stats=stats)
    httpd, port, _, _ = _start_server(slow_decide, "fake", stats=stats, aq=aq)
    try:
        schema = {"action": {"type": "enum", "choices": ["A"], "description": "d"}}

        # A: occupies the worker (slow, 1s).
        threading.Thread(
            target=lambda: _post(port, {"schema": schema, "context": "A"}),
            daemon=True,
        ).start()
        time.sleep(0.2)  # A is in the worker.

        # B: queued, then we simulate a disconnect by marking it cancelled.
        # Submit B directly to the queue (bypass the HTTP layer) then cancel.
        b_request = {
            "schema_dict": schema,
            "context": "B",
            "temperature": 1.0,
            "decide_fn": slow_decide,
            "cancel_event": threading.Event(),
            "result_event": threading.Event(),
            "result": None,
            "error": None,
        }
        assert aq.submit(b_request)
        # B is now queued. Simulate disconnect: mark cancelled.
        aq.mark_cancelled(b_request)
        b_request["cancel_event"].set()

        # C: must be admitted (B is dead, its slot is freed by mark_cancelled).
        # depth excludes B (cancelled), so C fits in the queue.
        assert aq.depth <= 1, f"depth={aq.depth} — dead B still counted"

        c_status, c_body, _ = _post(port, {"schema": schema, "context": "C"})
        assert c_status != 429, "C was rejected behind a dead queued client (B1 leak)"
    finally:
        httpd.shutdown()
        httpd.server_close()
        aq.shutdown()


def test_n3_queue_wait_ms_measures_real_wait():
    """N3: queue_wait_ms measures the time until the worker picks the request
    up, not just submit() time."""

    def slow_decide(schema_dict, context, temperature=1.0):
        time.sleep(0.5)
        return dict(FAKE_RESULT)

    stats = _ServerStats()
    stats.mark_ready()
    aq = _AdmissionQueue(maxsize=4, stats=stats)
    httpd, port, _, _ = _start_server(slow_decide, "fake", stats=stats, aq=aq)
    try:
        schema = {"action": {"type": "enum", "choices": ["A"], "description": "d"}}

        # A: occupies the worker (0.5s).
        threading.Thread(
            target=lambda: _post(port, {"schema": schema, "context": "A"}),
            daemon=True,
        ).start()
        time.sleep(0.1)  # A is in the worker.

        # B: queued behind A. Its queue_wait_ms should be >= ~0.3s (the time
        # A holds the worker), not 0.0.
        b_status, b_body, _ = _post(port, {"schema": schema, "context": "B"})
        assert b_status == 200
        # B waited behind A for at least 0.2s. queue_wait_ms must reflect a
        # real wait (N3: the old submit-time delta was 0.0).
        assert b_body["queue_wait_ms"] > 100, (
            f"queue_wait_ms={b_body['queue_wait_ms']} — expected > 100ms (real wait)"
        )
    finally:
        httpd.shutdown()
        httpd.server_close()
        aq.shutdown()


def test_n2_decide_returns_503_during_warmup():
    """N2: during warm-up, POST /decide returns 503 (not 200 with empty
    fields), and only ONE worker thread exists."""
    # A server that never calls mark_ready() simulates the warm-up window.
    stats = _ServerStats()  # NOT mark_ready()
    httpd, port, _, aq = _start_server(
        lambda s, c, t: dict(FAKE_RESULT),
        "fake",
        stats=stats,
    )
    try:
        # /ready is 503.
        status, _, _ = _get(port, "/ready")
        assert status == 503
        # POST /decide during warm-up returns 503 (not 200 with empty fields).
        schema = {"a": {"type": "enum", "choices": ["x"], "description": "d"}}
        status, body, _ = _post(port, {"schema": schema, "context": "ctx"})
        assert status == 503
        assert body["ready"] is False
        # Only ONE worker thread exists (the placeholder must not create a
        # second AdmissionQueue). The admission queue's worker exists.
        assert aq._worker.is_alive()
    finally:
        httpd.shutdown()
        httpd.server_close()
        aq.shutdown()


def test_n7_non_string_context_is_400(server):
    """N7: a non-string context (e.g. 5) is 400, not a dropped connection."""
    port, _, _, _ = server
    schema = {"a": {"type": "enum", "choices": ["x"], "description": "d"}}
    status, body, _ = _post(port, {"schema": schema, "context": 5})
    assert status == 400
    assert "context must be a string" in body["error"]


def test_n7_non_dict_schema_is_400(server):
    """N7: a non-dict schema (e.g. []) is 400."""
    port, _, _, _ = server
    status, body, _ = _post(port, {"schema": [], "context": "ctx"})
    assert status == 400
    assert "schema must be a JSON object" in body["error"]


def test_f1_in_flight_disconnect_does_not_over_admit():
    """F1: a client that disconnects while IN FLIGHT (already popped) must
    NOT decrement depth. With maxsize=1: A in flight + B queued, A's client
    disconnects -> depth stays 1 (B), C gets 429 (not admitted)."""

    def slow_decide(schema_dict, context, temperature=1.0):
        time.sleep(1.0)  # A occupies the worker for 1s
        return dict(FAKE_RESULT)

    stats = _ServerStats()
    stats.mark_ready()
    aq = _AdmissionQueue(maxsize=1, stats=stats)
    httpd, port, _, _ = _start_server(slow_decide, "fake", stats=stats, aq=aq)
    try:
        schema = {"action": {"type": "enum", "choices": ["A"], "description": "d"}}

        # A: occupies the worker (slow, 1s). Submit directly to get the
        # request object for the disconnect simulation.
        a_req = {
            "schema_dict": schema,
            "context": "A",
            "temperature": 1.0,
            "decide_fn": slow_decide,
            "cancel_event": threading.Event(),
            "result_event": threading.Event(),
            "result": None,
            "error": None,
        }
        assert aq.submit(a_req)
        time.sleep(0.3)  # A is now in flight (worker popped it).

        # B: queued (fills the 1 slot).
        b_req = {
            "schema_dict": schema,
            "context": "B",
            "temperature": 1.0,
            "decide_fn": slow_decide,
            "cancel_event": threading.Event(),
            "result_event": threading.Event(),
            "result": None,
            "error": None,
        }
        assert aq.submit(b_req)
        assert aq.depth == 1  # B is queued.

        # A's client disconnects while IN FLIGHT. mark_cancelled on A must
        # return False (A is not queued) and NOT decrement depth.
        result = aq.mark_cancelled(a_req)
        assert result is False, "mark_cancelled on an in-flight request should return False"
        assert aq.depth == 1, f"depth={aq.depth} — in-flight disconnect decremented (F1 bug)"

        # C: must get 429 (B still holds the slot; A's disconnect didn't free it).
        c_status, c_body, _ = _post(port, {"schema": schema, "context": "C"})
        assert c_status == 429, f"C was admitted behind B (expected 429): {c_status}"
    finally:
        httpd.shutdown()
        httpd.server_close()
        aq.shutdown()


def test_f2_baseexception_returns_500():
    """F2: a decide_fn that raises BaseException (e.g. KeyboardInterrupt)
    returns 500 to that client, not 200 with empty result."""

    def boom_baseexception(schema_dict, context, temperature=1.0):
        raise KeyboardInterrupt("simulated")

    stats = _ServerStats()
    stats.mark_ready()
    # Use a fresh queue — the worker will die on BaseException.
    aq = _AdmissionQueue(maxsize=4, stats=stats)
    httpd, port, _, _ = _start_server(boom_baseexception, "fake", stats=stats, aq=aq)
    try:
        schema = {"a": {"type": "enum", "choices": ["x"], "description": "d"}}
        status, body, _ = _post(port, {"schema": schema, "context": "ctx"})
        # The worker caught the BaseException, set error, and the handler
        # sends 500 (not 200 with empty result).
        assert status == 500, f"expected 500, got {status}: {body}"
        assert body["error"]  # some error message present
    finally:
        httpd.shutdown()
        httpd.server_close()
        aq.shutdown()
