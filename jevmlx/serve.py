"""`jevmlx serve`: HTTP wrapper around the parallel decision engine.

    jevmlx serve --model mlx-community/Qwen2.5-1.5B-Instruct-4bit --port 8000

POST /decide  {"schema": {...}, "context": "...", "temperature": 1.0}
GET  /health  process liveness + worker-alive + queue telemetry
GET  /ready   503 until model load + warm-up complete, then 200

W6-B7 (GPT-REVIEW-3): bounded admission queue, 429 backpressure with
Retry-After, /health vs /ready split, echoed/generated request ids, queue
depth + wait telemetry, client-disconnect cancellation, and hard
admission limits (max expanded rows, max prompt tokens) rejected with 413.
The projected-memory limit was dropped — the engine already chunks rows to
its measured budget.
"""

from __future__ import annotations

import gc
import json
import logging
import queue
import threading
import time
import uuid
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Admission limits (W6-B7 item 7)
# ---------------------------------------------------------------------------

#: Default max expanded rows across all schema fields. A schema that expands
#: to more scoring rows than this is rejected with 413 before any model work.
DEFAULT_MAX_ROWS = 2048

#: Default max prompt tokens (the context string, tokenized).
DEFAULT_MAX_PROMPT_TOKENS = 8192

#: Default bounded admission queue depth. Requests beyond the queue + the
#: in-flight worker are rejected with 429 + Retry-After.
DEFAULT_QUEUE_SIZE = 16

#: Fixed Retry-After (seconds) returned with every 429. A documented constant,
#: not a derived value (B7 review M3: depth-in-requests is not seconds).
RETRY_AFTER_SECONDS = 2


class AdmissionError(Exception):
    """A request rejected by the hard admission limits (413)."""

    def __init__(self, message: str, *, limit: str, value: int, ceiling: int):
        super().__init__(message)
        self.limit = limit
        self.value = value
        self.ceiling = ceiling


class _NotReadyError(Exception):
    """Raised by the placeholder decide_fn during warm-up (N2: 503, not 200)."""


# ---------------------------------------------------------------------------
# Server stats + ready gate
# ---------------------------------------------------------------------------


class _ServerStats:
    """Request counter + ready/worker-alive gates for the handler.

    The HTTP server is threaded (ThreadingHTTPServer) but the decide worker
    is serial: one request runs on the Metal GPU at a time. The admission
    queue holds the rest. ``queue_depth`` is reported in /health and in each
    decide response.
    """

    def __init__(self) -> None:
        self._requests_served = 0
        self._lock = threading.Lock()
        self._ready = threading.Event()
        self._worker_alive = threading.Event()
        self._worker_alive.set()

    def mark_ready(self) -> None:
        self._ready.set()

    def mark_worker_dead(self) -> None:
        self._worker_alive.clear()

    def mark_worker_alive(self) -> None:
        self._worker_alive.set()

    @property
    def ready(self) -> bool:
        return self._ready.is_set() and self._worker_alive.is_set()

    @property
    def worker_alive(self) -> bool:
        return self._worker_alive.is_set()

    @property
    def requests_served(self) -> int:
        with self._lock:
            return self._requests_served

    def record_served(self) -> None:
        with self._lock:
            self._requests_served += 1


# ---------------------------------------------------------------------------
# Admission queue + serial worker (B7 items 1, 2, 6)
# ---------------------------------------------------------------------------


class _AdmissionQueue:
    """A bounded admission queue feeding a single serial worker.

    B1 strict (round 4/F1): the bound is enforced by a ``_live_queued``
    counter + lock. Per-request state (``request['queued']``) decides whether
    ``mark_cancelled`` applies — a client that disconnects while IN FLIGHT
    (already popped) does NOT decrement the counter (only queued entries do).
    This prevents over-admission from in-flight disconnects and stale
    ``id()`` reuse. With maxsize=1: A in flight + B queued, A's client
    disconnects -> depth stays 1 (B), C gets 429.

    ``depth`` is the live count of queued (not-yet-popped, not-cancelled)
    entries.
    """

    def __init__(
        self,
        maxsize: int = DEFAULT_QUEUE_SIZE,
        stats: _ServerStats | None = None,
    ) -> None:
        self._q: queue.Queue[dict | None] = queue.Queue()  # unbounded
        self._max = maxsize
        self._stats = stats
        self._live_queued = 0
        self._lock = threading.Lock()
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()

    @property
    def depth(self) -> int:
        """Live entries in the queue (queued, not cancelled, not popped)."""
        with self._lock:
            return self._live_queued

    @property
    def capacity(self) -> int:
        return self._max

    def submit(self, request: dict) -> bool:
        """Try to enqueue. Returns True if accepted, False if full (429).

        F1: sets ``request['queued'] = True`` under the lock so mark_cancelled
        can check per-request state (not a global counter inference). N3:
        ``queued_at`` is stamped here; the worker records ``queue_wait_ms``
        when it picks the request up.
        """
        with self._lock:
            if self._live_queued >= self._max:
                return False
            self._live_queued += 1
            request["queued"] = True
        request["queued_at"] = time.perf_counter()
        self._q.put(request)
        return True

    def mark_cancelled(self, request: dict) -> bool:
        """B1 strict: mark a queued entry cancelled and free its slot.

        F1: acts ONLY when ``request['queued']`` is True (per-request state,
        not a global counter). An in-flight request (already popped, queued
        cleared) does NOT decrement — no over-admission from in-flight
        disconnects. Returns True if the entry was still queued (and is now
        cancelled), False if it was already popped by the worker.
        """
        with self._lock:
            if not request.get("queued"):
                return False  # worker already popped it (in-flight)
            request["queued"] = False
            request["cancelled"] = True
            self._live_queued -= 1
            return True

    def _run(self) -> None:
        # M5: the worker loop is wrapped in try/except BaseException — any
        # uncaught exception marks the worker dead so /ready goes 503.
        # F2: a BaseException from decide_fn sets request['error'] before
        # re-raising so the handler sends 500 (not 200 with empty result).
        try:
            while True:
                request = self._q.get()
                if request is None:
                    return
                cancel = request["cancel_event"]
                # N3: measure queue wait until the worker picks it up.
                queued_at = request.get("queued_at")
                if queued_at is not None:
                    request["queue_wait_ms"] = round((time.perf_counter() - queued_at) * 1000.0, 2)
                # F1: clear the queued flag under the lock (per-request state).
                # If mark_cancelled already cleared it, the request is cancelled.
                with self._lock:
                    was_queued = request.get("queued", False)
                    request["queued"] = False
                    if was_queued:
                        self._live_queued = max(0, self._live_queued - 1)
                cancelled = request.get("cancelled", False)
                if cancel.is_set() or cancelled:
                    request["result_event"].set()
                    continue
                try:
                    request["result"] = request["decide_fn"](
                        request["schema_dict"],
                        request["context"],
                        request["temperature"],
                    )
                    request["error"] = None
                except Exception as e:  # noqa: BLE001 — M5: worker survives
                    request["error"] = e
                except BaseException as e:  # noqa: BLE001 — F2: 500 not 200
                    request["error"] = e
                    request["result_event"].set()
                    if self._stats:
                        self._stats.mark_worker_alive()
                    raise
                finally:
                    request["result_event"].set()
                    if self._stats:
                        self._stats.mark_worker_alive()
        except BaseException:
            if self._stats:
                self._stats.mark_worker_dead()
            raise

    def shutdown(self) -> None:
        self._q.put(None)
        self._worker.join(timeout=2.0)


# ---------------------------------------------------------------------------
# Request id
# ---------------------------------------------------------------------------

# B7 review minor: cap the echoed request id to avoid header-injection /
# unbounded-length abuse.
_MAX_REQUEST_ID_LEN = 128


def _request_id(headers) -> str:
    """Echo the client's X-Request-Id (capped), or generate one."""
    rid = headers.get("X-Request-Id")
    if rid and len(rid) <= _MAX_REQUEST_ID_LEN:
        # Strip newlines to prevent header injection.
        return rid.replace("\r", "").replace("\n", "")
    return f"jev-{uuid.uuid4().hex[:12]}"


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# B3: /v1/systemone request adapter
# ---------------------------------------------------------------------------


def _systemone_to_schema(questions: dict) -> dict:
    """Map a systemone questions block to a jevmlx schema dict.

    choice -> enum with criteria as name->description.
    noul   -> boolean.
    score  -> enum ordered=True with criteria order as the level order.

    Raises ValueError with the question id named on an unknown type.
    """
    schema: dict[str, dict] = {}
    for qid, qspec in questions.items():
        if not isinstance(qspec, dict):
            raise ValueError(f"question '{qid}' spec must be an object")
        qtype = qspec.get("type")
        instructions = qspec.get("instructions", "")
        criteria = qspec.get("criteria", {})
        if qtype == "choice":
            if isinstance(criteria, dict):
                choices = list(criteria.keys())
            elif isinstance(criteria, list):
                choices = list(criteria)
            else:
                raise ValueError(f"question '{qid}' criteria must be object or array")
            schema[qid] = {
                "type": "enum",
                "description": instructions,
                "choices": choices,
            }
        elif qtype == "noul":
            schema[qid] = {
                "type": "boolean",
                "description": instructions,
            }
        elif qtype == "score":
            if isinstance(criteria, dict):
                choices = list(criteria.keys())
            elif isinstance(criteria, list):
                choices = list(criteria)
            else:
                raise ValueError(f"question '{qid}' criteria must be object or array")
            schema[qid] = {
                "type": "enum",
                "description": instructions,
                "choices": choices,
                "ordered": True,
            }
        else:
            raise ValueError(f"question '{qid}' has unknown type: {qtype!r}")
    return schema


def _probability_margin(ft: dict) -> float | None:
    """top1 - top2 probability from field_telemetry (None if unavailable)."""
    top = ft.get("top_choices") or []
    if len(top) >= 2:
        return top[0].get("probability", 0.0) - top[1].get("probability", 0.0)
    if len(top) == 1:
        return top[0].get("probability", 0.0)
    return ft.get("probability")


def _format_systemone_response(
    result: dict,
    questions: dict,
    model_id: str,
) -> dict:
    """Build the /v1/systemone response from a decide result.

    choice -> {type:'choice', choice, confidence, probabilities}
    noul   -> {type:'noul', noul: p_true, confidence}
    score  -> {type:'score', score: expected_index, argmax_level, probabilities, legend}
    """
    field_telemetry = result.get("field_telemetry") or {}
    parsed_json = result.get("parsed_json") or {}
    answers: dict[str, dict] = {}
    for qid, qspec in questions.items():
        qtype = qspec.get("type")
        ft = field_telemetry.get(qid, {})
        parsed = parsed_json.get(qid, {})
        probs = {tc["choice"]: tc.get("probability", 0.0) for tc in (ft.get("top_choices") or [])}
        confidence = _probability_margin(ft)
        if qtype == "choice":
            answers[qid] = {
                "type": "choice",
                "choice": parsed.get("value"),
                "confidence": confidence,
                "probabilities": probs,
            }
        elif qtype == "noul":
            # p_true: probability that the boolean is True.
            p_true = 0.0
            if "true" in probs:
                p_true = probs["true"]
            elif ft.get("value") is True:
                p_true = ft.get("probability", 0.0)
            else:
                p_true = probs.get("true", 0.0)
            answers[qid] = {
                "type": "noul",
                "noul": p_true,
                "confidence": confidence,
            }
        elif qtype == "score":
            criteria = qspec.get("criteria", {})
            if isinstance(criteria, dict):
                levels = list(criteria.keys())
            else:
                levels = list(criteria)
            legend = {str(i): levels[i] for i in range(len(levels))}
            argmax_level = parsed.get("value")
            # expected_index = sum(i * p_i) over the ordered levels.
            expected_index = 0.0
            for i, level in enumerate(levels):
                expected_index += i * probs.get(level, 0.0)
            answers[qid] = {
                "type": "score",
                "score": expected_index,
                "argmax_level": argmax_level,
                "probabilities": probs,
                "legend": legend,
            }
    usage = {
        "prompt_tokens": result.get("computed_prompt_token_positions"),
        "computed_positions": result.get("computed_suffix_token_positions"),
    }
    return {
        "model": model_id,
        "answers": answers,
        "usage": usage,
    }


def make_handler(
    decide_fn: Callable[[dict, str, float | None], dict],
    model_id: str,
    stats: _ServerStats | None = None,
    *,
    admission_queue: _AdmissionQueue | None = None,
    max_rows: int = DEFAULT_MAX_ROWS,
    max_prompt_tokens: int = DEFAULT_MAX_PROMPT_TOKENS,
    tokenize_fn: Callable[[str], list[int]] | None = None,
    count_rows_fn: Callable[[dict], int] | None = None,
):
    """Build a request handler around decide_fn(schema_dict, context, temperature).

    ``tokenize_fn`` and ``count_rows_fn`` are the admission-limit seams, both
    REQUIRED. In production ``tokenize_fn`` is the engine's tokenizer and
    ``count_rows_fn`` is ``_count_rows`` (O(schema) arithmetic on the raw
    dict). Tests inject deterministic stand-ins with the SAME call signatures
    — there is no dual-path fallback (M2 round 3: the old chars/4 default is
    gone; tokenize_fn is required, never defaulted).

    N1: the projected-memory limit is DROPPED entirely. The engine already
    chunks rows to its measured width-bin budget. Admission is rows + prompt
    tokens only.
    """
    stats = stats or _ServerStats()
    admission_queue = admission_queue or _AdmissionQueue(stats=stats)

    # M2 round 3: tokenize_fn is REQUIRED (no chars/4 default). count_rows_fn
    # defaults to _count_rows (O(schema) arithmetic, always available).
    if tokenize_fn is None:
        raise TypeError("tokenize_fn is required (no chars/4 fallback)")
    if count_rows_fn is None:
        count_rows_fn = _count_rows

    def _admit(schema_dict: dict, context: str) -> None:
        """Hard admission limits (B7 item 7). Raises AdmissionError (413) or
        ValueError (400) on a bad schema."""
        # B3: a malformed schema raises ValueError here — caught by the
        # caller and returned as 400 (the base contract), not a dropped
        # connection.
        # N1: O(schema) arithmetic, NOT _build_schema_rows.
        rows = count_rows_fn(schema_dict)
        if rows > max_rows:
            raise AdmissionError(
                f"too many scoring rows: {rows} > limit {max_rows}",
                limit="max_rows",
                value=rows,
                ceiling=max_rows,
            )
        n_prompt = len(tokenize_fn(context))
        if n_prompt > max_prompt_tokens:
            raise AdmissionError(
                f"prompt too long: {n_prompt} tokens > limit {max_prompt_tokens}",
                limit="max_prompt_tokens",
                value=n_prompt,
                ceiling=max_prompt_tokens,
            )

    class Handler(BaseHTTPRequestHandler):
        # HTTP/1.0: no keep-alive, no pipelining — simpler disconnect
        # semantics (B7 review M7). One request per connection.

        protocol_version = "HTTP/1.0"

        def _send(self, code: int, payload: dict, *, extra_headers: dict | None = None) -> None:
            body = json.dumps(payload, default=str).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("X-Request-Id", self._rid)
            if extra_headers:
                for k, v in extra_headers.items():
                    self.send_header(k, str(v))
            self.end_headers()
            self.wfile.write(body)

        def _send_429(self) -> None:
            """Full queue: 429 + Retry-After (not 529)."""
            self._send(
                429,
                {
                    "error": "admission queue full",
                    "queue_depth": admission_queue.depth,
                    "queue_capacity": admission_queue.capacity,
                    "retry_after_s": RETRY_AFTER_SECONDS,
                },
                extra_headers={"Retry-After": str(RETRY_AFTER_SECONDS)},
            )

        def _send_413(self, err: AdmissionError) -> None:
            self._send(
                413,
                {
                    "error": str(err),
                    "limit": err.limit,
                    "value": err.value,
                    "ceiling": err.ceiling,
                },
            )

        def do_GET(self) -> None:
            self._rid = _request_id(self.headers)
            if self.path == "/health":
                # /health = process liveness + worker-alive (always 200 if
                # the process is alive; worker_dead is reported but not a
                # 503 — that's /ready's job). Carries queue telemetry.
                self._send(
                    200,
                    {
                        "ok": True,
                        "model": model_id,
                        "worker_alive": stats.worker_alive,
                        "requests_served": stats.requests_served,
                        "queue_depth": admission_queue.depth,
                        "queue_capacity": admission_queue.capacity,
                    },
                )
            elif self.path == "/v1/models":
                # B3: advertise only our resolved model id.
                self._send(
                    200,
                    {"data": [{"id": model_id, "owned_by": "jevmlx"}]},
                )
            elif self.path == "/ready":
                # /ready = 503 until model load + warm-up complete AND the
                # worker is alive, then 200.
                if stats.ready:
                    self._send(200, {"ready": True, "model": model_id})
                else:
                    self._send(503, {"ready": False, "model": model_id})
            else:
                self._send(404, {"error": "not found"})

        def _run_through_queue(
            self, schema_dict: dict, context: str, temperature: float | None = 1.0
        ) -> tuple[int, dict]:
            """Shared path: admit, queue, wait for the serial worker, handle
            errors. Returns (status_code, response_body). Both /decide and
            /v1/systemone call this so backpressure is identical."""
            # N2: 503 before touching the queue during warm-up.
            if not stats.ready:
                return 503, {"error": "server is warming up", "ready": False}
            # Hard admission limits (413).
            try:
                _admit(schema_dict, context)
            except AdmissionError as e:
                return 413, {
                    "error": str(e),
                    "limit": e.limit,
                    "value": e.value,
                    "ceiling": e.ceiling,
                }
            cancel_event = threading.Event()
            result_event = threading.Event()
            request = {
                "schema_dict": schema_dict,
                "context": context,
                "temperature": temperature,
                "decide_fn": decide_fn,
                "cancel_event": cancel_event,
                "result_event": result_event,
                "result": None,
                "error": None,
            }
            if not admission_queue.submit(request):
                return 429, {
                    "error": "admission queue full",
                    "queue_depth": admission_queue.depth,
                    "queue_capacity": admission_queue.capacity,
                    "retry_after_s": RETRY_AFTER_SECONDS,
                }
            # Wait for the worker, watching for client disconnect.
            while not result_event.wait(timeout=0.5):
                if self._client_gone():
                    cancel_event.set()
                    admission_queue.mark_cancelled(request)
                    return 444, {"error": "client disconnected"}
            err = request["error"]
            if err is not None:
                if isinstance(err, _NotReadyError):
                    return 503, {"error": "server is warming up", "ready": False}
                if isinstance(err, ValueError):
                    return 400, {"error": str(err)}
                if isinstance(err, AdmissionError):
                    return 413, {
                        "error": str(err),
                        "limit": err.limit,
                        "value": err.value,
                        "ceiling": err.ceiling,
                    }
                logger.exception("decide failed", exc_info=err)
                first_line = str(err).splitlines() or [type(err).__name__]
                return 500, {"error": first_line[0]}
            result = dict(request["result"] or {})
            result["queue_depth"] = admission_queue.depth
            result["queue_wait_ms"] = request["queue_wait_ms"]
            result["request_id"] = self._rid
            stats.record_served()
            return 200, result

        def _handle_systemone(self) -> None:
            """POST /v1/systemone: map to ONE schema, run through the same
            queue as /decide, format the systemone response."""
            if not stats.ready:
                self._send(503, {"error": "server is warming up", "ready": False})
                return
            try:
                length = int(self.headers.get("Content-Length", 0))
                payload = json.loads(self.rfile.read(length) or b"{}")
                questions = payload["questions"]
                state = payload.get("state", "")
                temperature = payload.get("temperature", 1.0)
            except (json.JSONDecodeError, KeyError) as e:
                self._send(400, {"error": f"bad request: {e}"})
                return
            if not isinstance(questions, dict):
                self._send(400, {"error": "questions must be a JSON object"})
                return
            # Map questions -> schema. ValueError names the bad question id.
            try:
                schema_dict = _systemone_to_schema(questions)
            except ValueError as e:
                self._send(400, {"error": str(e)})
                return
            # state object -> json.dumps(indent=2) as context.
            if isinstance(state, (dict, list)):
                context = json.dumps(state, indent=2)
            elif isinstance(state, str):
                context = state
            else:
                self._send(400, {"error": "state must be a string or object"})
                return
            code, result = self._run_through_queue(schema_dict, context, temperature)
            if code != 200:
                # 429 carries Retry-After.
                if code == 429:
                    self._send(
                        code, result, extra_headers={"Retry-After": str(RETRY_AFTER_SECONDS)}
                    )
                else:
                    self._send(code, result)
                return
            response = _format_systemone_response(result, questions, model_id)
            logger.info("POST /v1/systemone 200", extra={"elapsed_ms": result.get("elapsed_ms")})
            self._send(200, response)

        def do_POST(self) -> None:
            self._rid = _request_id(self.headers)
            if self.path == "/v1/systemone":
                self._handle_systemone()
                return
            if self.path != "/decide":
                self._send(404, {"error": "not found"})
                return
            # N2: during warm-up, return 503 BEFORE touching the admission
            # queue.
            if not stats.ready:
                self._send(503, {"error": "server is warming up", "ready": False})
                return
            try:
                length = int(self.headers.get("Content-Length", 0))
                payload = json.loads(self.rfile.read(length) or b"{}")
                schema_dict = payload["schema"]
                context = payload["context"]
                temperature = payload.get("temperature", 1.0)
            except (json.JSONDecodeError, KeyError) as e:
                self._send(400, {"error": f"bad request: {e}"})
                return

            # N7: validate types — schema must be a dict, context must be a
            # str.
            if not isinstance(schema_dict, dict):
                self._send(400, {"error": "schema must be a JSON object"})
                return
            if not isinstance(context, str):
                self._send(400, {"error": "context must be a string"})
                return

            code, result = self._run_through_queue(schema_dict, context, temperature)
            if code == 429:
                self._send(code, result, extra_headers={"Retry-After": str(RETRY_AFTER_SECONDS)})
            elif code != 200:
                self._send(code, result)
            else:
                logger.info(
                    "POST /decide 200",
                    extra={
                        "elapsed_ms": result.get("elapsed_ms"),
                        "queue_wait_ms": result.get("queue_wait_ms"),
                    },
                )
                self._send(200, result)

        def _client_gone(self) -> bool:
            """Detect a disconnected client via select (no byte stealing).

            B7 review M7: the old non-blocking read(1) stole a byte from a
            pipelining client. With HTTP/1.0 there's no pipelining, but we
            still use select to check the socket for an error/EOF without
            consuming data.
            """
            import select

            try:
                r, _w, x = select.select([self.rfile], [], [self.connection], 0)
                if x or (r and self.rfile.peek(1) == b""):
                    return True  # exceptional condition or EOF
            except (OSError, ValueError):
                return True
            return False

        def log_message(self, fmt, *args) -> None:
            logger.debug("%s - %s", self.address_string(), fmt % args)

    return Handler


# ---------------------------------------------------------------------------
# O(schema) row counting (N1: NOT _build_schema_rows)
# ---------------------------------------------------------------------------


def _count_rows(schema_dict: dict) -> int:
    """Count expanded scoring rows from the raw schema dict.

    O(schema size) arithmetic: enum -> len(choices), multi -> len(choices)+1
    (one Y/N row per option + one count row), bool -> 2, count/other -> 1.
    This is an UPPER bound (the real trie shares prefixes), which is safe for
    an admission limit (reject-too-much beats reject-too-little). It does NOT
    call _build_schema_rows (38 s for a 200-choice enum with a real tokenizer).
    """
    total = 0
    for _name, spec in (schema_dict or {}).items():
        if not isinstance(spec, dict):
            continue
        ftype = spec.get("type")
        choices = spec.get("choices") or []
        if ftype == "enum":
            total += max(1, len(choices))
        elif ftype == "multi":
            total += max(1, len(choices)) + 1  # options + count row
        elif ftype == "boolean":
            total += 2
        else:
            total += 1
    return max(1, total)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def serve(
    model_id: str,
    host: str = "127.0.0.1",
    port: int = 8000,
    *,
    queue_size: int = DEFAULT_QUEUE_SIZE,
    max_rows: int = DEFAULT_MAX_ROWS,
    max_prompt_tokens: int = DEFAULT_MAX_PROMPT_TOKENS,
) -> None:
    """Load the model once, then serve decisions until interrupted.

    B7 review M6: the port binds BEFORE warm-up so /ready is reachable
    during the warm-up window (a 503 is a real response, not a connection
    refusal). Warm-up runs in the main thread; /ready flips to 200 when it
    completes.

    N2: during warm-up the bound server returns 503 for /decide (not 200
    with empty fields). Only ONE AdmissionQueue exists — it is created
    after warm-up, not during the placeholder phase.
    """
    from jevmlx.engine import load_engine, run_parallel_generation
    from jevmlx.schema import StructuredSchema

    stats = _ServerStats()

    # N2 round 3: create the ONE AdmissionQueue BEFORE the placeholder so
    # both the placeholder and the real handler share it (no second worker
    # thread). The placeholder's do_POST short-circuits to 503 before
    # touching the queue, so no slot is consumed during warm-up.
    admission_queue = _AdmissionQueue(maxsize=queue_size, stats=stats)

    # M6 + N2: bind the socket first so /ready is reachable during warm-up.
    # The placeholder handler returns 503 for /decide until mark_ready().
    def _not_ready_decide(*args, **kwargs):
        raise _NotReadyError("server is warming up")

    # The placeholder's do_POST short-circuits to 503 BEFORE _admit (the
    # `if not stats.ready` check), so tokenize_fn is never called during
    # warm-up. It exists only to satisfy make_handler's required parameter
    # (no dual path — the real tokenize_fn is wired after warm-up).
    server = ThreadingHTTPServer(
        (host, port),
        make_handler(
            _not_ready_decide,
            model_id,
            stats,
            admission_queue=admission_queue,
            max_rows=max_rows,
            max_prompt_tokens=max_prompt_tokens,
            tokenize_fn=lambda text: [],  # never called (503 before _admit)
            count_rows_fn=_count_rows,
        ),
    )
    listen_port = server.server_port
    threading.Thread(target=server.serve_forever, daemon=True).start()
    logger.info("jevmlx serving %s on http://%s:%s (warming up)", model_id, host, listen_port)

    # Load the model + warm up in THIS thread (the main thread); /ready
    # stays 503 until warm-up completes.
    engine = load_engine(model_id)

    def decide_fn(schema_dict: dict, context: str, temperature: float | None = 1.0) -> dict:
        schema = StructuredSchema(schema_dict)
        return run_parallel_generation(engine, context, schema, temperature=temperature)

    def tokenize_fn(text: str) -> list[int]:
        return engine.tokenizer.encode(text, add_special_tokens=False)

    # N1: O(schema) arithmetic via _count_rows, NOT _build_schema_rows.
    count_rows_fn = _count_rows

    # Replace the placeholder handler with the real one (SAME admission_queue).
    server.RequestHandlerClass = make_handler(
        decide_fn,
        model_id,
        stats,
        admission_queue=admission_queue,
        max_rows=max_rows,
        max_prompt_tokens=max_prompt_tokens,
        tokenize_fn=tokenize_fn,
        count_rows_fn=count_rows_fn,
    )

    # Warm-up: one trivial forward so the first real request doesn't pay
    # the compile cost. /ready stays 503 until this completes.
    try:
        run_parallel_generation(
            engine,
            "warmup",
            StructuredSchema({"x": {"type": "enum", "choices": ["a"], "description": "d"}}),
        )
    except Exception:  # noqa: BLE001 — warm-up failure is logged, not fatal
        logger.exception("warm-up forward failed; /ready will stay 503")
    finally:
        gc.collect()
    stats.mark_ready()

    logger.info("jevmlx ready: %s on http://%s:%s", model_id, host, listen_port)
    try:
        # Block until interrupted.
        threading.Event().wait()
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        server.shutdown()
        server.server_close()
