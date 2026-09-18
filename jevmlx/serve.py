"""`jevmlx serve`: HTTP wrapper around the parallel decision engine.

    jevmlx serve --model mlx-community/Qwen2.5-1.5B-Instruct-4bit --port 8000

POST /decide  {"schema": {...}, "context": "...", "temperature": 1.0}
GET  /health  -> {"ok": true, "model": M, "busy": bool, "requests_served": n}
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, HTTPServer

logger = logging.getLogger(__name__)


class _ServerStats:
    """Request counters for the handler. Plain attributes: HTTPServer is
    single-threaded and serial by design (one Metal GPU)."""

    def __init__(self) -> None:
        self.requests_served = 0
        self.busy = False

    def begin_request(self) -> None:
        self.busy = True

    def end_request(self) -> None:
        self.busy = False
        self.requests_served += 1


def make_handler(
    decide_fn: Callable[[dict, str, float | None], dict],
    model_id: str,
    stats: _ServerStats | None = None,
):
    """Build a request handler around decide_fn(schema_dict, context, temperature).

    decide_fn is the seam tests use: a callable that returns the result dict
    (or raises ValueError for schema errors, anything else for 500s).
    """
    stats = stats or _ServerStats()

    class Handler(BaseHTTPRequestHandler):
        # Serial by design: one Metal GPU, so requests are processed one at a
        # time. Revisit with a queue/batching layer only if concurrency matters.

        def _send(self, code: int, payload: dict) -> None:
            body = json.dumps(payload, default=str).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path == "/health":
                self._send(
                    200,
                    {
                        "ok": True,
                        "model": model_id,
                        "busy": stats.busy,
                        "requests_served": stats.requests_served,
                    },
                )
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self) -> None:
            if self.path != "/decide":
                self._send(404, {"error": "not found"})
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
            try:
                stats.begin_request()
                result = decide_fn(schema_dict, context, temperature)
            except ValueError as e:  # StructuredSchema schema errors
                self._send(400, {"error": str(e)})
                return
            except Exception as e:
                logger.exception("decide failed")  # full traceback via logging
                first_line = str(e).splitlines() or [type(e).__name__]
                self._send(500, {"error": first_line[0]})
                return
            finally:
                stats.end_request()
            logger.info(
                "POST /decide 200",
                extra={"elapsed_ms": result.get("elapsed_ms")},
            )
            self._send(200, result)

    return Handler


def serve(model_id: str, host: str = "127.0.0.1", port: int = 8000) -> None:
    """Load the model once, then serve decisions until interrupted."""
    from jevmlx.cli import _engine
    from jevmlx.schema import StructuredSchema

    load_engine, run_parallel_generation = _engine()
    logger.info("Loading %s ...", model_id)
    model, tokenizer = load_engine(model_id)

    def decide_fn(schema_dict: dict, context: str, temperature: float | None = 1.0) -> dict:
        schema = StructuredSchema(schema_dict)
        return run_parallel_generation(model, tokenizer, context, schema, temperature=temperature)

    _serve_handler(decide_fn, model_id, host, port)


def _serve_handler(decide_fn, model_id: str, host: str, port: int) -> None:
    """Bind + serve forever (the engine-free seam: tests inject decide_fn)."""
    server = HTTPServer((host, port), make_handler(decide_fn, model_id, _ServerStats()))
    logger.info(
        "jevmlx serving %s on http://%s:%s", model_id, host or "127.0.0.1", server.server_port
    )
    server.serve_forever()
