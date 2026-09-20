# @jevmlx/client

A zero-dependency TypeScript client for the [`jevmlx serve`](../..) HTTP API.

## Install

Install from the GitHub path (no npm publish yet):

```bash
npm install github:bnsd55/jevmlx#main --save
```

Or from a local clone:

```bash
npm install /path/to/jevmlx/js
```

## Usage

```typescript
import { JevmlxClient, JevmlxError } from "@jevmlx/client";

const client = new JevmlxClient({
  baseUrl: process.env.JEVMLX_BASE_URL ?? "http://127.0.0.1:8000",
  timeoutMs: 30_000,
});

// POST /decide — structured decision
const result = await client.decide(
  { action: { type: "enum", choices: ["APPROVE", "DENY"], description: "decision" } },
  "Customer was charged twice and wants a refund.",
  { temperature: 0.7 },
);
console.log(result.parsed_json.action.value);   // "APPROVE"
console.log(result.request_id);                 // echoed server-side
console.log(result.queue_wait_ms);              // admission-queue wait

// POST /v1/systemone — one-shot state + questions
const so = await client.systemOne(
  { user: "alice", amount: 200 },
  {
    category: { type: "choice", criteria: { billing: "billing issue", tech: "technical" } },
    is_urgent: { type: "noul" },
    severity: { type: "score", criteria: { low: "low", med: "medium", high: "high" } },
  },
);
console.log(so.answers.category.choice);        // "billing"
console.log(so.answers.is_urgent.noul);         // 0.0–1.0 (P(true))
console.log(so.answers.severity.legend["2"]);   // "high"

// GET /health — liveness + queue telemetry (always 200 if the process is up)
const h = await client.health();
console.log(h.ok, h.worker_alive, h.queue_depth);

// GET /ready — 503 until model load + warm-up, then 200
const r = await client.ready();
if (!r.ready) { /* still warming up */ }

// GET /v1/models
const m = await client.models();
console.log(m.data[0].id);                      // resolved model id
```

## Errors

Every non-2xx response is thrown as a `JevmlxError`:

```typescript
try {
  await client.decide(schema, context);
} catch (e) {
  if (e instanceof JevmlxError) {
    console.log(e.status);        // HTTP status code (429, 413, 503, 400, 500, …)
    console.log(e.body);          // parsed JSON body (shape depends on the code)
    console.log(e.retryAfterMs);  // null, except 429 (parsed from Retry-After, in ms)
  }
}
```

| Status | When | Body shape | Retry-After |
|--------|------|------------|-------------|
| 400 | Bad request (malformed JSON, missing `schema`/`context`) | `{ error: string }` | — |
| 413 | Admission limit (too many rows or prompt too long) | `{ error, limit, value, ceiling }` | — |
| 429 | Admission queue full | `{ error, queue_depth, queue_capacity, retry_after_s }` | yes (seconds) |
| 503 | Server warming up (model not loaded / worker dead) | `{ error, ready: false }` or `{ ready, model }` | — |
| 500 | Unhandled server error | `{ error: string }` | — |

**No retries are built in** — the caller owns the retry policy. For 429, wait
`retryAfterMs` before retrying.

## Timeouts

Every request has an `AbortController` timeout (default 30s, override per-call
via `timeoutMs`). A timeout throws a `JevmlxError` with `status: 0` and a
`"timed out after Nms"` message. Pass an external `AbortSignal` via
`opts.signal` to cancel an in-flight request.

## API

- `new JevmlxClient({ baseUrl?, timeoutMs?, fetch? })` — `baseUrl` defaults to
  `process.env.JEVMLX_BASE_URL ?? "http://127.0.0.1:8000"`.
- `client.decide(schema, context, opts?)` → `DecideResponse`
- `client.systemOne(state, questions, opts?)` → `SystemOneResponse`
- `client.models()` → `ModelsResponse`
- `client.health()` → `HealthResponse`
- `client.ready()` → `ReadyResponse`

All response types are exported for IDE autocompletion.

## Development

```bash
cd js
npm install      # devDependencies only (tsx, typescript)
npm test         # node:test, no network — swaps global fetch
npm run build    # tsc -> dist/ (ESM + CJS + .d.ts)
npm run typecheck
```

The test fixtures in `tests/fixtures/*.json` are dumped by the Python test
`tests/test_dump_serve_fixtures.py` (run with `python tests/test_dump_serve_fixtures.py`)
so the TS tests parse the SAME bytes the server produces — one source of truth.

## License

MIT
