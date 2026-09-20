/**
 * B12 TypeScript client tests (node:test, no network).
 *
 * The global fetch is swapped with a capture function that records the
 * outgoing URL/method/headers/body and returns a canned Response. Fixture
 * responses are the EXACT bytes the server produces (dumped by
 * tests/test_dump_serve_fixtures.py), so the client parses real shapes.
 */

import { describe, it, before, after, beforeEach, afterEach } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

import { JevmlxClient, JevmlxError } from "../src/index.ts";

const __dirname = dirname(fileURLToPath(import.meta.url));
const FIXTURES = join(__dirname, "fixtures");

/** Load a fixture JSON file. */
function fixture(name: string): { status: number; headers: Record<string, string>; body: unknown } {
  return JSON.parse(readFileSync(join(FIXTURES, name), "utf-8"));
}

/** A minimal fetch replacement that captures the request and returns a canned response. */
function captureFetch(canned: { status: number; headers: Record<string, string>; body: unknown }) {
  const calls: Array<{ url: string; method: string; headers: Record<string, string>; body: string | null }> = [];
  const fakeFetch = async (url: string | URL | Request, init?: RequestInit): Promise<Response> => {
    const u = typeof url === "string" ? url : url.toString();
    calls.push({
      url: u,
      method: init?.method ?? "GET",
      headers: (init?.headers as Record<string, string>) ?? {},
      body: init?.body ? String(init.body) : null,
    });
    const bodyText = JSON.stringify(canned.body);
    return new Response(bodyText, {
      status: canned.status,
      headers: canned.headers,
    });
  };
  return { fakeFetch, calls };
}

const originalFetch = globalThis.fetch;

after(() => {
  globalThis.fetch = originalFetch;
});

describe("JevmlxClient — decide", () => {
  it("posts to /decide with the right URL, method, headers, body", async () => {
    const fx = fixture("decide-200.json");
    const { fakeFetch, calls } = captureFetch(fx);
    const client = new JevmlxClient({ baseUrl: "http://localhost:8000", fetch: fakeFetch });

    const schema = { action: { type: "enum", choices: ["APPROVE", "DENY"], description: "d" } };
    const result = await client.decide(schema, "ctx here", { temperature: 0.7 });

    assert.equal(calls.length, 1);
    assert.equal(calls[0].url, "http://localhost:8000/decide");
    assert.equal(calls[0].method, "POST");
    assert.equal(calls[0].headers["Content-Type"], "application/json");
    const sent = JSON.parse(calls[0].body!);
    assert.deepEqual(sent.schema, schema);
    assert.equal(sent.context, "ctx here");
    assert.equal(sent.temperature, 0.7);

    // The parsed body matches the fixture.
    assert.equal(result.request_id, fx.body.request_id);
    assert.equal(result.queue_depth, fx.body.queue_depth);
    assert.equal(result.parsed_json.action.value, "APPROVE");
  });

  it("omits temperature when not provided", async () => {
    const fx = fixture("decide-200.json");
    const { fakeFetch, calls } = captureFetch(fx);
    const client = new JevmlxClient({ fetch: fakeFetch });

    await client.decide({ a: { type: "boolean" } }, "x");
    const sent = JSON.parse(calls[0].body!);
    assert.equal("temperature" in sent, false);
  });
});

describe("JevmlxClient — systemOne", () => {
  it("posts to /v1/systemone and parses the typed response", async () => {
    const fx = fixture("systemone-200.json");
    const { fakeFetch, calls } = captureFetch(fx);
    const client = new JevmlxClient({ fetch: fakeFetch });

    const questions = {
      q1: { type: "choice" as const, criteria: { A: "a", B: "b" } },
      q2: { type: "noul" as const },
      q3: { type: "score" as const, criteria: { none: "n", low: "l", med: "m", high: "h" } },
    };
    const result = await client.systemOne({ user: "alice" }, questions);

    assert.equal(calls[0].url, "http://127.0.0.1:8000/v1/systemone");
    assert.equal(result.model, "fake");
    assert.equal(result.answers.q1.type, "choice");
    assert.equal(result.answers.q1.choice, "A");
    assert.equal(result.answers.q2.type, "noul");
    assert.equal(typeof result.answers.q2.noul, "number");
    assert.equal(result.answers.q3.type, "score");
    assert.equal(result.answers.q3.legend["0"], "none");
    assert.equal(result.answers.q3.legend["3"], "high");
  });
});

describe("JevmlxClient — models / health / ready", () => {
  it("GET /v1/models", async () => {
    const fx = fixture("models-200.json");
    const { fakeFetch, calls } = captureFetch(fx);
    const client = new JevmlxClient({ fetch: fakeFetch });

    const result = await client.models();
    assert.equal(calls[0].method, "GET");
    assert.equal(calls[0].url, "http://127.0.0.1:8000/v1/models");
    assert.equal(result.data[0].id, "fake");
    assert.equal(result.data[0].owned_by, "jevmlx");
  });

  it("GET /health", async () => {
    const fx = fixture("health-200.json");
    const { fakeFetch, calls } = captureFetch(fx);
    const client = new JevmlxClient({ fetch: fakeFetch });

    const result = await client.health();
    assert.equal(calls[0].url, "http://127.0.0.1:8000/health");
    assert.equal(result.ok, true);
    assert.equal(result.worker_alive, true);
    assert.equal(typeof result.queue_depth, "number");
    assert.equal(typeof result.queue_capacity, "number");
  });

  it("GET /ready (200)", async () => {
    const { fakeFetch } = captureFetch({ status: 200, headers: {}, body: { ready: true, model: "fake" } });
    const client = new JevmlxClient({ fetch: fakeFetch });
    const result = await client.ready();
    assert.equal(result.ready, true);
  });
});

describe("JevmlxClient — errors", () => {
  it("429 -> JevmlxError with retryAfterMs (from Retry-After header)", async () => {
    const fx = fixture("decide-429.json");
    const { fakeFetch } = captureFetch(fx);
    const client = new JevmlxClient({ fetch: fakeFetch });

    await assert.rejects(
      client.decide({ a: { type: "boolean" } }, "x"),
      (err: unknown) => {
        assert.ok(err instanceof JevmlxError);
        assert.equal(err.status, 429);
        assert.equal(err.retryAfterMs, 2000); // 2 seconds -> 2000ms
        const body = err.body as { error: string; retry_after_s: number };
        assert.equal(body.error, "admission queue full");
        assert.equal(body.retry_after_s, 2);
        return true;
      },
    );
  });

  it("413 -> JevmlxError with admission limit body", async () => {
    const fx = fixture("decide-413.json");
    const { fakeFetch } = captureFetch(fx);
    const client = new JevmlxClient({ fetch: fakeFetch });

    await assert.rejects(
      client.decide({ a: { type: "enum", choices: ["x", "y"] } }, "ctx"),
      (err: unknown) => {
        assert.ok(err instanceof JevmlxError);
        assert.equal(err.status, 413);
        assert.equal(err.retryAfterMs, null); // no Retry-After on 413
        const body = err.body as { limit: string; value: number; ceiling: number };
        assert.equal(typeof body.limit, "string");
        assert.equal(typeof body.value, "number");
        assert.equal(typeof body.ceiling, "number");
        return true;
      },
    );
  });

  it("503 -> JevmlxError (warming up)", async () => {
    const fx = fixture("ready-503.json");
    const { fakeFetch } = captureFetch(fx);
    const client = new JevmlxClient({ fetch: fakeFetch });

    await assert.rejects(
      client.ready(),
      (err: unknown) => {
        assert.ok(err instanceof JevmlxError);
        assert.equal(err.status, 503);
        assert.equal(err.retryAfterMs, null);
        return true;
      },
    );
  });

  it("400 -> JevmlxError with error message", async () => {
    const { fakeFetch } = captureFetch({
      status: 400,
      headers: {},
      body: { error: "bad request: missing context" },
    });
    const client = new JevmlxClient({ fetch: fakeFetch });

    await assert.rejects(
      client.decide({ a: { type: "boolean" } }, "x"),
      (err: unknown) => {
        assert.ok(err instanceof JevmlxError);
        assert.equal(err.status, 400);
        assert.equal(err.message, "bad request: missing context");
        return true;
      },
    );
  });

  it("500 -> JevmlxError", async () => {
    const { fakeFetch } = captureFetch({ status: 500, headers: {}, body: { error: "boom" } });
    const client = new JevmlxClient({ fetch: fakeFetch });

    await assert.rejects(
      client.decide({ a: { type: "boolean" } }, "x"),
      (err: unknown) => err instanceof JevmlxError && err.status === 500,
    );
  });
});

describe("JevmlxClient — timeout", () => {
  it("aborts and throws JevmlxError when the timeout fires", async () => {
    // A fetch that rejects on abort (mimics real fetch AbortError behavior).
    const fakeFetch = (_url: string, init?: RequestInit) =>
      new Promise<Response>((_resolve, reject) => {
        init?.signal?.addEventListener("abort", () => {
          reject(new DOMException("aborted", "AbortError"));
        });
      });
    const client = new JevmlxClient({ fetch: fakeFetch, timeoutMs: 50 });

    await assert.rejects(
      client.decide({ a: { type: "boolean" } }, "x"),
      (err: unknown) => {
        assert.ok(err instanceof JevmlxError);
        assert.equal(err.status, 0);
        assert.match(err.message, /timed out after 50ms/);
        return true;
      },
    );
  });

  it("per-call timeoutMs overrides the client default", async () => {
    const fakeFetch = (_url: string, init?: RequestInit) =>
      new Promise<Response>((_resolve, reject) => {
        init?.signal?.addEventListener("abort", () => {
          reject(new DOMException("aborted", "AbortError"));
        });
      });
    const client = new JevmlxClient({ fetch: fakeFetch, timeoutMs: 10_000 });

    await assert.rejects(
      client.decide({ a: { type: "boolean" } }, "x", { timeoutMs: 30 }),
      (err: unknown) => err instanceof JevmlxError && err.message.includes("30ms"),
    );
  });
});

describe("JevmlxClient — config", () => {
  it("baseUrl trailing slash is stripped", async () => {
    const { fakeFetch, calls } = captureFetch(fixture("health-200.json"));
    const client = new JevmlxClient({ baseUrl: "http://localhost:8000///", fetch: fakeFetch });
    await client.health();
    assert.equal(calls[0].url, "http://localhost:8000/health");
  });

  it("default baseUrl comes from JEVMLX_BASE_URL env", async () => {
    const prev = process.env.JEVMLX_BASE_URL;
    process.env.JEVMLX_BASE_URL = "http://example.test:9999";
    try {
      const { fakeFetch, calls } = captureFetch(fixture("health-200.json"));
      const client = new JevmlxClient({ fetch: fakeFetch });
      await client.health();
      assert.equal(calls[0].url, "http://example.test:9999/health");
    } finally {
      if (prev === undefined) delete process.env.JEVMLX_BASE_URL;
      else process.env.JEVMLX_BASE_URL = prev;
    }
  });

  it("external AbortSignal aborts the request", async () => {
    const fakeFetch = (_url: string, init?: RequestInit) =>
      new Promise<Response>((_resolve, reject) => {
        init?.signal?.addEventListener("abort", () => {
          reject(new DOMException("aborted", "AbortError"));
        });
      });
    const client = new JevmlxClient({ fetch: fakeFetch, timeoutMs: 60_000 });
    const controller = new AbortController();

    const promise = client.decide({ a: { type: "boolean" } }, "x", { signal: controller.signal });
    setTimeout(() => controller.abort(), 10);
    await assert.rejects(
      promise,
      (err: unknown) => err instanceof JevmlxError && err.status === 0,
    );
  });
});
