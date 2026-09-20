/**
 * TypeScript client for the jevmlx serve HTTP API.
 *
 * Zero runtime dependencies — uses the global `fetch` (Node 18+, browsers,
 * Deno, Bun). Methods mirror the five endpoints:
 *
 *   POST /decide         — run a structured decision (schema + context)
 *   POST /v1/systemone   — one-shot state+questions decision
 *   GET  /v1/models      — the resolved model id
 *   GET  /health         — process liveness + queue telemetry (always 200)
 *   GET  /ready          — 503 until model load + warm-up, then 200
 *
 * Every non-2xx response is thrown as a {@link JevmlxError} carrying the
 * status, the parsed body, and (for 429) `retryAfterMs` derived from the
 * `Retry-After` header. No retries are built in — the caller owns that
 * policy. Request timeouts use `AbortController`.
 */

// ---------------------------------------------------------------------------
// Shared types — mirror the server's JSON shapes exactly (one source of
// truth: the fixtures in js/tests/fixtures/*.json, dumped by the server).
// ---------------------------------------------------------------------------

/** One field in a /decide schema. */
export interface SchemaField {
  type: "enum" | "boolean" | "multi" | "count";
  description?: string;
  choices?: string[];
  choice_descriptions?: Record<string, string>;
  ordered?: boolean;
  depends_on?: string;
}

/** A /decide schema = field name -> field spec. */
export type Schema = Record<string, SchemaField>;

/** The full /decide response (the engine result + server-added telemetry). */
export interface DecideResponse {
  parsed_json: Record<string, { value: unknown; prob?: number | null }>;
  field_telemetry: Record<string, Record<string, unknown>>;
  elapsed_ms: number;
  prior_ms: number;
  prefill_ms: number;
  plan_compile_ms: number;
  cache_broadcast_ms: number;
  suffix_eval_ms: number;
  lm_head_gather_ms: number;
  total_ms: number;
  per_item_end_to_end_ms: number;
  padded_token_positions: number;
  naive_branch_prompt_tokens: number;
  shared_prefix_tokens: number;
  logical_suffix_token_positions: number;
  computed_suffix_token_positions: number;
  computed_prompt_token_positions: number;
  retry_wasted_ms: number;
  total_tokens_generated: number;
  peak_active_bytes: number;
  peak_incremental_bytes: number;
  sequential_forward_passes: number;
  rescored_fields: string[];
  failed_attempts: number;
  schema_match: boolean;
  confidence_model: string;
  prompt_sha256: string;
  prompt_version: string;
  probability_status: string;
  prior_correction: boolean;
  constraints_applied: boolean;
  reconciled_fields: string[];
  rerun_fields: string[];
  rerun_rows: number;
  second_pass_ms: number;
  num_fields: number;
  /** Server-added: live queue depth at response time. */
  queue_depth: number;
  /** Server-added: ms spent waiting in the admission queue. */
  queue_wait_ms: number;
  /** Server-added: echoed/generated request id. */
  request_id: string;
  [key: string]: unknown;
}

// --- /v1/systemone types ---------------------------------------------------

/** A choice question for /v1/systemone. */
export interface ChoiceQuestion {
  type: "choice";
  instructions?: string;
  criteria: Record<string, string> | string[];
}

/** A noul (boolean) question for /v1/systemone. */
export interface NoulQuestion {
  type: "noul";
  instructions?: string;
}

/** A score (ordinal) question for /v1/systemone. */
export interface ScoreQuestion {
  type: "score";
  instructions?: string;
  criteria: Record<string, string> | string[];
}

/** Any /v1/systemone question. */
export type SystemOneQuestion = ChoiceQuestion | NoulQuestion | ScoreQuestion;

/** The /v1/systemone request body. */
export interface SystemOneRequest {
  state: string | Record<string, unknown> | unknown[];
  questions: Record<string, SystemOneQuestion>;
  temperature?: number;
}

/** One answer in a /v1/systemone response. */
export interface ChoiceAnswer {
  type: "choice";
  choice: string | null;
  confidence: number | null;
  probabilities: Record<string, number>;
}

export interface NoulAnswer {
  type: "noul";
  noul: number;
  confidence: number | null;
}

export interface ScoreAnswer {
  type: "score";
  score: number;
  argmax_level: string | null;
  probabilities: Record<string, number>;
  legend: Record<string, string>;
}

export type SystemOneAnswer = ChoiceAnswer | NoulAnswer | ScoreAnswer;

/** The /v1/systemone response. */
export interface SystemOneResponse {
  model: string;
  answers: Record<string, SystemOneAnswer>;
  usage: {
    prompt_tokens: number | null;
    computed_positions: number | null;
  };
}

// --- /v1/models, /health, /ready ------------------------------------------

export interface ModelsResponse {
  data: Array<{ id: string; owned_by: string }>;
}

export interface HealthResponse {
  ok: boolean;
  model: string;
  worker_alive: boolean;
  requests_served: number;
  queue_depth: number;
  queue_capacity: number;
}

export interface ReadyResponse {
  ready: boolean;
  model: string;
}

// --- error shapes ---------------------------------------------------------

/** The body of a 413 (admission limit) error. */
export interface AdmissionErrorBody {
  error: string;
  limit: string;
  value: number;
  ceiling: number;
}

/** The body of a 429 (queue full) error. */
export interface QueueFullErrorBody {
  error: string;
  queue_depth: number;
  queue_capacity: number;
  retry_after_s: number;
}

/** The body of a 503 (warming up) error. */
export interface WarmingUpErrorBody {
  error: string;
  ready: false;
}

/**
 * Every non-2xx server response is thrown as a `JevmlxError`.
 *
 * `status` is the HTTP code; `body` is the parsed JSON (shape depends on the
 * code — see {@link AdmissionErrorBody}, {@link QueueFullErrorBody},
 * {@link WarmingUpErrorBody}); `retryAfterMs` is set ONLY for 429 (parsed
 * from the `Retry-After` header, in milliseconds).
 */
export class JevmlxError extends Error {
  readonly status: number;
  readonly body: unknown;
  readonly retryAfterMs: number | null;

  constructor(status: number, body: unknown, retryAfterMs: number | null) {
    const msg =
      typeof (body as { error?: unknown })?.error === "string"
        ? (body as { error: string }).error
        : `HTTP ${status}`;
    super(msg);
    this.name = "JevmlxError";
    this.status = status;
    this.body = body;
    this.retryAfterMs = retryAfterMs;
  }
}

// ---------------------------------------------------------------------------
// Client
// ---------------------------------------------------------------------------

export interface JevmlxClientOptions {
  /** Base URL of the jevmlx serve instance. */
  baseUrl?: string;
  /** Per-request timeout in milliseconds. */
  timeoutMs?: number;
  /** Inject a custom fetch (testing, custom agents). Defaults to global fetch. */
  fetch?: typeof fetch;
}

/** Options for a single /decide or /v1/systemone call. */
export interface DecideCallOptions {
  temperature?: number;
  /** Per-call timeout override (milliseconds). */
  timeoutMs?: number;
  /** Abort an in-flight request via an external signal. */
  signal?: AbortSignal;
}

const DEFAULT_TIMEOUT_MS = 30_000;

/** Read the base URL from the env at construction time (not module load). */
function defaultBaseUrl(): string {
  if (typeof process !== "undefined" && process.env?.JEVMLX_BASE_URL) {
    return process.env.JEVMLX_BASE_URL;
  }
  return "http://127.0.0.1:8000";
}

/**
 * A minimal HTTP client for the jevmlx serve API.
 *
 * @example
 * ```ts
 * const client = new JevmlxClient({ baseUrl: "http://localhost:8000" });
 * const result = await client.decide(schema, "customer was charged twice");
 * console.log(result.parsed_json);
 *
 * try {
 *   await client.decide(schema, "...");
 * } catch (e) {
 *   if (e instanceof JevmlxError && e.status === 429) {
 *     await sleep(e.retryAfterMs!);
 *     // retry...
 *   }
 * }
 * ```
 */
export class JevmlxClient {
  private readonly baseUrl: string;
  private readonly timeoutMs: number;
  private readonly fetchFn: typeof fetch;

  constructor(options: JevmlxClientOptions = {}) {
    this.baseUrl = (options.baseUrl ?? defaultBaseUrl()).replace(/\/+$/, "");
    this.timeoutMs = options.timeoutMs ?? DEFAULT_TIMEOUT_MS;
    this.fetchFn = options.fetch ?? fetch;
  }

  /** POST /decide — run a structured decision. */
  async decide(
    schema: Schema,
    context: string,
    opts: DecideCallOptions = {},
  ): Promise<DecideResponse> {
    const body: Record<string, unknown> = { schema, context };
    if (opts.temperature !== undefined) body.temperature = opts.temperature;
    return this._post<DecideResponse>("/decide", body, opts);
  }

  /** POST /v1/systemone — one-shot state + questions decision. */
  async systemOne(
    state: SystemOneRequest["state"],
    questions: SystemOneRequest["questions"],
    opts: DecideCallOptions = {},
  ): Promise<SystemOneResponse> {
    const body: Record<string, unknown> = { state, questions };
    if (opts.temperature !== undefined) body.temperature = opts.temperature;
    return this._post<SystemOneResponse>("/v1/systemone", body, opts);
  }

  /** GET /v1/models — the resolved model id. */
  async models(): Promise<ModelsResponse> {
    return this._get<ModelsResponse>("/v1/models");
  }

  /** GET /health — process liveness + queue telemetry (always 200 if alive). */
  async health(): Promise<HealthResponse> {
    return this._get<HealthResponse>("/health");
  }

  /** GET /ready — 503 until model load + warm-up, then 200. */
  async ready(): Promise<ReadyResponse> {
    return this._get<ReadyResponse>("/ready");
  }

  // --- internals ---------------------------------------------------------

  private async _get<T>(path: string): Promise<T> {
    const url = this.baseUrl + path;
    const res = await this._fetch(url, { method: "GET" });
    return this._parse<T>(res);
  }

  private async _post<T>(
    path: string,
    body: Record<string, unknown>,
    opts: DecideCallOptions,
  ): Promise<T> {
    const url = this.baseUrl + path;
    const res = await this._fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }, opts);
    return this._parse<T>(res);
  }

  private async _fetch(
    url: string,
    init: RequestInit,
    opts: DecideCallOptions = {},
  ): Promise<Response> {
    const timeoutMs = opts.timeoutMs ?? this.timeoutMs;
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);

    // Merge an external signal (if any) with our timeout signal.
    const signal = opts.signal ?? controller.signal;
    if (opts.signal) {
      opts.signal.addEventListener("abort", () => controller.abort(), { once: true });
    }

    try {
      return await this.fetchFn(url, { ...init, signal });
    } catch (err) {
      if (controller.signal.aborted) {
        throw new JevmlxError(0, { error: `request timed out after ${timeoutMs}ms` }, null);
      }
      throw new JevmlxError(0, { error: String(err) }, null);
    } finally {
      clearTimeout(timer);
    }
  }

  /** Parse a response: 2xx -> body, non-2xx -> JevmlxError. */
  private async _parse<T>(res: Response): Promise<T> {
    const text = await res.text();
    let body: unknown;
    try {
      body = text ? JSON.parse(text) : {};
    } catch {
      body = { error: text };
    }
    if (!res.ok) {
      const retryAfterMs = this._retryAfterMs(res.headers.get("Retry-After"));
      throw new JevmlxError(res.status, body, retryAfterMs);
    }
    return body as T;
  }

  /** Parse a Retry-After header (seconds or HTTP-date) into milliseconds. */
  private _retryAfterMs(header: string | null): number | null {
    if (!header) return null;
    // Seconds form (the server always sends this).
    const seconds = Number(header);
    if (!Number.isNaN(seconds)) return seconds * 1000;
    // HTTP-date form.
    const date = Date.parse(header);
    if (!Number.isNaN(date)) return Math.max(0, date - Date.now());
    return null;
  }
}
