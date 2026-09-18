"""OpenAI-compatible "slots" backend: the native slots decision semantics
through any chat-completions endpoint that returns logprobs.

Same prompt (V1's prompt builder), same aliases, same result-dict shape as
:func:`jevmlx.engine.run_parallel_generation` — different executor. Per field
ONE ``max_tokens=1`` request reads the next-token distribution at the field's
decision row and renormalises it over the quoted alias candidates. Works for
Ollama, oMLX, MTPLX, vLLM, and hosted APIs.

Slower (one request per field instead of one batched pass) and degraded (only
the endpoint's ``top_logprobs`` candidates are observable). Degradation is
explicit: aliases missing from the returned top-k get a floor probability of
``exp(min returned logprob)`` and the field's telemetry flags
``truncated: true``. Probabilities still sum to 1 after renormalisation.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import time
from typing import Any

from jevmlx.engine import PROMPT_V2_SYSTEM
from jevmlx.http import ChatCompletionsError, chat_completions_raw
from jevmlx.schema import StructuredSchema

__all__ = [
    "ChatCompletionsError",
    "OPENAI_SLOTS_PROMPT_VERSION",
    "decide_openai",
]

logger = logging.getLogger(__name__)

OPENAI_SLOTS_PROMPT_VERSION = "jevmlx-openai-slots-v1"

_TOP_LOGPROBS = 20


def _user_content(schema: StructuredSchema, context: str) -> str:
    """The prompt-v2 user turn: alias schema block + delimited context.

    Identical text to the native slots mode's prefill prompt (the system
    paragraph travels in the system message; here and natively).
    """
    schema_str = schema.to_alias_schema_str()
    return f"Classify the following fields.\n\n{schema_str}\n\n<<<CONTEXT\n{context}\nCONTEXT>>>"


def _scalar_messages(schema: StructuredSchema, context: str, name: str) -> list[dict]:
    """Messages for one scalar field: system + user + assistant row prefill.

    The assistant prefix is the JSON decision row, byte-identical to the
    native slots row (``'{\\n  "<field>": '``); the scored next token is the
    quoted alias. Endpoints without assistant-prefill support will treat the
    assistant turn as ordinary history — the payload is still well-formed and
    the top-k read just degrades (logged once per field).
    """
    row = "{\n" + f"  {json.dumps(name)}: "
    return [
        {"role": "system", "content": PROMPT_V2_SYSTEM},
        {"role": "user", "content": _user_content(schema, context)},
        {"role": "assistant", "content": row, "prefix": True},
    ]


def _alias_logprob(entries: list[dict], alias: str) -> float | None:
    """Logprob of a quoted alias in a top_logprobs list, or None.

    Token text varies by endpoint: ``"A``, ``"A``, `` A``, A. Quoted-first
    because the decision row ends with a colon+space inside a JSON string.
    """
    by_text = {e.get("token", ""): float(e.get("logprob", -math.inf)) for e in entries}
    for variant in (f'"{alias}"', f'"{alias}', alias, f" {alias}"):
        if variant in by_text:
            return by_text[variant]
    return None


def _renormalise(logprobs: dict[str, float], aliases: list[str]) -> tuple[dict[str, float], bool]:
    """Renormalise found alias logprobs; floor the missing ones.

    Returns ({alias: probability}, truncated). The floor is exp(min found
    logprob); with nothing found at all the mass splits evenly (full
    truncation) and truncated is True.
    """
    found = {a: lp for a in aliases if (lp := logprobs.get(a)) is not None}
    truncated = len(found) < len(aliases)
    if found:
        floor = math.exp(min(found.values()))
    else:
        floor = 1.0 / len(aliases)
    weights = {a: math.exp(logprobs[a]) if a in logprobs else floor for a in aliases}
    total = sum(weights.values())
    return {a: w / total for a, w in weights.items()}, truncated


def _decide_scalar_field(
    base_url: str,
    model: str,
    api_key: str | None,
    schema: StructuredSchema,
    context: str,
    name: str,
    field,
    timeout: float,
) -> tuple[dict, dict]:
    """One request for one enum/boolean field. Returns (parsed, telemetry)."""
    choices_list = ["true", "false"] if field.field_type == "boolean" else list(field.choices)
    aliases = [schema.alias_for_index(i) for i in range(len(choices_list))]

    messages = _scalar_messages(schema, context, name)
    t0 = time.perf_counter()
    choice = chat_completions_raw(
        base_url,
        model,
        messages,
        api_key=api_key,
        timeout=timeout,
        temperature=0.0,
        extra_payload={"max_tokens": 1, "logprobs": True, "top_logprobs": _TOP_LOGPROBS},
    )
    request_ms = (time.perf_counter() - t0) * 1000

    logprobs_block = choice.get("logprobs") or {}
    content = logprobs_block.get("content") or []
    top_entries = content[0].get("top_logprobs") if content else []
    by_text = {e.get("token", ""): float(e.get("logprob", -math.inf)) for e in (top_entries or [])}

    alias_probs, truncated = _renormalise(
        {a: lp for a in aliases if (lp := _alias_logprob(top_entries or [], a)) is not None},
        aliases,
    )
    if truncated:
        logger.debug(
            "openai_slots: field '%s' truncated at top-%d (floor applied)", name, _TOP_LOGPROBS
        )

    prob_by_choice = {
        choice_text: alias_probs[alias]
        for alias, choice_text in zip(aliases, choices_list, strict=True)
    }
    winner = max(prob_by_choice, key=prob_by_choice.__getitem__)
    probability = prob_by_choice[winner]
    value = (winner.lower() == "true") if field.field_type == "boolean" else winner
    log_scores = {c: math.log(p) if p > 0 else float("-inf") for c, p in prob_by_choice.items()}
    ranked = sorted(log_scores.items(), key=lambda kv: -kv[1])
    alternatives = tuple((c, prob_by_choice[c]) for c, _ in ranked[:3])

    parsed = {"value": value, "prob": probability}
    telemetry = {
        "value": value,
        "type": field.field_type,
        "probability": probability,
        "log_scores": log_scores,
        "alternatives": alternatives,
        "rows": 1,
        "truncated": truncated,
        "request_ms": round(request_ms, 2),
        "top_logprobs_seen": sorted(by_text, key=lambda t: -by_text[t])[:5],
    }
    return parsed, telemetry


def _yes_no_probabilities(entries: list[dict]) -> tuple[dict[str, float], bool]:
    """Renormalised P(yes)/P(no) from a top_logprobs list.

    The aliases are the quoted Y/N the multi rows score (same as the native
    engine's candidates); lowercase yes/no spellings stay tolerated for
    endpoints that normalize the answer. Missing sides floor at exp(min
    found logprob) (0.01 when nothing is found).
    """
    by_text = {e.get("token", ""): float(e.get("logprob", -math.inf)) for e in entries}

    def lp_of(variants: list[str]) -> float | None:
        for v in variants:
            if v in by_text:
                return by_text[v]
        return None

    yes_lp = lp_of(['"Y"', "Y", '"yes"', "yes", " Yes", '"Yes"'])
    no_lp = lp_of(['"N"', "N", '"no"', "no", " No", '"No"'])
    truncated = yes_lp is None or no_lp is None
    if truncated:
        floor = (
            math.exp(min(v for v in (yes_lp, no_lp) if v is not None))
            if (yes_lp is not None or no_lp is not None)
            else 0.01
        )
        yes_lp = math.log(floor) if yes_lp is None else yes_lp
        no_lp = math.log(floor) if no_lp is None else no_lp
    w_yes, w_no = math.exp(yes_lp), math.exp(no_lp)
    total = w_yes + w_no
    return {"yes": w_yes / total, "no": w_no / total}, truncated


def _decide_multi_field(
    base_url: str,
    model: str,
    api_key: str | None,
    schema: StructuredSchema,
    context: str,
    name: str,
    field,
    timeout: float,
    calibration: dict | None,
) -> tuple[dict, dict, int]:
    """One Y/N request per option. Returns (parsed, telemetry, n_requests)."""
    per_option: dict[str, float] = {}
    truncated_any = False
    n_requests = 0
    for option in field.choices:
        # The option row as assistant prefill: the same natural yes/no
        # question the native multi rows pose ('"<field>/<option>": ' with
        # the quoted Y/N aliases), so both backends answer the same question.
        row_key = json.dumps(f"{name}/{option}")
        row = "{\n" + f'  {row_key}: "'
        messages = [
            {"role": "system", "content": PROMPT_V2_SYSTEM},
            {"role": "user", "content": _user_content(schema, context)},
            {"role": "assistant", "content": row, "prefix": True},
        ]
        choice = chat_completions_raw(
            base_url,
            model,
            messages,
            api_key=api_key,
            timeout=timeout,
            temperature=0.0,
            extra_payload={"max_tokens": 1, "logprobs": True, "top_logprobs": _TOP_LOGPROBS},
        )
        n_requests += 1
        logprobs_block = choice.get("logprobs") or {}
        content = logprobs_block.get("content") or []
        top_entries = content[0].get("top_logprobs") if content else []
        probs, truncated = _yes_no_probabilities(top_entries or [])
        per_option[option] = probs["yes"]
        truncated_any = truncated_any or truncated
    # W2-E step 2: same dual contract as the native engine. With calibration
    # ({"multi": {"a", "b"}}) the option's P(yes) is folded to log-odds
    # (log p - log(1-p)), calibrated a*log_odds + b, and selected when > 0;
    # without it the fixed P(yes) >= 0.5 rule stands. The raw probabilities
    # stay in per_option either way.
    multi_ab = calibration.get("multi") if calibration else None
    if multi_ab is not None:
        a_coef, b_coef = multi_ab["a"], multi_ab["b"]
        calibrated = {}
        for option, p in per_option.items():
            p_clamped = min(max(p, 1e-12), 1.0 - 1e-12)
            calibrated[option] = a_coef * math.log(p_clamped / (1.0 - p_clamped)) + b_coef
        selected = [option for option, c in calibrated.items() if c > 0]
        margin = min((abs(c) for c in calibrated.values()), default=0.0)
    else:
        selected = [option for option, p in per_option.items() if p >= 0.5]
        margin = min((abs(p - 0.5) for p in per_option.values()), default=0.0)
    ranked = sorted(per_option.items(), key=lambda kv: -kv[1])
    parsed = {"value": selected, "prob": None}
    telemetry = {
        "value": selected,
        "type": "multi",
        "probability": None,
        "margin": margin,
        "per_option": per_option,
        "alternatives": tuple(ranked),
        "rows": len(field.choices),
        "truncated": truncated_any,
        "calibrated": {"a": multi_ab["a"], "b": multi_ab["b"]} if multi_ab else None,
    }
    return parsed, telemetry, n_requests


def decide_openai(
    base_url: str,
    model: str,
    api_key: str | None,
    schema: StructuredSchema,
    context: str,
    *,
    timeout: float = 120.0,
    calibration: str | dict | None = None,
) -> dict[str, Any]:
    """Decide every schema field through an OpenAI-compatible endpoint.

    Same result-dict shape as :func:`jevmlx.engine.run_parallel_generation`:
    ``parsed_json``, ``field_telemetry`` (``probability``, ``log_scores``,
    ``alternatives``, ``rows``, ``passes``), ``confidence_model:
    "openai_slots"``, ``prompt_version``, ``prompt_sha256``, ``elapsed_ms``.
    One ``max_tokens=1`` request per scalar field; one Y/N request per
    multi option. ``calibration`` (JSON path or dict) selects multi options
    by calibrated log-odds > 0, exactly like the native engine. Raises
    ChatCompletionsError on non-2xx responses.
    """
    from jevmlx.engine import _load_calibration

    calib = _load_calibration(calibration)
    t0 = time.perf_counter()
    parsed_json: dict[str, Any] = {}
    field_telemetry: dict[str, Any] = {}
    n_requests = 0
    for name, field in schema.fields.items():
        if field.field_type == "multi":
            parsed, telemetry, n = _decide_multi_field(
                base_url,
                model,
                api_key,
                schema,
                context,
                name,
                field,
                timeout,
                calib,
            )
            n_requests += n
        else:
            parsed, telemetry = _decide_scalar_field(
                base_url, model, api_key, schema, context, name, field, timeout
            )
            n_requests += 1
        parsed_json[name] = parsed
        field_telemetry[name] = telemetry

    elapsed_ms = (time.perf_counter() - t0) * 1000
    user_content = _user_content(schema, context)
    return {
        "elapsed_ms": round(elapsed_ms, 2),
        "prefill_ms": None,
        "suffix_eval_ms": None,
        "total_tokens_generated": 0,
        "sequential_forward_passes": n_requests,
        "schema_match": True,
        "confidence_model": "openai_slots",
        # The prompt TEXT is identical for every field's request (the decision
        # row travels as the assistant prefill); the sha covers the rendered
        # user turn. Native mode hashes the full prompt token ids — a
        # different tokenization path, so the two hashes are not comparable.
        "prompt_sha256": hashlib.sha256(user_content.encode("utf-8")).hexdigest(),
        "prompt_version": OPENAI_SLOTS_PROMPT_VERSION,
        "probability_status": (
            "renormalised top_logprobs mass at T=0; degraded (top-k only, "
            "floor for missing aliases), uncalibrated as decision confidence"
        ),
        "parsed_json": parsed_json,
        "field_telemetry": field_telemetry,
        "num_fields": len(schema),
    }
