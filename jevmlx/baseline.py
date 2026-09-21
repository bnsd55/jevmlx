"""Naive-JSON baseline for the eval harness (OpenAI-compatible chat API).

The eval compares jevmlx's constrained path against the same model (or a
bigger API model) writing the whole JSON object itself: same schema, same
context, same information. Two modes: ``'text'`` (default, truly naive — no
response_format) and ``'json'`` (response_format json_object). Parsing is
strict and never raises on bad model output — that IS the measurement.
"""

from __future__ import annotations

import json
import time

from jevmlx.http import chat_completions_raw, extract_content
from jevmlx.schema import StructuredSchema

__all__ = [
    "baseline_decide",
    "build_baseline_messages",
    "call_chat_completions",
    "parse_baseline_output",
]


def _field_line(name: str, field) -> str:
    """One 'key: type and allowed values' line for the instruction block."""
    if field.field_type == "boolean":
        return f'- "{name}" (boolean: true or false) — {field.description}'
    if field.field_type == "multi":
        choices = ", ".join(f'"{c}"' for c in field.choices)
        return (
            f'- "{name}" (an array, possibly empty, of allowed strings: {choices})'
            f" — {field.description}"
        )
    choices = ", ".join(f'"{c}"' for c in field.choices)
    return f'- "{name}" (one of exactly: {choices}) — {field.description}'


def build_baseline_messages(
    schema: StructuredSchema, context: str, *, mode: str = "text"
) -> list[dict]:
    """Build the single-user-message prompt for the naive-JSON baseline.

    Deterministic text, no system role. ``mode='text'`` (default) is truly
    naive: a plain JSON-only instruction with no response_format on the wire.
    ``mode='json'`` says the same thing but pairs with
    ``response_format={'type': 'json_object'}`` in the API call. The
    instruction adds no constraints the schema does not have (multi fields
    may be empty arrays).
    """
    field_lines = "\n".join(_field_line(name, field) for name, field in schema.fields.items())
    content = (
        "You will be given a context and a list of fields to decide.\n"
        "Output ONLY a JSON object — no markdown, no explanation — with exactly these keys:\n"
        f"{field_lines}\n\n"
        "Context:\n"
        f"{context}"
    )
    return [{"role": "user", "content": content}]


def call_chat_completions(
    base_url: str,
    model: str,
    messages: list[dict],
    *,
    api_key: str | None,
    timeout: float = 120.0,
    temperature: float = 0.0,
    mode: str = "text",
) -> str:
    """POST to ``{base_url}/chat/completions`` and return the assistant content.

    Thin wrapper over the shared client in :mod:`jevmlx.http`.
    ``mode='text'`` (default) sends no response_format — truly naive;
    ``mode='json'`` adds ``response_format={'type': 'json_object'}``
    best-effort. Raises ChatCompletionsError on any non-2xx response with the status
    and the first 200 chars of the body.
    """
    extra = {"response_format": {"type": "json_object"}} if mode == "json" else None
    if mode not in ("text", "json"):
        raise ValueError(f"mode must be 'text' or 'json', got {mode!r}")
    choice = chat_completions_raw(
        base_url,
        model,
        messages,
        api_key=api_key,
        timeout=timeout,
        temperature=temperature,
        extra_payload=extra,
    )
    return extract_content(choice)


def _validate_field(name: str, field, raw) -> tuple[bool, object, str | None]:
    """Strictly validate one parsed value against its field definition.

    Returns (ok, value, problem): value is the parsed value when usable for
    salvage, None only for wrong-typed values; problem is the error string or
    None. Multi arrays may be empty (the schema adds no non-empty constraint)
    but duplicates and out-of-choices items are errors. Numeric scalars (int/float)
    for string-digit enum choices are str-coerced before the membership check,
    so int 2 for choices ('0','1','2','3') is accepted as '2' (a baseline must
    not lose a semantically-correct case to a type-only mismatch); wrong values
    like str(2.5)=='2.5' or a non-choice string still error.
    """
    if field.field_type == "boolean":
        if isinstance(raw, bool):
            return True, raw, None
        return False, None, f"wrong type for {name}: expected boolean, got {type(raw).__name__}"
    if field.field_type == "multi":
        if not isinstance(raw, list):
            return False, None, f"wrong type for {name}: expected array, got {type(raw).__name__}"
        invalid = [item for item in raw if not isinstance(item, str) or item not in field.choices]
        if invalid:
            return False, None, f"invalid items for {name}: {invalid} (allowed: {field.choices})"
        if len(set(raw)) != len(raw):
            dupes = sorted({item for item in raw if raw.count(item) > 1})
            return False, None, f"duplicate items for {name}: {dupes}"
        return True, raw, None
    if isinstance(raw, str) and raw in field.choices:
        return True, raw, None
    # A naive baseline must not lose a semantically-correct case to a type-only
    # mismatch (int 2 where the schema wants string '2'). Coerce non-str
    # scalars (int/float) to str before the membership check; this rescues
    # str(2)=='2' but not str(2.5)=='2.5' or a wrong string like 'MAYBE'.
    # Booleans are already handled by the earlier field_type == 'boolean' branch.
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        coerced = str(raw)
        if coerced in field.choices:
            return True, coerced, None
    return False, None, f"invalid value for {name}: {raw!r} (allowed: {field.choices})"


def parse_baseline_output(text: str, schema: StructuredSchema) -> tuple[dict, dict, list[str]]:
    """Strictly parse a baseline model's JSON output against the schema.

    Extraction uses ``json.JSONDecoder().raw_decode`` at the first '{' — the
    object must be followed by nothing but whitespace, else a 'trailing text'
    error. Then per-field strict validation: missing keys, extra keys,
    duplicate items in a multi array, wrong types, and values not in choices
    are all errors. Never raises on model output — that is the measurement.

    Returns ``(strict_values, salvage_values, errors)``: ``strict_values`` maps
    every schema field to its value or None when anything was wrong;
    ``salvage_values`` keeps every field whose own value was parseable, even
    when other fields or surrounding text failed.
    """
    fields_all_none = {name: None for name in schema.get_field_names()}
    decoder = json.JSONDecoder()
    stripped = text.lstrip()
    brace = stripped.find("{")
    errors: list[str] = []
    parsed: dict | None = None
    if brace == -1:
        errors.append("no JSON object found in output")
    else:
        try:
            obj, end = decoder.raw_decode(stripped[brace:])
            if stripped[brace + end :].strip():
                errors.append("trailing text after JSON object")
            parsed = obj
        except json.JSONDecodeError as e:
            errors.append(f"invalid JSON: {e}")
    if parsed is None:
        return fields_all_none, fields_all_none.copy(), errors
    if not isinstance(parsed, dict):
        errors.append("JSON value is not an object")
        return fields_all_none, fields_all_none.copy(), errors

    strict_values: dict = {}
    salvage_values: dict = {}
    for name, field in schema.fields.items():
        if name not in parsed:
            strict_values[name] = None
            salvage_values[name] = None
            errors.append(f"missing key: {name}")
            continue
        raw = parsed[name]
        ok, value, problem = _validate_field(name, field, raw)
        strict_values[name] = value if ok else None
        salvage_values[name] = value
        if problem is not None:
            errors.append(problem)
    for name in sorted(set(parsed) - set(schema.fields)):
        errors.append(f"extra key: {name}")
    if errors:
        # strict = zero errors: any problem voids the strict object entirely
        strict_values = fields_all_none.copy()
    return strict_values, salvage_values, errors


def baseline_decide(
    base_url: str,
    model: str,
    api_key: str | None,
    schema: StructuredSchema,
    context: str,
    *,
    mode: str = "text",
) -> dict:
    """One baseline decision: prompt, call, strict-parse, and time it.

    Returns ``{"values", "salvage_values", "errors", "raw", "latency_ms",
    "strict_valid", "schema_valid"}`` (``schema_valid`` is kept as an alias of
    ``strict_valid``). Never raises on bad model output — only on
    transport-level failures (ChatCompletionsError).
    """
    messages = build_baseline_messages(schema, context, mode=mode)
    t0 = time.perf_counter()
    raw = call_chat_completions(base_url, model, messages, api_key=api_key, mode=mode)
    latency_ms = (time.perf_counter() - t0) * 1000
    values, salvage_values, errors = parse_baseline_output(raw, schema)
    return {
        "values": values,
        "salvage_values": salvage_values,
        "errors": errors,
        "raw": raw,
        "latency_ms": latency_ms,
        "strict_valid": not errors,
        "schema_valid": not errors,
    }
