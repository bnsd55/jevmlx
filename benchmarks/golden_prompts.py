"""Golden prompt vectors (W5c-5, GPT-review B3): generate + check.

The byte-exact prompt unit is
``prompt version x prompt profile x tokenizer revision x representative
request`` — there is NO single byte-exact prompt across chat templates.

This module WRITES the committed vectors (``--write``) and CHECKS the
committed bytes against the live renderer (``--check``). NOT circular: the
committed vector is the only source of "expected"; ``--check`` re-renders
and diffs, never regenerating both sides from the same code path in one
run (the test below compares committed bytes to the renderer the same way).

Vector shape (tests/golden/prompts/<profile>__<tokenizer-rev>__<case>.json):
{
  "prompt_version": "jevmlx-parallel-v9",
  "profile": "qwen2.5",                     # engine profile name
  "tokenizer_revision": "a5339a41...",      # HF snapshot sha (fake: "fake")
  "case": "risk_enum_bool",                 # representative request name
  "input": {"schema": {...}, "context": "...", "scoring": "slots",
             "system": "...", "template_kwargs": {...},
             "supports_system": true},
  "rendered_text": "<decoded prompt ids>",  # the exact user+system render
  "token_ids_sha256": "...",                # sha256 over the prompt token ids
  "prompt_sha256": "...",                   # the engine's provenance hash
  "plan_hash": "..."                        # the compiled plan's hash
}

Tokenizer load is cheap (no model): the real-tokenizer vector renders with
``transformers.AutoTokenizer`` only. Everything runs on CPU/any OS.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from jevmlx.engine import (
    PROMPT_V2_SYSTEM,
    PROMPT_VERSION,
    _chat_ids,
    _probe_system_role,
    _profile_for,
    _prompt_sha256,
    _user_content,
)
from jevmlx.schema import StructuredSchema

GOLDEN_DIR = Path(__file__).resolve().parent.parent / "tests" / "golden" / "prompts"
PROTOCOL_PATH = Path(__file__).resolve().parent.parent / "PROMPT_PROTOCOL.md"

# The representative requests every profile renders: the enum+boolean
# slots case, the same schema in labels mode (real choice strings, no
# aliases), and a multi-field case (count question + per-option Y/N menu).
CASES: dict[str, dict] = {
    "risk_enum_bool": {
        "schema": {
            "risk_tier": {
                "type": "enum",
                "description": "Credit risk tier",
                "choices": ["LOW", "MEDIUM", "HIGH"],
                "choice_descriptions": {
                    "LOW": "stable income",
                    "HIGH": "many missed payments",
                },
            },
            "flag": {"type": "boolean", "description": "manually flagged"},
        },
        "context": "The applicant pays late sometimes.",
        "scoring": "slots",
    },
    "risk_enum_bool_labels": {
        "schema": {
            "risk_tier": {
                "type": "enum",
                "description": "Credit risk tier",
                "choices": ["LOW", "MEDIUM", "HIGH"],
                "choice_descriptions": {
                    "LOW": "stable income",
                    "HIGH": "many missed payments",
                },
            },
            "flag": {"type": "boolean", "description": "manually flagged"},
        },
        "context": "The applicant pays late sometimes.",
        "scoring": "labels",
    },
    "tags_multi": {
        "schema": {
            "tags": {
                "type": "multi",
                "description": "observed tags",
                "choices": ["late_payment", "dispute"],
            },
        },
        "context": "The applicant disputes one charge.",
        "scoring": "slots",
    },
    "tags_multi_labels": {
        "schema": {
            "tags": {
                "type": "multi",
                "description": "observed tags",
                "choices": ["late_payment", "dispute"],
            },
        },
        "context": "The applicant disputes one charge.",
        "scoring": "labels",
    },
}

# The profiles: engine-resolved from a representative model id. qwen2.5 =
# default (system supported, no template kwargs); qwen3 = enable_thinking
# off; gemma = system merged into the user turn. Tokenizer revisions come
# from the HF snapshot the vector was written against ("" for the fake).
PROFILES: dict[str, dict] = {
    "qwen2.5": {
        "model_id": "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
        "real": True,
        "revision": "a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3",
    },
    "qwen3": {
        "model_id": "mlx-community/Qwen3-4B-Instruct-2507-4bit",
        "real": False,
        "revision": "fake",
    },
    "gemma": {
        "model_id": "mlx-community/gemma-2-2b-it-4bit",
        "real": True,
        "revision": "2c715097ff9c081a6ac1e5cd239e2ac756b5bd99",
    },
}

FAKE_REV = "fake"


def _token_ids_sha256(ids: list[int]) -> str:
    return hashlib.sha256(json.dumps(list(ids)).encode("utf-8")).hexdigest()


def _conftest_fake(profile: str):
    """The fake tokenizer from tests/conftest (extended for **kwargs and the
    Gemma-style no-system rejection) — one fake, no duplicate here.

    Note: benchmarks importing tests/conftest via sys.path is unusual
    (conftest is not a package); accepted here because the fake tokenizers
    must be the EXACT objects the tests use — a second copy would defeat
    the one-fake rule.
    """
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))
    import conftest

    return conftest.FakeTokenizerNoSystem() if profile == "gemma" else conftest.FakeTokenizer()


def _real_tokenizer(model_id: str, revision: str):
    """The pinned-revision tokenizer (F1): the sha comes from PROFILES, so an
    upstream repo push can never silently change a committed vector.

    A cold-offline cache miss is a CLEAR error (F3), not a traceback: the
    vector cannot be verified without the pinned tokenizer files.
    """
    from transformers import AutoTokenizer

    try:
        tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
    except Exception as exc:  # noqa: BLE001 — one clear message for any cause
        raise SystemExit(
            f"{model_id}@{revision}: tokenizer not available ({exc.__class__.__name__}). "
            "The golden-prompt check needs the PINNED tokenizer files in the local "
            "HF cache — CI populates them via actions/cache; locally run: "
            f"huggingface-cli download {model_id} --revision {revision} "
            "(tokenizer files only; no model weights are ever loaded)"
        ) from exc
    return tok, revision


def build_vector(profile: str, case: str, tok, tokenizer_revision: str) -> dict:
    """Render one (profile, case) through the REAL renderer and return the vector.

    This is the single write path; --check re-runs it and diffs against the
    committed bytes.
    """
    spec = CASES[case]
    schema = StructuredSchema(spec["schema"])
    scoring = spec["scoring"]
    engine_profile = _profile_for(PROFILES[profile]["model_id"])
    if profile == "gemma":
        # The engine probes at load; a fake tokenizer here needs the probe
        # too (the real Gemma tokenizer is probed the same way).
        engine_profile = _probe_system_role(tok, engine_profile)
    user_content = _user_content(spec["context"], schema, tok, scoring)
    ids = _chat_ids(tok, user_content, PROMPT_V2_SYSTEM, engine_profile)
    if hasattr(ids, "keys"):  # transformers BatchEncoding
        ids = list(ids["input_ids"])
    ids = [int(i) for i in ids]
    # rendered_text must be HUMAN-READABLE in every committed vector. For a
    # tokenizer with decode() (the real ones) that is the decode; for the
    # fake (a reversible character cipher with no decode) it is the exact
    # message text _chat_ids rendered from — the same bytes the ids encode.
    messages = (
        [{"role": "system", "content": PROMPT_V2_SYSTEM}, {"role": "user", "content": user_content}]
        if engine_profile.supports_system
        else [{"role": "user", "content": f"{PROMPT_V2_SYSTEM}\n\n{user_content}"}]
    )
    if hasattr(tok, "decode"):
        rendered = tok.decode(ids)
    else:
        rendered = "\n".join(m["content"] for m in messages)
    return {
        "prompt_version": PROMPT_VERSION,
        "profile": profile,
        "tokenizer_revision": tokenizer_revision,
        "case": case,
        "input": {
            "schema": spec["schema"],
            "context": spec["context"],
            "scoring": scoring,
            "system": PROMPT_V2_SYSTEM,
            "template_kwargs": engine_profile.template_kwargs,
            "supports_system": engine_profile.supports_system,
        },
        "rendered_text": rendered,
        "token_ids_sha256": _token_ids_sha256(ids),
        "prompt_sha256": _prompt_sha256(ids),
        "plan_hash": schema.plan_hash(tok, scoring),
    }


def vector_path(profile: str, tokenizer_revision: str, case: str) -> Path:
    return GOLDEN_DIR / f"{profile}__{tokenizer_revision}__{case}.json"


# ---- PROMPT_PROTOCOL.md generated sections (F2) --------------------------
# The doc's EXAMPLE blocks are rendered by this module between HTML comments
# (the doc's prose lives in the file; the bytes that claim to come from the
# renderer DO come from the renderer). --write refreshes them; --check
# diffs them like a vector.
PROTOCOL_MARKERS = {
    "system_block": ("<!-- generated:system_block -->", "<!-- /generated:system_block -->"),
    "user_content": ("<!-- generated:user_content -->", "<!-- /generated:user_content -->"),
    "rendered_vectors": (
        "<!-- generated:rendered_vectors -->",
        "<!-- /generated:rendered_vectors -->",
    ),
}


def _protocol_sections() -> dict[str, str]:
    """The generated doc sections, from the REAL renderer."""
    from jevmlx.engine import _context_nonce

    tok = _conftest_fake("qwen2.5")
    case = CASES["risk_enum_bool"]
    schema = StructuredSchema(case["schema"])
    user_content = _user_content(case["context"], schema, tok, "slots")

    system_block = PROMPT_V2_SYSTEM  # the exact engine-owned constant

    # One rendered vector per profile, fenced code block, filename-labeled.
    # Guarded reads: a half-written or foreign file must not crash the doc
    # render (check_vectors reports it separately as a stale vector).
    rendered = []
    for path in sorted(GOLDEN_DIR.glob("*.json")):
        try:
            vector = json.loads(path.read_text(encoding="utf-8"))
            label = f"{vector['profile']} / {vector['case']} ({vector['input']['scoring']})"
            text = vector["rendered_text"]
        except (OSError, json.JSONDecodeError, KeyError):
            continue
        # Fence safety (re-review a): rendered_text may not end with a
        # newline (the qwen3 fake vectors end mid-line), so a bare ``` fence
        # would land ON the last text line and swallow the rest of the doc
        # into the code block. Normalize to exactly one trailing newline
        # before closing the fence.
        rendered.append(f"**{label}** — `{path.name}`:\n\n```text\n{text.rstrip(chr(10))}\n```")

    return {
        "system_block": f"```text\n{system_block}\n```",
        "user_content": f"```text\n{user_content}\n```",
        "rendered_vectors": "\n\n".join(rendered),
        "nonce_example": _context_nonce(case["context"]),
    }


def write_protocol() -> None:
    """(Re)write PROMPT_PROTOCOL.md's generated sections in place."""
    sections = _protocol_sections()
    doc = PROTOCOL_PATH.read_text(encoding="utf-8")
    for name, (begin, end) in PROTOCOL_MARKERS.items():
        i = doc.find(begin)
        j = doc.find(end)
        if i == -1 or j == -1 or j < i:
            raise SystemExit(f"PROMPT_PROTOCOL.md: markers for {name} missing — add {begin!r}")
        doc = doc[: i + len(begin)] + "\n" + sections[name] + "\n" + doc[j:]
    PROTOCOL_PATH.write_text(doc, encoding="utf-8")


def check_protocol() -> list[str]:
    """Diff the committed doc's generated sections against a fresh render."""
    sections = _protocol_sections()
    problems: list[str] = []
    try:
        doc = PROTOCOL_PATH.read_text(encoding="utf-8")
    except OSError as exc:
        return [f"PROMPT_PROTOCOL.md unreadable: {exc}"]
    for name, (begin, end) in PROTOCOL_MARKERS.items():
        i = doc.find(begin)
        j = doc.find(end)
        if i == -1 or j == -1 or j < i:
            problems.append(f"PROMPT_PROTOCOL.md: markers for {name} missing")
            continue
        committed = doc[i + len(begin) : j].strip()
        fresh = sections[name]
        if committed != fresh:
            problems.append(f"PROMPT_PROTOCOL.md[{name}]: stale — regenerate with --write")
    # Fence sanity (re-review a): every generated code block must OPEN and
    # CLOSE — an odd fence count means a closing fence landed on a text
    # line and the rest of the doc renders inside a code block.
    fence_count = sum(
        1 for line in doc.splitlines() if line.strip() == "```" or line.strip() == "```text"
    )
    if fence_count % 2 != 0:
        problems.append(
            f"PROMPT_PROTOCOL.md: {fence_count} code fences (odd) — a closing "
            "fence landed on a text line; regenerate with --write"
        )
    return problems


def write_vectors() -> list[Path]:
    """(Re)write every committed vector and the doc's generated sections."""
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    write_protocol()
    written = []
    for profile, spec in sorted(PROFILES.items()):
        tok, revision = (
            _real_tokenizer(spec["model_id"], spec["revision"])
            if spec["real"]
            else (_conftest_fake(profile), FAKE_REV)
        )
        for case in sorted(CASES):
            path = vector_path(profile, revision, case)
            vector = build_vector(profile, case, tok, revision)
            path.write_text(json.dumps(vector, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            written.append(path)
    return written


def check_vectors() -> list[str]:
    """Re-render every vector and diff against the committed bytes.

    Returns a list of stale/missing-vector problems (empty = fresh).
    Also diffs PROMPT_PROTOCOL.md's generated sections (F2).
    """
    problems: list[str] = check_protocol()
    expected_names = set()
    for profile, spec in sorted(PROFILES.items()):
        try:
            tok, revision = (
                _real_tokenizer(spec["model_id"], spec["revision"])
                if spec["real"]
                else (_conftest_fake(profile), FAKE_REV)
            )
        except SystemExit as exc:
            problems.append(f"{profile}: {exc}")
            continue
        for case in sorted(CASES):
            name = f"{profile}__{revision}__{case}"
            expected_names.add(name)
            path = vector_path(profile, revision, case)
            if not path.exists():
                problems.append(f"{path.relative_to(GOLDEN_DIR.parent.parent)}: missing vector")
                continue
            committed = json.loads(path.read_text(encoding="utf-8"))
            fresh = build_vector(profile, case, tok, revision)
            if committed != fresh:
                diffs = sorted(
                    key
                    for key in set(committed) | set(fresh)
                    if committed.get(key) != fresh.get(key)
                )
                problems.append(f"{name}: stale — differing keys {diffs}")
    # Vectors from a revision that is no longer what the renderer resolves
    # are stale by definition: flag committed files with no current (profile,
    # revision, case) so --check fails instead of silently ignoring them.
    for path in sorted(GOLDEN_DIR.glob("*.json")):
        name = path.stem
        if name not in expected_names:
            problems.append(
                f"{path.name}: committed vector's (profile, tokenizer_revision) "
                "no longer matches the renderer — regenerate with --write"
            )
    return problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Golden prompt vectors: write or check")
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--write", action="store_true", help="(re)write the committed vectors")
    group.add_argument(
        "--check", action="store_true", help="fail when the committed vectors are stale"
    )
    args = ap.parse_args(argv)
    if args.write:
        for path in write_vectors():
            print(f"wrote {path}")
        return 0
    problems = check_vectors()
    if problems:
        for problem in problems:
            print(f"STALE: {problem}", file=sys.stderr)
        return 1
    print("golden prompts: all vectors fresh")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
