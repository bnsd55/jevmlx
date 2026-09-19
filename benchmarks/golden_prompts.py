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

# The representative request every profile renders (one schema x one context;
# covers enum with gloss, boolean, and the nonce-fenced context).
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
}

# The profiles: engine-resolved from a representative model id. qwen2.5 =
# default (system supported, no template kwargs); qwen3 = enable_thinking
# off; gemma = system merged into the user turn. Tokenizer revisions come
# from the HF snapshot the vector was written against ("" for the fake).
PROFILES: dict[str, dict] = {
    "qwen2.5": {"model_id": "mlx-community/Qwen2.5-0.5B-Instruct-4bit", "real": True},
    "qwen3": {"model_id": "mlx-community/Qwen3-4B-Instruct-2507-4bit", "real": False},
    "gemma": {"model_id": "mlx-community/gemma-2-2b-it-4bit", "real": True},
}

FAKE_REV = "fake"


def _token_ids_sha256(ids: list[int]) -> str:
    return hashlib.sha256(json.dumps(list(ids)).encode("utf-8")).hexdigest()


class _KwargsTokenizer:
    """Fake tokenizer that accepts (and records) profile template kwargs.

    Same character encoding as tests.conftest.FakeTokenizer; records the
    kwargs it was handed so the committed vector shows what the profile
    passed to apply_chat_template.
    """

    name_or_path = "fake-engine"
    pad_token_id = 0

    def __init__(self):
        self.last_kwargs: dict = {}

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [ord(c) % 60 for c in text]

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=True, **kwargs):
        self.last_kwargs = dict(kwargs)
        return self.encode("\n".join(m["content"] for m in messages))

    def __len__(self) -> int:
        return 64


class _NoSystemTokenizer(_KwargsTokenizer):
    """Fake tokenizer whose template rejects a system role (Gemma-style)."""

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=True, **kwargs):
        if any(m["role"] == "system" for m in messages):
            from jinja2.exceptions import TemplateError

            raise TemplateError("system role not supported")
        return super().apply_chat_template(messages, add_generation_prompt, tokenize, **kwargs)


def _fake_tokenizer_for(profile: str) -> _KwargsTokenizer:
    return _NoSystemTokenizer() if profile == "gemma" else _KwargsTokenizer()


def _real_tokenizer(model_id: str):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    revision = _revision_of(model_id)
    if revision is None:
        raise SystemExit(
            f"{model_id}: no HF snapshot in the local cache — run the model "
            "once (or `huggingface-cli download`) before writing real vectors"
        )
    return tok, revision


def _revision_of(model_id: str) -> str | None:
    """The HF snapshot sha from the local cache (mirrors engine._revision_of)."""
    try:
        from huggingface_hub import constants

        cache_dir = Path(constants.HF_HUB_CACHE)
    except Exception:  # noqa: BLE001 — provenance is best-effort
        return None
    repo_dir = cache_dir / f"models--{model_id.replace('/', '--')}"
    main_ref = repo_dir / "refs" / "main"
    snapshot_dir = repo_dir / "snapshots"
    if main_ref.exists():
        return main_ref.read_text(encoding="utf-8").strip()
    if snapshot_dir.is_dir():
        snapshots = [p for p in snapshot_dir.iterdir() if p.is_dir()]
        if len(snapshots) == 1:
            return snapshots[0].name
    return None


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
        "rendered_text": tok.decode(ids)
        if hasattr(tok, "decode")
        else "".join(chr(35 + (i % 60)) for i in ids),
        "token_ids_sha256": _token_ids_sha256(ids),
        "prompt_sha256": _prompt_sha256(ids),
        "plan_hash": schema.plan_hash(tok, scoring),
    }


def vector_path(profile: str, tokenizer_revision: str, case: str) -> Path:
    return GOLDEN_DIR / f"{profile}__{tokenizer_revision}__{case}.json"


def write_vectors() -> list[Path]:
    """(Re)write every committed vector; return the paths written."""
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    written = []
    for profile, spec in sorted(PROFILES.items()):
        tok, revision = (
            _real_tokenizer(spec["model_id"])
            if spec["real"]
            else (_fake_tokenizer_for(profile), FAKE_REV)
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
    """
    problems: list[str] = []
    expected_names = set()
    for profile, spec in sorted(PROFILES.items()):
        try:
            tok, revision = (
                _real_tokenizer(spec["model_id"])
                if spec["real"]
                else (_fake_tokenizer_for(profile), FAKE_REV)
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
