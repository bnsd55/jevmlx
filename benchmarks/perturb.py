"""Deterministic label-preserving perturbations of an eval cases JSONL.

Robustness fixture for the eval harness: each perturbation paraphrases a
case's context in a way that must NOT change any label — whitespace
normalisation, order shuffling of independent blocks, a neutral preamble
line, and number reformatting. Running the same track over the original and
perturbed cases measures how much the model's predictions depend on surface
form (see ``jevmlx.evalmetrics.perturbation_flip_rate``).

Every variant becomes a new case appended after the originals:

- ``id`` = ``"<orig id>#p<k>"`` (k = 1-based variant index for that case)
- ``group_id`` = the original case id
- ``meta.perturbation`` = the perturbation kind (``ws``/``preamble``/
  ``numfmt``/``shuffle``)

Everything is deterministic: no wall clock, and shuffles draw from a
per-case ``random.Random`` seeded with ``(--seed, case id)``. Variants whose
transformation is a no-op (already-normalised text, nothing to reformat, a
single block there is no point shuffling) are skipped.

Usage:
    python -m benchmarks.perturb --in cases.jsonl --out perturbed.jsonl \
        --variants 3 --seed 0
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

# Neutral preamble prepended by the "preamble" kind (must not change labels).
PREAMBLE = "Record follows."

# Thousands separators are removed ("1,850" -> "1850") and trailing ".00"
# dropped ("1,850.00" -> "1850"). Other decimals are left alone ("3.5").
_THOUSANDS_RE = re.compile(r"(?<=\d),(?=\d{3}(?:\D|$))")
_POINT_ZERO_ZERO_RE = re.compile(r"\.00(?=\D|$)")


def _norm_whitespace(context: str) -> str:
    """Collapse space/tab runs, strip line edges, squeeze blank-line runs."""
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in context.splitlines()]
    squeezed: list[str] = []
    for line in lines:
        if line or not squeezed or squeezed[-1]:
            squeezed.append(line)
    return "\n".join(squeezed).strip()


def _format_numbers(context: str) -> str:
    """Rewrite ``1,850.00`` -> ``1850`` and ``1,850`` -> ``1850``."""
    context = _THOUSANDS_RE.sub("", context)
    return _POINT_ZERO_ZERO_RE.sub("", context)


def _blocks(context: str) -> list[str]:
    """Split into blank-line-separated blocks (the shuffle unit)."""
    return [block for block in re.split(r"\n\s*\n", context.strip()) if block.strip()]


def _shuffle(context: str, rng: random.Random) -> str:
    """Shuffle the order of independent blocks, keeping blocks intact."""
    blocks = _blocks(context)
    if len(blocks) < 2:
        return context
    return "\n\n".join(rng.sample(blocks, k=len(blocks)))


# kind -> (transform, needs_rng). Order is the variant-emission order.
# Context transforms take/return a context string; schema transforms take
# the whole case dict and return a new case (they change the SCHEMA, not
# the context — optrev reverses option order, criterion prefixes the
# field description).
TRANSFORMS: dict[str, tuple[Callable[..., str], bool]] = {
    "ws": (_norm_whitespace, False),
    "preamble": (lambda context: f"{PREAMBLE}\n\n{context}", False),
    "numfmt": (_format_numbers, False),
    "shuffle": (_shuffle, True),
}

# Schema perturbation kinds (W6-B2): label-preserving schema changes. These
# transform the CASE (schema), not the context. optrev reverses every enum
# field's option order (labels are strings — the option description — so a
# reversed order keeps the label valid). criterion prefixes each field's
# description with an evidence-grounding instruction.
_CRITERION_PREFIX = "Using only the supplied evidence, decide the following criterion: "


def _optrev(case: dict) -> dict:
    """Reverse the option order of every enum field in the schema.

    Label-preserving: labels are stored as the option DESCRIPTION (a
    string), not an index — reversing ``choices`` does not change which
    description is the gold answer.
    """
    import copy

    new = copy.deepcopy(case)
    schema = new.get("schema") or {}
    changed = False
    for field in schema.values():
        if field.get("type") == "enum" and len(field.get("choices", [])) >= 2:
            field["choices"] = list(reversed(field["choices"]))
            changed = True
    return new if changed else case


def _criterion(case: dict) -> dict:
    """Prefix each field description with an evidence-grounding instruction.

    Label-preserving: the instruction changes the phrasing, not the
    decision space (choices and labels are untouched).
    """
    import copy

    new = copy.deepcopy(case)
    schema = new.get("schema") or {}
    changed = False
    for field in schema.values():
        desc = field.get("description", "")
        if not desc.startswith(_CRITERION_PREFIX):
            field["description"] = _CRITERION_PREFIX + desc
            changed = True
    return new if changed else case


# Schema transforms: (transform, needs_rng=False). They take the whole case.
SCHEMA_TRANSFORMS: dict[str, tuple[Callable[[dict], dict], bool]] = {
    "optrev": (_optrev, False),
    "criterion": (_criterion, False),
}


def _candidate_kinds(variants: int) -> list[str]:
    """Kinds to try, in order, for ``--variants N``.

    The three deterministic context kinds first, then the two schema kinds
    (optrev, criterion), then as many shuffle attempts as the requested
    variant count (single-block cases skip shuffles, so a few spare attempts
    cost nothing).
    """
    base = ["ws", "preamble", "numfmt", "optrev", "criterion"]
    return base + ["shuffle"] * max(0, variants - len(base) + 1)


def perturb_case(case: dict, variants: int, seed: int) -> list[dict]:
    """Return up to ``variants`` perturbed copies of one case (originals kept).

    Kinds are tried in the fixed ``TRANSFORMS`` order; a kind whose transform
    does not change the context is skipped. Variant k draws its shuffle from
    ``random.Random((seed, case id, k))`` so the same inputs always produce
    the same file.
    """
    context = case["context"]
    out: list[dict] = []
    k = 0
    for kind in _candidate_kinds(variants):
        if len(out) >= variants:
            break
        if kind in SCHEMA_TRANSFORMS:
            # Schema perturbation (optrev/criterion): transforms the case
            # dict (schema), not the context. A no-op (returns the original)
            # is skipped.
            transform, _needs_rng = SCHEMA_TRANSFORMS[kind]
            candidate_case = transform(case)
            if candidate_case is case:
                continue
            k += 1
            variant = dict(candidate_case)
            variant["id"] = f"{case.get('id')}#p{k}"
            variant["group_id"] = case.get("id")
            meta = dict(case.get("meta") or {})
            meta["perturbation"] = kind
            variant["meta"] = meta
            out.append(variant)
            continue
        transform, needs_rng = TRANSFORMS[kind]
        if needs_rng:
            rng = random.Random(f"{seed}:{case.get('id')}:{k}")
            candidate = transform(context, rng)
        else:
            candidate = transform(context)
        if candidate == context:
            continue  # a no-op "perturbation" would only duplicate the case
        k += 1
        variant = dict(case)
        variant["id"] = f"{case.get('id')}#p{k}"
        variant["group_id"] = case.get("id")
        meta = dict(case.get("meta") or {})
        meta["perturbation"] = kind
        variant["meta"] = meta
        variant["context"] = candidate
        out.append(variant)
    return out


def perturb_cases(cases: list[dict], variants: int, seed: int) -> list[dict]:
    """Originals (unchanged, first) followed by every perturbed variant.

    Originals are kept so one file feeds both sides of the flip-rate
    comparison: a run over this file yields original and variant predictions
    sharing a ``group_id``.
    """
    out: list[dict] = [dict(case) for case in cases]
    for case in cases:
        out.extend(perturb_case(case, variants, seed))
    return out


def build_parser() -> argparse.ArgumentParser:
    """The perturb CLI parser (exposed for parse-only tests; see
    tests/test_m5_e2e.py TestPlannedArgvParses)."""
    parser = argparse.ArgumentParser(
        prog="python -m benchmarks.perturb",
        description="Generate deterministic label-preserving perturbations of eval cases.",
    )
    parser.add_argument("--in", dest="input", required=True, help="input cases JSONL")
    parser.add_argument("--out", required=True, help="output JSONL (originals + variants)")
    parser.add_argument("--variants", type=int, default=3, help="max variants per case (default 3)")
    parser.add_argument("--seed", type=int, default=0, help="seed for shuffles (default 0)")
    parser.add_argument(
        "--lock",
        default=None,
        help="lock path (default: <out stem>.dataset.lock.json next to --out)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    with open(args.input, encoding="utf-8") as f:
        cases = [json.loads(line) for line in f if line.strip() and not line.startswith("#")]

    records = perturb_cases(cases, args.variants, args.seed)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # B6: write the dataset lock so build_datasets finds the registered
    # <name>.dataset.lock.json (without this, the bench passed a lock path
    # that was never written and run.json's dataset_lock_sha256 raised
    # OSError at eval time). The lock records the sha256 of the cases file
    # just written — same shape as the synthetic/typesafe locks.
    lock_path = (
        Path(args.lock) if args.lock else out_path.parent / f"{out_path.stem}.dataset.lock.json"
    )
    lock = {
        "sources": [
            {
                "kind": "perturbation",
                "input": str(args.input),
                "variants": args.variants,
                "seed": args.seed,
            }
        ],
        "parser_version": "perturb-v1",
        "counts": {"originals": len(cases), "total": len(records)},
        "cases_sha256": hashlib.sha256(out_path.read_bytes()).hexdigest(),
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    lock_path.write_text(json.dumps(lock, indent=1) + "\n", encoding="utf-8")

    n_variants = len(records) - len(cases)
    print(f"cases: {len(cases)}")
    print(f"variants: {n_variants}")
    print(f"wrote {args.out}")
    print(f"wrote {lock_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
