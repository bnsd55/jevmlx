"""Schema definitions and batch-plan compilation for parallel constrained decisions.

Supports booleans, categorical enums (cardinality up to 255), and multi fields
(subset of choices, 2-64 options, decided as one boolean decision per option).
"""

import hashlib
import json
import logging
import weakref
from typing import Any

_LOGGER = logging.getLogger(__name__)


def _alias_code(index: int) -> str:
    """Neutral choice alias for slot scoring: A..Z, then AA..ZZ (base 26)."""
    if index < 0:
        raise ValueError(f"alias index must be >= 0, got {index}")
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    if index < 26:
        return letters[index]
    code = ""
    while index >= 0:
        code = letters[index % 26] + code
        index = index // 26 - 1
    return code


def _common_token_prefix(sequences: list[list[int]]) -> list[int]:
    """Longest common token-ID prefix of every sequence (at least one required)."""
    if not sequences:
        return []
    shared: list[int] = []
    shortest = min(len(sequence) for sequence in sequences)
    for position in range(shortest):
        tokens_at = {sequence[position] for sequence in sequences}
        if len(tokens_at) != 1:
            break
        shared.append(sequences[0][position])
    return shared


class SchemaCompileError(ValueError):
    """The plan compilers cannot produce a scorable plan for one field.

    Raised for token-identical choices, strict token-prefix continuations,
    candidates with no shared token prefix, and duplicate-choice keys.
    Carries the offending field name so linters can attribute the failure.
    """

    def __init__(self, field: str, message: str):
        super().__init__(message)
        self.field = field


class FieldDefinition:
    def __init__(
        self,
        name: str,
        field_type: str,
        description: str,
        choices: list[str] | None = None,
        choice_descriptions: dict[str, str] | None = None,
    ):
        self.name = name
        self.field_type = field_type.lower()
        self.description = description
        # Optional per-choice glosses keyed by choice string. Validated only
        # when choices exist: every key must be a declared choice. (coder1's
        # prompt v2 renders them; the engine never reads them.)
        self.choice_descriptions: dict[str, str] = dict(choice_descriptions or {})

        if self.field_type == "boolean":
            self.choices = ["true", "false"]
        elif self.field_type == "multi":
            if not choices or len(choices) < 2:
                raise ValueError(f"Field '{name}' of type multi must have at least 2 choices.")
            if len(choices) > 64:
                raise ValueError(
                    f"Field '{name}' exceeds maximum cardinality of 64 choices "
                    f"for type multi (got {len(choices)})."
                )
            self.choices = choices
        elif self.field_type in ("enum", "choice", "selection"):
            if not choices or len(choices) == 0:
                raise ValueError(f"Field '{name}' of type enum must have choices defined.")
            if len(choices) > 255:
                raise ValueError(
                    f"Field '{name}' exceeds maximum cardinality of 255 choices "
                    f"(got {len(choices)})."
                )
            self.choices = choices
        else:
            raise ValueError(
                f"Unsupported field type '{field_type}'. "
                "Supported types: 'boolean', 'enum' and 'multi'."
            )
        if self.field_type in ("multi", "enum", "choice", "selection"):
            if self.choice_descriptions:
                unknown = sorted(set(self.choice_descriptions) - set(self.choices))
                if unknown:
                    raise ValueError(
                        f"Field '{name}': choice_descriptions keys not in choices: "
                        f"{', '.join(repr(u) for u in unknown)}"
                    )
            seen: set[str] = set()
            for choice in self.choices:
                if choice in seen:
                    raise ValueError(
                        f"Field '{name}': duplicate choice '{choice}'; the engine "
                        "cannot distinguish duplicate choices"
                    )
                seen.add(choice)

    @property
    def cardinality(self) -> int:
        return len(self.choices)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "type": self.field_type,
            "description": self.description,
            "choices": self.choices,
            "cardinality": self.cardinality,
            "choice_descriptions": self.choice_descriptions,
        }


class StructuredSchema:
    def __init__(self, schema_dict: dict[str, Any]):
        self.fields: dict[str, FieldDefinition] = {}
        for field_name, spec in schema_dict.items():
            if "." in field_name:
                # Field names become JSON row keys (json.dumps'd at candidate
                # build time); a dot inside a field name could collide with
                # another field's literal name (C3: row keys must be
                # injective).
                raise ValueError(
                    f"Field name '{field_name}' contains '.'; dot-free field "
                    "names keep multi row keys injective"
                )
            if "/" in field_name:
                # Multi option rows are keyed '<field>/<option>'; a slash
                # inside a field name would collide across fields (C3: row
                # keys must be injective).
                raise ValueError(
                    f"Field name '{field_name}' contains '/'; slash-free field "
                    "names keep multi option row keys '<field>/<option>' injective"
                )
            self.fields[field_name] = FieldDefinition(
                name=field_name,
                field_type=spec.get("type", "enum"),
                description=spec.get("description", ""),
                choices=spec.get("choices", None),
                choice_descriptions=spec.get("choice_descriptions", None),
            )
        # Compiled plans, keyed by tokenizer OBJECT IDENTITY (P2: a
        # WeakKeyDictionary keys by __eq__/__hash__, so two equal-but-distinct
        # tokenizers would wrongly share one plan). dict[id] =
        # (weakref.ref(tok), plan); weakref.finalize evicts the entry when the
        # tokenizer dies, so an id can never be reused by a live object while
        # its entry lingers. One schema object can be reused with several
        # models, and token IDs are tokenizer-specific.
        self._plans: dict[int, tuple[weakref.ref, dict[str, Any]]] = {}
        self._logged_non_weakref = False

    def get_field_names(self) -> list[str]:
        return list(self.fields.keys())

    def __getitem__(self, key: str) -> FieldDefinition:
        return self.fields[key]

    def __len__(self) -> int:
        return len(self.fields)

    def to_json_schema_prompt_str(self) -> str:
        """Returns a clean TypeScript/JSON schema representation for naive LLM prompting."""
        lines = ["{"]
        for name, field in self.fields.items():
            if field.field_type == "boolean":
                lines.append(f'  "{name}": boolean, // {field.description}')
            elif field.field_type == "multi":
                choices_str = " | ".join(f'"{c}"' for c in field.choices)
                lines.append(
                    f'  "{name}": [{choices_str}], // {field.description} (select all that apply)'
                )
            else:
                # Every choice, always: the parallel path sees all choices,
                # so a naive baseline that truncates large enums would not
                # be a fair comparison.
                choices_str = " | ".join(f'"{c}"' for c in field.choices)
                lines.append(f'  "{name}": {choices_str}, // {field.description}')
        lines.append("}")
        return "\n".join(lines)

    def _multi_field_header(self, field: FieldDefinition) -> str:
        """Schema-block text for one multi field: options as a described
        yes/no menu ('each option is answered yes or no')."""
        parts = []
        for choice in field.choices:
            gloss = field.choice_descriptions.get(choice)
            gloss_part = f" — {gloss}" if gloss else ""
            parts.append(f"{choice}{gloss_part}")
        return "; ".join(parts)

    def to_schema_str(self, mode: str = "slots") -> str:
        """Schema block for prompt v2, rendered per scoring mode.

        mode ``"slots"`` lists each field's choices as neutral aliases
        (``A) <choice>`` plus `` — <gloss>`` when a gloss exists); mode
        ``"labels"`` lists the real choice strings (``LOW | MEDIUM``). The
        aliases are what slot-trie scoring reads back on assembly; labels
        mode shows exactly the text the scorer reads.

        Multi fields render once as a described yes/no menu — the field
        header states that each option is answered yes or no; the per-option
        decision rows below (``"<field>/<option>"``) are answered with the
        aliases Y/N.
        """
        if mode not in ("slots", "labels"):
            raise ValueError(f"mode must be 'slots' or 'labels', got {mode!r}")
        lines = []
        for name, field in self.fields.items():
            desc = field.description.split("\n")[0].strip()
            if field.field_type == "multi":
                menu = self._multi_field_header(field)
                lines.append(
                    f'  "{name}": {menu}  // {desc} (select all that apply; '
                    "each option is answered yes or no)"
                )
                continue
            choices_list = (
                ["true", "false"] if field.field_type == "boolean" else list(field.choices)
            )
            if mode == "slots":
                parts = []
                for i, choice in enumerate(choices_list):
                    alias = _alias_code(i)
                    gloss = field.choice_descriptions.get(choice)
                    parts.append(f"{alias}) {choice} — {gloss}" if gloss else f"{alias}) {choice}")
            else:
                parts = []
                for choice in choices_list:
                    gloss = field.choice_descriptions.get(choice)
                    parts.append(f"{choice} — {gloss}" if gloss else choice)
            lines.append(f'  "{name}": {"  ".join(parts)}  // {desc}')
        return "\n".join(lines)

    def to_alias_schema_str(self) -> str:
        """Schema block in slots mode (neutral aliases)."""
        return self.to_schema_str("slots")

    def to_labels_schema_str(self) -> str:
        """Schema block in labels mode (real choice strings)."""
        return self.to_schema_str("labels")

    @staticmethod
    def alias_for_index(index: int) -> str:
        """The neutral alias for the choice at ``index`` (A, B, ..., AA, AB...)."""
        return _alias_code(index)

    def plan_hash(self, tokenizer, mode: str) -> str:
        """sha256 of the compiled plan for ``mode`` (stable within a process).

        The plan captures the schema block, choice order, and token
        segmentation the scoring pass depends on — everything the neutral
        prior must match. Compiled on demand; the result equals the hash of
        ``json.dumps(plan, sort_keys=True)`` with token ids as ints.
        """
        if mode == "slots":
            plan = self.compile_slot_plan(tokenizer)
        elif mode == "labels":
            plan = self.compile_labels_plan(tokenizer)
        else:
            raise ValueError(f"mode must be 'slots' or 'labels', got {mode!r}")
        return hashlib.sha256(
            json.dumps(plan, sort_keys=True, default=list).encode("utf-8")
        ).hexdigest()

    def _cached_plan(self, tokenizer, mode: str) -> dict[str, Any] | None:
        """Return the live cached plan for (tokenizer, mode), or None.

        Identity check: the stored ref() must resolve to THIS tokenizer, not
        just an equal one (P2 — two equal-but-distinct tokenizers must not
        share a plan; token ids are tokenizer-specific). Callers compile and
        call _cache_plan when this returns None.
        """
        entry = self._plans.get((id(tokenizer), mode))
        if entry is not None and entry[0]() is tokenizer:
            return entry[1]
        return None

    def _cache_plan(self, tokenizer, plan: dict[str, Any], mode: str) -> None:
        """Store a plan keyed by tokenizer identity, evicted on tokenizer death.

        Non-weak-referenceable tokenizers are not cached at all (N3): an
        id()-keyed entry without a liveness check could be returned for a
        different object after id reuse. The finalizer is registered only
        when the key is new — re-storing an existing key must not stack
        finalizer objects on the tokenizer.
        """
        try:
            ref = weakref.ref(tokenizer)
        except TypeError:
            if not self._logged_non_weakref:
                _LOGGER.debug(
                    "tokenizer %s is not weak-referenceable; plan cache disabled "
                    "for it (compiling on every call)",
                    type(tokenizer).__name__,
                )
                self._logged_non_weakref = True
            return
        key = id(tokenizer)
        cache_key = (key, mode)
        if cache_key not in self._plans:
            weakref.finalize(tokenizer, self._plans.pop, cache_key, None)
        self._plans[cache_key] = (ref, plan)

    def compile_slot_plan(self, tokenizer) -> dict[str, dict[str, Any]]:
        """Slot-trie plan (the default scoring mode): the decision row stays
        JSON — ``'{\n  "<field>": '`` — and the scored candidates are the
        QUOTED neutral aliases ``'"A"'``, `'"B"'``, ... (base-26 codes beyond
        26 choices), mapped back to the real choice strings on assembly.

        Multi fields share the same per-option yes/no rows in both modes
        (see compile_labels_plan): the row text is the natural question and
        the scored candidates are the quoted "Y"/"N" aliases. Booleans get
        aliases too (A -> true, B -> false).

        Returns the same shape as the labels plan
        (``{"lead_in_ids": ..., "fields": {fname: plan}}``) with per-field
        ``alias_map`` ({alias: real value}) added; the engine scores
        ``shared_ids``/``remainders`` through the token trie unchanged and
        maps winners back via ``alias_map``.
        """
        cached = self._cached_plan(tokenizer, "slots")
        if cached is not None:
            return cached
        fields_plan: dict[str, dict[str, Any]] = {}

        def slot_candidate_text(name: str, alias: str) -> str:
            """The assistant tail for one alias row: '{\n' + row + ',\n' with
            the quoted alias as the value."""
            return "{\n" + f'  {json.dumps(name)}: "{alias}"' + ",\n"

        # Lead-in candidates: scalar fields' shared prefixes. Computed after
        # the per-field plans exist (same two-pass shape as labels mode).
        for fname, fdef in self.fields.items():
            if fdef.field_type == "multi":
                continue
            if fdef.field_type == "boolean":
                values = ["true", "false"]
            else:
                values = list(fdef.choices)
            aliases = [_alias_code(i) for i in range(len(values))]
            alias_map = dict(zip(aliases, values, strict=True))

            candidates = [
                tokenizer.encode(slot_candidate_text(fname, alias), add_special_tokens=False)
                for alias in aliases
            ]
            shared = _common_token_prefix(candidates)
            remainders = [full[len(shared) :] for full in candidates]
            for i, remainder in enumerate(remainders):
                for j, other in enumerate(remainders):
                    if i == j or other[: len(remainder)] != remainder:
                        continue
                    if other == remainder:
                        raise SchemaCompileError(
                            fname,
                            f"field '{fname}': aliases '{aliases[i]}' and '{aliases[j]}' "
                            "are token-identical; the engine cannot distinguish them",
                        )
                    raise SchemaCompileError(
                        fname,
                        f"field '{fname}': alias '{aliases[i]}' is a strict token-prefix "
                        f"of '{aliases[j]}' in token space; the engine would never "
                        "distinguish them",
                    )
            if not shared and len({r[0] for r in remainders}) > 1:
                raise SchemaCompileError(
                    fname,
                    f"field '{fname}': alias candidates share no token prefix "
                    f"(tokenizer {type(tokenizer).__name__}); cannot place the "
                    "decision row",
                )
            fields_plan[fname] = {
                "shared_ids": shared,
                "remainders": remainders,
                "alias_map": alias_map,
                "aliases": aliases,
                "choices": values,
            }

        if any(f.field_type == "multi" for f in self.fields.values()):
            labels_plan = self.compile_labels_plan(tokenizer)
            for fname, fdef in self.fields.items():
                if fdef.field_type == "multi":
                    fields_plan[fname] = labels_plan["fields"][fname]
        row_prefixes = [p["shared_ids"] for p in fields_plan.values() if "shared_ids" in p]
        lead_in = _common_token_prefix(row_prefixes) if row_prefixes else []
        if lead_in:
            for p in fields_plan.values():
                if "shared_ids" in p:
                    p["shared_ids"] = p["shared_ids"][len(lead_in) :]
        result = {"lead_in_ids": list(lead_in), "fields": fields_plan}
        self._cache_plan(tokenizer, result, mode="slots")
        return result

    def compile_labels_plan(self, tokenizer) -> dict[str, dict[str, Any]]:
        """Labels scoring plan (choice-text trie): candidates are the real
        choice strings; the decision row is the full JSON row text. The
        engine maps winners straight to the choice strings (no alias hop).
        """
        return self._compile_labels(tokenizer)

    def _compile_labels(self, tokenizer) -> dict[str, dict[str, Any]]:  # noqa: D401
        """Pre-index everything the engine needs for the batched suffix pass.

        Token-aligned at both boundaries: every choice is encoded as ONE
        tokenization of its complete assistant-tail candidate — ``{\n`` + the
        JSON row text + ``,\n`` (the row text being ``  "name": "choice"``
        or the bare literal, built with json.dumps so quotes and backslashes
        survive). Tokenizing a prefix and its remainder separately would be
        wrong: BPE merges are not compositional, so the concatenated ids would
        not be the tokenization of the full candidate. The prefill prompt ends
        exactly at the chat template's generation marker; ``{\n`` belongs to
        the candidate tail, not the prompt.

        Per enum/boolean field the plan carries ``shared_ids`` (the common
        token-ID prefix across the candidates of ALL fields — typically the
        ``{\n  "`` lead-in) and per-choice ``remainders``.
        Per multi field it carries one ``suffix_ids_list`` entry per option —
        that option's shared token prefix (everything before its yes/no
        divergence, i.e. the row the engine runs) — and ``remainders`` with
        the option's two Y/N continuations. The returned mapping is
        ``{"lead_in_ids": [...], "fields": {field_name: plan}}`` — metadata
        sits beside the field plans, never inside them (D1: a field could
        legally be named "_lead_in_ids"). Plans are cached per tokenizer
        identity (name_or_path + vocab size).
        """
        cached = self._cached_plan(tokenizer, "labels")
        if cached is not None:
            return cached
        plan: dict[str, dict[str, Any]] = {}

        def candidate_text(name: str, value_text: str) -> str:
            """The complete assistant tail for one row: '{\n' + row + ',\n'.

            The row key is always json.dumps(name) (C3: no raw interpolation;
            field names are dot- and slash-free, so '<field>.<option>' scalar
            keys and '<field>/<option>' option keys are injective across
            (field, option) pairs and field names).
            """
            return "{\n" + f"  {json.dumps(name)}: {value_text}" + ",\n"

        for fname, fdef in self.fields.items():
            if fdef.field_type == "multi":
                # One yes/no row per option. The row is the natural question
                # ('"<field>/<option>": '), and the scored candidates are the
                # QUOTED aliases "Y"/"N" — the same slot machinery enums use
                # (V4: the model never sees a synthetic 'field.option' JSON
                # key; the schema block describes each option and states that
                # every option is answered yes or no).
                # The plan is built PER OPTION from that option's own candidate
                # pair: shared = everything up to the Y/N divergence (the
                # option's full row lead-in plus any common token start of
                # 'Y'/'N'), remainders = the two continuations. A single
                # cross-option prefix would put branch nodes at the option-name
                # position and read the Y/N logits at the wrong spot.
                suffix_ids_list = []
                remainders_per_option = []
                for option in fdef.choices:
                    pair = [
                        tokenizer.encode(
                            candidate_text(f"{fname}/{option}", f'"{alias}"'),
                            add_special_tokens=False,
                        )
                        for alias in ("Y", "N")
                    ]
                    option_shared = _common_token_prefix(pair)
                    option_remainders = [full[len(option_shared) :] for full in pair]
                    if not option_shared:
                        # Zero-length row: the Y/N decision would sit
                        # directly at the generation boundary (C2, same rule
                        # as the scalar guard).
                        raise SchemaCompileError(
                            fname,
                            f"field '{fname}': option '{option}' Y/N "
                            f"candidates share no token prefix (tokenizer "
                            f"{type(tokenizer).__name__}); cannot place the "
                            "decision row",
                        )
                    if option_remainders[0] == option_remainders[1]:
                        raise SchemaCompileError(
                            fname,
                            f"field '{fname}': option '{option}' tokenizes to "
                            "identical Y/N candidates; the engine cannot "
                            "distinguish them",
                        )
                    # Same rule as enums (R5): a strict-prefix continuation can
                    # never be distinguished by branch scoring.
                    if len(option_remainders[0]) < len(option_remainders[1]):
                        shorter, longer = option_remainders
                        short_name, long_name = "Y", "N"
                    else:
                        shorter, longer = option_remainders[1], option_remainders[0]
                        short_name, long_name = "N", "Y"
                    if longer[: len(shorter)] == shorter:
                        raise SchemaCompileError(
                            fname,
                            f"field '{fname}': option '{option}' has a strict "
                            f"token-prefix continuation ({short_name} is a prefix of "
                            f"{long_name} in token space); the engine would never "
                            "distinguish them",
                        )
                    suffix_ids_list.append(option_shared)
                    remainders_per_option.append(option_remainders)
                plan[fname] = {
                    "options": list(fdef.choices),
                    # Stored WITHOUT the schema-wide lead-in; the engine
                    # prepends it to every row (one rule for all row types).
                    "suffix_ids_list": suffix_ids_list,
                    "remainders": remainders_per_option,
                }
                continue

            if fdef.field_type == "boolean":
                value_texts = ["true", "false"]
            else:
                value_texts = [json.dumps(choice) for choice in fdef.choices]
            candidates = [
                tokenizer.encode(candidate_text(fname, value_text), add_special_tokens=False)
                for value_text in value_texts
            ]

            shared = _common_token_prefix(candidates)
            remainders = [full[len(shared) :] for full in candidates]
            for i, remainder in enumerate(remainders):
                for j, other in enumerate(remainders):
                    if i == j or other[: len(remainder)] != remainder:
                        continue
                    if other == remainder:
                        raise SchemaCompileError(
                            fname,
                            f"field '{fname}': choices '{fdef.choices[i]}' and "
                            f"'{fdef.choices[j]}' are token-identical; the engine "
                            "cannot distinguish them",
                        )
                    raise SchemaCompileError(
                        fname,
                        f"field '{fname}': choice '{fdef.choices[i]}' is a strict "
                        f"token-prefix of '{fdef.choices[j]}' in token space; the "
                        "engine would never distinguish them",
                    )
            if not shared and len({r[0] for r in remainders}) > 1:
                # After lead-in removal this field would have an empty row: a
                # branch directly at the generation boundary cannot be scored
                # by a broadcast row (B2).
                raise SchemaCompileError(
                    fname,
                    f"field '{fname}': candidates share no token prefix "
                    f"(tokenizer {type(tokenizer).__name__}); cannot place the "
                    "decision row",
                )
            plan[fname] = {
                "shared_ids": shared,
                "remainders": remainders,
            }

        # Schema-wide lead-in (typically '{\n  "') shared by every field's
        # candidates: lifted out of shared_ids so the engine can keep it in
        # the prefill broadcast cache. Remainders stay relative to the full
        # per-field shared prefix; rows are lead_in + shared_ids + path.
        # Lead-in candidates: every row prefix — scalar fields' shared_ids
        # AND multi option prefixes (B1: with only a multi field, the lead-in
        # must still be the common prefix of the option rows, never their
        # longer per-option text).
        field_shared_prefixes = [p["shared_ids"] for p in plan.values() if "shared_ids" in p] + [
            ids for p in plan.values() if "suffix_ids_list" in p for ids in p["suffix_ids_list"]
        ]
        if not field_shared_prefixes:
            wrapped: dict[str, Any] = {"lead_in_ids": [], "fields": plan}
            self._cache_plan(tokenizer, wrapped, mode="labels")
            return wrapped
        lead_in = _common_token_prefix(field_shared_prefixes)
        # An empty schema-wide lead-in is legal (e.g. char-level tokenizers
        # where '{\n' fuses with the field name): the engine then runs one row
        # per field with no broadcast prefix — each row still carries that
        # field's full shared_ids.
        # Apply the same strip to multi option prefixes so the engine can
        # prepend lead_in uniformly to every row (R1: one rule for all rows).
        for p in plan.values():
            if "suffix_ids_list" in p and lead_in:
                p["suffix_ids_list"] = [
                    ids[len(lead_in) :] if ids[: len(lead_in)] == lead_in else ids
                    for ids in p["suffix_ids_list"]
                ]
        for p in plan.values():
            if "shared_ids" in p:
                p["shared_ids"] = p["shared_ids"][len(lead_in) :]
        # Metadata lives beside the field plans, never mixed into them (D1:
        # a field could legally be named "_lead_in_ids").
        result = {"lead_in_ids": list(lead_in), "fields": plan}
        self._cache_plan(tokenizer, result, mode="labels")
        return result
