"""Shared test fixtures, fakes and result factories.

PARITY_ATOL is the tolerance for real-model batch-vs-chunked log_score
parity: Metal batched matmuls tile differently at different batch shapes,
and the legal-mass full-vocab logsumexp (W2-D) adds a reduction that
perturbs the lazy evaluation graph.

Measured (W3-C, Qwen2.5-0.5B-4bit on M5, fintech_fraud + support_triage,
max_rows=3 vs full): worst drift 0.0293 nats — IDENTICAL on main before and
after the W3-C branch, i.e. pre-existing batch-shape noise, not a W3-C
regression. Coder1's F3 proposed 1e-2; the measurement shows 1e-2 fails on
main itself, so the constant ships at 5e-2 pending a remeasure.

W3-E: the constant now LIVES in jevmlx.engine as INSTABILITY_BAND (the
near-tie rescore band uses the same measurement) and is re-exported here so
tests and engine share exactly one number.
The FakeModel path stays exact (deterministic zeros).

W5b-2: shared fakes live HERE, once. One tokenizer per tokenization family,
one model per biasing behaviour, one engine-result factory — the per-file
copies drifted from the engine contract (a missing 'prior_ms' broke three
tests). The factory's keys are pinned to the engine by
test_make_engine_result_keys_match_engine in this file.
"""

import math

import pytest

# Real-model log_score parity tolerance (nats) == engine.INSTABILITY_BAND;
# re-exported under the tests' name for the parity suites.
from jevmlx.engine import INSTABILITY_BAND

PARITY_ATOL = INSTABILITY_BAND

MODEL_ID = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"


# %97+1 tokenizer: ids = ord(c) % 97 + 1 (so ids start at 1), the shape
# shared by test_multi / test_w2e_calib / test_w2e_count / test_w2_setcons /
# test_w4b_parity.
class _Mod97Tokenizer:
    """Char tokenizer, ids = ord(c) % 97 + 1 (ids start at 1; '' -> [1])."""

    name_or_path = "fake-engine"
    pad_token_id = 0

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [ord(c) % 97 + 1 for c in text] or [1]

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=True):
        assert tokenize
        return self.encode("\n".join(m["content"] for m in messages))

    def __len__(self) -> int:
        return 128


# ord(c) % 60 tokenizer: the FakeTokenizer shape shared by test_engine_fake /
# test_prompt_v2 / test_w1c / test_w2d_telemetry / test_openai_slots.
class FakeTokenizer:
    """Character tokenizer with the working-set size readable for the guard."""

    name_or_path = "fake-engine"

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [ord(c) % 60 for c in text]

    pad_token_id = 0

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=True):
        assert tokenize
        return self.encode("\n".join(m["content"] for m in messages))

    def __len__(self) -> int:
        return 64


class FakeModel:
    """Minimal model: zeros logits, real KVCache objects sized by layers."""

    def __init__(self, vocab_size: int = 64, n_layers: int = 2):
        self.vocab_size = vocab_size
        self.n_layers = n_layers
        self.args = type("Args", (), {"vocab_size": vocab_size})()
        self.layers = [None] * n_layers  # make_prompt_cache counts these

    def parameters(self):
        # tree_flatten-able empty tree: zero-weight fake model.
        return {}

    def __call__(self, tokens, cache=None):
        import mlx.core as mx

        batch, seq_len = tokens.shape
        if cache is not None:
            for c in cache:
                # Drive the cache like a real layer: update_and_fetch keeps
                # BatchKVCache/KVCache offsets correct after the merge-based
                # broadcast (the old direct keys/values stomp only worked for
                # unbatched KVCache).
                c.update_and_fetch(
                    mx.zeros((batch, 2, seq_len, 8)), mx.zeros((batch, 2, seq_len, 8))
                )
        return mx.zeros((batch, seq_len, self.vocab_size))


def _mod97(token_id: int) -> int:
    """The %97+1 token id of a character code (counts/YN models read the
    same ids the %97+1 tokenizer writes)."""
    return token_id % 97 + 1


class YNLogitModel(FakeModel):
    """Constant Y/N logits: every option row sees the same pair.

    P(yes) = sigmoid(yes_logit - no_logit). Y (P(yes) = sigmoid(2) ≈ 0.88
    for every option) at the defaults yes_logit=1.0 / no_logit=-1.0.
    """

    def __init__(self, yes_logit: float = 1.0, no_logit: float = -1.0, vocab_size: int = 128):
        super().__init__(vocab_size=vocab_size)
        self.yes_logit = yes_logit
        self.no_logit = no_logit

    def __call__(self, tokens, cache=None):

        out = super().__call__(tokens, cache)
        return (
            out.at[:, :, _mod97(ord("Y"))]
            .add(self.yes_logit)
            .at[:, :, _mod97(ord("N"))]
            .add(self.no_logit)
        )


class CountCodeModel(FakeModel):
    """Bias the count row and the option Y/N rows independently.

    Logits are keyed by token id, so biases are set on the character ids of
    the count codes ('0'..'4') and of Y/N. `count_bias` maps a count-code
    character to a logit bump; `yes_logit`/`no_logit` set the option rows.
    """

    def __init__(
        self,
        count_bias: dict[str, float],
        yes_logit: float,
        no_logit: float,
        vocab_size: int = 128,
    ):
        super().__init__(vocab_size=vocab_size)
        self.count_bias = count_bias
        self.yes_logit = yes_logit
        self.no_logit = no_logit

    def __call__(self, tokens, cache=None):

        out = super().__call__(tokens, cache)
        for ch, bump in self.count_bias.items():
            # The count codes are read at their divergence tokens: '4' is a
            # two-char code whose first token is '4' — bias that id.
            out = out.at[:, :, _mod97(ord(ch[0]))].add(bump)
        return (
            out.at[:, :, _mod97(ord("Y"))]
            .add(self.yes_logit)
            .at[:, :, _mod97(ord("N"))]
            .add(self.no_logit)
        )


# ------------------------------------------------- engine result factories --
# The FULL result dict `run_parallel_generation` returns (ARCHITECTURE.md
# 'Engine result dict'), with neutral defaults. These three builders are the
# only sanctioned way a test fakes an engine result; the contract test below
# pins the key set to the real engine so they cannot rot.


def _base_field_telemetry() -> dict:
    """A scalar enum entry with every key the scalar branch of
    run_parallel_generation's field loop emits."""
    return {
        "value": "A",
        "type": "enum",
        "probability": 0.6,
        "cardinality": 2,
        "log_scores": {"A": math.log(0.6), "B": math.log(0.4)},
        "top_choices": [
            {"choice": "A", "probability": 0.6},
            {"choice": "B", "probability": 0.4},
        ],
        "rows": 2,
        "tie": False,
        "rescored": False,
        "legal_mass": 1.0,
        "legal_mass_logs": {"A": 0.0, "B": 0.0},
    }


def make_field_telemetry(
    value: object = "A",
    type_: str = "enum",
    *,
    choices: list[str] | None = None,
    per_option: dict[str, float] | None = None,
    count_choice: str = "2",
    count_margin: float = 0.9,
    reconciled_by: str = "count",
    **overrides: object,
) -> dict:
    """One field_telemetry entry factory: the three shapes the engine emits.

    - ``type_='enum'`` (or 'boolean'): scalar shape — log_scores/top_choices/
      tie/rescored/legal_mass keyed by the choice strings.
    - ``type_='multi'``: multi shape — per_option/option_logit_pairs/
      calibrated/count_choice/count_margin/reconciled_by/margin; no
      field-level probability. ``per_option`` defaults to every option at
      0.9 (all-yes); ``value`` defaults to the top half by P(yes).
    - ``type_='count'``: the scalar-shaped '<field>#count' entry —
      log_scores over the '0'..'4' codes, margin_nats, no tie/rescored/
      legal_mass keys.
    ``choices`` sets the choice list (scalar keys / multi options);
    ``overrides`` patches keys on the finished entry.
    """
    if type_ == "multi":
        opts = list(choices) if choices is not None else ["a", "b"]
        n = len(opts)
        pairs = dict(per_option) if per_option is not None else {o: 0.9 for o in opts}
        ranked = sorted(pairs, key=pairs.get, reverse=True)
        entry: dict = {
            "value": list(value) if value != "A" else ranked[: max(1, n // 2)],
            "type": "multi",
            "probability": None,
            "margin": 0.4,
            "cardinality": n,
            "per_option": pairs,
            "option_logit_pairs": {o: [1.0, -1.0] for o in opts},
            "alternatives": tuple((o, pairs[o]) for o in ranked),
            "top_choices": [{"choice": o, "probability": pairs[o]} for o in ranked],
            "rows": n,
            "calibrated": None,
            "count_choice": count_choice,
            "count_margin": count_margin,
            "reconciled_by": reconciled_by,
            "legal_mass": 1.0,
            "legal_mass_logs": {o: 0.0 for o in opts},
            "rescored": False,
        }
    elif type_ == "count":
        codes = [str(i) for i in range(5)]
        probs = [0.1, 0.2, 0.4, 0.2, 0.1]
        entry = {
            "value": count_choice,
            "type": "enum",
            "probability": 0.3,
            "cardinality": len(codes),
            "log_scores": {c: math.log(p) for c, p in zip(codes, probs, strict=True)},
            "top_choices": [
                {"choice": c, "probability": p}
                for c, p in sorted(zip(codes, probs, strict=True), key=lambda cp: -cp[1])
            ],
            "rows": 5,
            "margin_nats": count_margin,
        }
    else:
        opts = list(choices) if choices is not None else ["A", "B"]
        p_winner = 0.6 if len(opts) > 1 else 1.0
        p_rest = (1.0 - p_winner) / max(1, len(opts) - 1)
        probs = [p_winner if i == 0 else p_rest for i in range(len(opts))]
        scores = [math.log(p) for p in probs]
        entry = {
            "value": value,
            "type": type_,
            "probability": p_winner,
            "cardinality": len(opts),
            "log_scores": dict(zip(opts, scores, strict=True)),
            "top_choices": [
                {"choice": c, "probability": p}
                for c, p in sorted(zip(opts, probs, strict=True), key=lambda cp: -cp[1])
            ],
            "rows": 2,
            "tie": False,
            "rescored": False,
            "legal_mass": 1.0,
            "legal_mass_logs": {c: 0.0 for c in opts},
        }
    entry.update(overrides)
    return entry


def make_engine_result(
    *,
    fields: dict[str, dict] | None = None,
    parsed: dict[str, dict] | None = None,
    **overrides: object,
) -> dict:
    """The FULL current result dict shape from ARCHITECTURE.md 'Engine result
    dict' (the keys run_parallel_generation actually returns), with neutral
    defaults.

    ``fields`` fills parsed_json + field_telemetry from the same telemetry
    entries (value rides both) — per-field parsed overrides go through
    ``parsed``. Case-level keys can be set or overridden via kwargs
    (``prior_ms=...``, ``temperature_status``-style extras land verbatim).
    """
    field_telemetry = dict(fields or {})
    parsed_json = dict(parsed or {})
    for fname, ft in field_telemetry.items():
        if fname in parsed_json:
            continue
        if "#count" in fname:
            continue  # count rows surface only in field_telemetry
        parsed_json[fname] = {"value": ft.get("value"), "prob": ft.get("probability")}
    return {
        "elapsed_ms": 5.0,
        "prior_ms": 0.0,
        "prefill_ms": 2.0,
        "plan_compile_ms": 0.1,
        "cache_broadcast_ms": 0.2,
        "suffix_eval_ms": 2.5,
        "lm_head_gather_ms": 0.3,
        "total_ms": 5.0,
        "padded_token_positions": 6,
        "total_tokens_generated": 0,
        "peak_active_bytes": 1024,
        "sequential_forward_passes": 1,
        "rescored_fields": [],
        "schema_match": True,
        "confidence_model": "slots",
        "prompt_sha256": "abc",
        "prompt_version": "jevmlx-parallel-v8",
        "probability_status": (
            "constrained-path probability at T=1; uncalibrated as decision confidence"
        ),
        "prior_correction": False,
        "constraints_applied": False,
        "reconciled_fields": [],
        "rerun_fields": [],
        "rerun_rows": 0,
        "second_pass_ms": 0.0,
        "parsed_json": parsed_json,
        "field_telemetry": field_telemetry,
        "num_fields": len(parsed_json),
        **overrides,
    }


@pytest.fixture(scope="module")
def engine():
    """The real 0.5B model, loaded once per module. Shared by every slow
    test that needs a live engine (test_engine, test_w4b_parity)."""
    from jevmlx.engine import load_engine

    return load_engine(MODEL_ID)


# --------------------------------------------------------- contract tests --


def test_make_engine_result_keys_match_engine():
    """W5b-2 contract: the factory's top-level keys EQUAL the keys
    run_parallel_generation actually returns on the fake model — the
    factory cannot rot when the engine's result shape changes."""
    from jevmlx.engine import run_parallel_generation
    from jevmlx.schema import StructuredSchema

    schema = StructuredSchema(
        {
            "flag": {"type": "boolean", "description": "d"},
            "action": {"type": "enum", "description": "d", "choices": ["A", "B"]},
            "tags": {"type": "multi", "description": "d", "choices": ["x", "y"]},
        }
    )
    real = run_parallel_generation(FakeModel(), FakeTokenizer(), "ctx", schema)
    fake = make_engine_result(
        fields={
            "flag": _base_field_telemetry(),
            "action": _base_field_telemetry(),
            "tags": make_field_telemetry(type_="multi", choices=["x", "y"]),
            "tags#count": make_field_telemetry(type_="count", count_choice="2"),
        }
    )
    assert set(real) == set(fake), (
        "make_engine_result drifted from the engine result contract: "
        f"engine-only={set(real) - set(fake)}, factory-only={set(fake) - set(real)}"
    )


def test_make_field_telemetry_shapes_match_engine():
    """The factory's scalar / multi / count entry shapes carry the key sets
    the engine's field loop emits for a boolean+enum+multi schema on the
    fake model (count row included) — entry-level rot is caught here too."""
    from jevmlx.engine import run_parallel_generation
    from jevmlx.schema import StructuredSchema

    schema = StructuredSchema(
        {
            "flag": {"type": "boolean", "description": "d"},
            "action": {"type": "enum", "description": "d", "choices": ["A", "B"]},
            "tags": {"type": "multi", "description": "d", "choices": ["x", "y"]},
        }
    )
    real = run_parallel_generation(FakeModel(), FakeTokenizer(), "ctx", schema)["field_telemetry"]

    scalar_keys = set(real["flag"]) | set(real["action"])
    multi_keys = set(real["tags"])
    count_keys = set(real["tags#count"])

    scalar_fake = make_field_telemetry(type_="boolean", choices=["true", "false"])
    # Optional engine keys (prior correction, set constraints) may be absent
    # from the neutral fake; every REQUIRED scalar key must exist.
    required_scalar = scalar_keys - {
        "prior_option_pairs",
        "prior_corrected",
        "set_constraints",
        "set_selection",
    }
    assert required_scalar <= set(scalar_fake), "scalar telemetry shape drifted"
    multi_fake = make_field_telemetry(type_="multi", choices=["x", "y"])
    required_multi = multi_keys - {"prior_option_pairs", "prior_corrected"}
    assert required_multi <= set(multi_fake), "multi telemetry shape drifted"
    count_fake = make_field_telemetry(type_="count")
    assert count_keys == set(count_fake), "count telemetry shape drifted"
    """Every per-file alias is the SAME conftest object — the per-file copies
    cannot come back quietly."""
    import tests.test_engine_fake as ef_mod
    import tests.test_multi as multi_mod
    import tests.test_prompt_v2 as prompt_mod
    import tests.test_w2_setcons as setcons_mod
    import tests.test_w2e_calib as calib_mod
    import tests.test_w2e_count as count_mod
    import tests.test_w3d_dag as dag_mod
    import tests.test_w3f_batch as batch_mod
    import tests.test_w4b_parity as parity_mod

    assert count_mod._CountTokenizer is _Mod97Tokenizer
    assert setcons_mod._CountTokenizer is _Mod97Tokenizer
    assert parity_mod._CountTokenizer is _Mod97Tokenizer
    assert multi_mod.FakeTokenizer is _Mod97Tokenizer
    assert calib_mod.FakeTokenizer is _Mod97Tokenizer
    assert prompt_mod.FakeTokenizer is FakeTokenizer
    assert prompt_mod.FakeModel is FakeModel
    assert ef_mod.FakeModel is FakeModel
    assert ef_mod.FakeTokenizer is FakeTokenizer
    assert dag_mod.FakeModel is FakeModel
    assert dag_mod.FakeTokenizer is FakeTokenizer
    assert batch_mod.FakeModel is FakeModel
    assert batch_mod.FakeTokenizer is FakeTokenizer
    assert calib_mod._BiasedMultiModel is YNLogitModel
    assert setcons_mod._YNModel is YNLogitModel
    assert count_mod._BiasedModel is CountCodeModel
    assert parity_mod._StableModel is YNLogitModel
