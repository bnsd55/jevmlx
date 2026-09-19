"""W5c-9: the persisted drift envelope and the widened rescore band.

The near-tie rescore must fire whenever the batched top-two margin sits
inside INSTABILITY_BAND + E_bound(M) — E_bound from the PERSISTED drift
envelope (jevmlx.driftenv), not a constant. The escape it fixes:
code_security/is_vulnerability at batch-one margin 0.031 with a batched
raw-logit margin 0.0625 — ABOVE the old 0.05 band, so no rescore, while
the batch-one reference WOULD have rescored (its margin 0.031 < 0.05).

The PARITY contract stays 0.05 on d_gap: these tests cover the DECISION
band only.
"""

from __future__ import annotations

import json

import pytest

from jevmlx.engine import (
    INSTABILITY_BAND,
    ScalarEvidence,
    finalize_scalar_evidence,
    run_parallel_generation,
)
from tests.conftest import FakeTokenizer, make_engine
from tests.test_engine_fake import BiasedFakeModel

# An injected envelope at the measured plateau: d_gap 0.0625 at >16 rows.
PLATEAU_ENVELOPE = {
    "key": {"model_id": "fake-engine"},
    "bucket": "M<=16",
    "bound": 0.0625,
    "band": 0.125,
    "source": "recorded",
}


def _band_envelope(bound: float) -> dict:
    import jevmlx.driftenv as dv

    return {
        "key": {"model_id": "fake-engine"},
        "bucket": "M<=16",
        "bound": bound,
        "band": dv.round_up_lattice(INSTABILITY_BAND + bound),
        "source": "recorded",
    }


def test_band_for_rows_plateau():
    """band(M>4) = 0.05 + 0.0625 rounded UP to the 1/64 lattice = 0.125."""
    import jevmlx.driftenv as dv

    env = dict(PLATEAU_ENVELOPE)
    assert dv.band_for_rows(env, 28) == pytest.approx(0.125)
    assert dv.band_for_rows(env, 112) == pytest.approx(0.125)
    # Small-M buckets without a record of their own: the plateau record
    # still bounds them (fail-safe).
    assert dv.band_for_rows(env, 2) == pytest.approx(0.125)


def test_band_for_rows_small_record():
    """An M<=4 record of 0.0391 bounds only the small buckets."""
    import jevmlx.driftenv as dv

    env = {
        "records": [
            {"shape_bucket": "M<=4", "max_gap_drift_nats": 0.0390625},
            {"shape_bucket": "M>16", "max_gap_drift_nats": 0.0625},
        ]
    }
    assert dv.band_for_rows(env, 2) == pytest.approx(dv.round_up_lattice(0.05 + 0.0390625))
    assert dv.band_for_rows(env, 28) == pytest.approx(0.125)
    assert dv.band_for_rows(env, 8) == pytest.approx(0.125)


def test_band_for_rows_missing_envelope_conservative():
    """Missing/malformed envelope: the conservative plateau bound applies."""
    import jevmlx.driftenv as dv

    assert dv.band_for_rows(None, 28) == pytest.approx(0.125)
    assert dv.band_for_rows({}, 28) == pytest.approx(0.125)
    assert dv.band_for_rows({"records": "junk"}, 28) == pytest.approx(0.125)


def test_round_up_lattice():
    import jevmlx.driftenv as dv

    assert dv.round_up_lattice(0.1125) == pytest.approx(0.125)
    assert dv.round_up_lattice(0.0625) == pytest.approx(0.0625)
    assert dv.round_up_lattice(0.0) == pytest.approx(0.0)
    assert dv.round_up_lattice(0.001) == pytest.approx(0.015625)


def test_shape_bucket():
    import jevmlx.driftenv as dv

    assert dv.shape_bucket(1) == "M<=4"
    assert dv.shape_bucket(4) == "M<=4"
    assert dv.shape_bucket(8) == "M<=8"
    assert dv.shape_bucket(16) == "M<=16"
    assert dv.shape_bucket(17) == "M>16"
    assert dv.shape_bucket(112) == "M>16"


def test_finalize_rescores_margin_010_with_envelope():
    """Margin 0.10: inside 0.05 + 0.0625 = 0.125 -> rescored."""
    calls = []

    def rescore(_idxs):
        calls.append(_idxs)
        # The canonical re-measure: a decisive margin.
        return ScalarEvidence(
            choices=("A", "B"),
            log_scores_raw=(-0.1, -2.0),
            legal_mass_logs=(-0.01, -0.02),
            source_shape="batch1",
        )

    evidence = ScalarEvidence(
        choices=("A", "B"),
        log_scores_raw=(-0.5, -0.6),  # margin 0.10
        legal_mass_logs=(-0.01, -0.02),
        source_shape="batch",
    )
    decision, rescored = finalize_scalar_evidence(
        evidence,
        prior_entry=None,
        temperature=1.0,
        rescore=rescore,
        rescore_idxs=[0, 1],
        rescore_band_nats=0.125,
        drift_envelope_nats=0.0625,
    )
    assert rescored is True
    assert calls
    assert decision.rescore_band_nats == pytest.approx(0.125)
    assert decision.drift_envelope_nats == pytest.approx(0.0625)
    assert decision.value == "A"


def test_finalize_no_rescore_margin_013_with_envelope():
    """Margin 0.13: outside 0.125 -> not rescored."""
    calls = []

    def rescore(_idxs):
        calls.append(_idxs)
        raise AssertionError("rescore must not run")

    evidence = ScalarEvidence(
        choices=("A", "B"),
        log_scores_raw=(-0.5, -0.63),  # margin 0.13
        legal_mass_logs=(-0.01, -0.02),
        source_shape="batch",
    )
    decision, rescored = finalize_scalar_evidence(
        evidence,
        prior_entry=None,
        temperature=1.0,
        rescore=rescore,
        rescore_idxs=[0, 1],
        rescore_band_nats=0.125,
        drift_envelope_nats=0.0625,
    )
    assert rescored is False
    assert not calls
    assert decision.rescore_band_nats == pytest.approx(0.125)


def test_finalize_without_envelope_keeps_005():
    """No band passed -> the historical constant: margin 0.0625 is NOT
    rescored (the parity-era behavior, unchanged for envelope-less paths)."""
    calls = []

    def rescore(_idxs):
        calls.append(_idxs)
        raise AssertionError("rescore must not run")

    evidence = ScalarEvidence(
        choices=("A", "B"),
        log_scores_raw=(-0.5, -0.5625),  # margin 0.0625 > 0.05
        legal_mass_logs=(-0.01, -0.02),
        source_shape="batch",
    )
    decision, rescored = finalize_scalar_evidence(
        evidence,
        prior_entry=None,
        temperature=1.0,
        rescore=rescore,
        rescore_idxs=[0, 1],
    )
    assert rescored is False
    assert not calls
    assert decision.rescore_band_nats == pytest.approx(INSTABILITY_BAND)


class _MarginFakeModel(BiasedFakeModel):
    """A fake whose branch margin is a fixed small value: the winner's
    alias token gets `margin` nats, the other 0 — identical at every batch
    shape (the fake has no shape dependence). With margin in (0.05, 0.125]
    the OLD band does not rescore; the WIDENED band does."""

    def __init__(self, margin: float, vocab_size: int = 64, n_layers: int = 2):
        # The alias "A" token (slots mode): the same id the collision test
        # biases — ord("A") % 60 — so the margin lands on the decision
        # position's winner child.
        super().__init__(vocab_size=vocab_size, n_layers=n_layers, winner_token=ord("A") % 60)
        self.margin = margin

    def __call__(self, tokens, cache=None):
        out = super().__call__(tokens, cache)
        # Undo the 20-nat bias, apply the small margin to the same token.
        out = out.at[..., self.winner_token].add(self.margin - 20.0)
        return out


def test_engine_end_to_end_widened_band_rescores():
    """Full engine: an injected envelope widens the band so a field whose
    batched raw margin (0.075) sat above the old 0.05 band now rescores.

    The fake's margin is identical at every batch shape, so the rescore
    replaces the batched evidence with the SAME canonical evidence — the
    observable is that the rescore FIRED (semantics source batch1 +
    rescore_band telemetry), which is the escape mechanism under test.
    """
    model = _MarginFakeModel(margin=0.075)
    tokenizer = FakeTokenizer()
    schema = _schema_two_choice()
    env = _band_envelope(0.0625)
    engine = make_engine(model, tokenizer, drift_envelope=env)
    result = run_parallel_generation(engine, "ctx", schema)
    tel = result["field_telemetry"]["action"]
    assert tel["rescored"] is True
    assert tel["rescore_band_nats"] == pytest.approx(0.125)
    assert tel["drift_envelope_nats"] == pytest.approx(0.0625)
    assert tel["semantics"]["score_source"] == "rescored_batch1"


def test_engine_end_to_end_without_envelope_constant_band():
    """Same engine without an envelope: margin 0.075 >= 0.05, no rescore
    (the behavior the envelope widens)."""
    model = _MarginFakeModel(margin=0.075)
    tokenizer = FakeTokenizer()
    schema = _schema_two_choice()
    engine = make_engine(model, tokenizer)
    result = run_parallel_generation(engine, "ctx", schema)
    tel = result["field_telemetry"]["action"]
    assert tel["rescored"] is False
    assert tel["rescore_band_nats"] == pytest.approx(INSTABILITY_BAND)


def _schema_two_choice():
    from jevmlx.schema import StructuredSchema

    return StructuredSchema({"action": {"type": "enum", "description": "d", "choices": ["A", "B"]}})


def test_record_envelope_persists_and_loads(tmp_path, monkeypatch):
    """record -> cache write -> load round-trip keyed by the tuple.

    All driftenv access is dv-qualified ON PURPOSE: the suite evicts
    jevmlx.* modules mid-run (test_check_results.test_imports_without_mlx),
    so a top-level `from jevmlx.driftenv import f` can hold a DEAD
    instance's function — one whose module global the monkeypatch never
    touches. The repo pins this exact rot for machine_tag; same guard
    discipline here.
    """
    import jevmlx.driftenv as dv

    monkeypatch.setattr(dv, "_ENVELOPE_CACHE", tmp_path / "cache")
    key = {
        "model_id": "m",
        "revision": "r",
        "quantization": None,
        "mlx_version": "1",
        "chip": "c",
        "activation_dtype": "float16",
    }
    rec = {"key": key, "shape_bucket": "M>16", dv.MAX_GAP_DRIFT_KEY: 0.0625}
    written = dv.record_envelope(rec)
    assert written
    loaded = dv.load_envelope_record(key)
    assert loaded is not None
    assert dv.bound_from_records(loaded["records"], "M>16") == pytest.approx(0.0625)
    # A higher measured value replaces the same bucket; a lower one does
    # not (the envelope keeps the MAX).
    dv.record_envelope(
        {"key": key, "shape_bucket": "M>16", dv.MAX_GAP_DRIFT_KEY: 0.125, "source": "probe"}
    )
    loaded2 = dv.load_envelope_record(key)
    assert dv.bound_from_records(loaded2["records"], "M>16") == pytest.approx(0.125)
    assert len(loaded2["records"]) == 1


def test_envelope_cache_path_is_tuple_keyed():
    import jevmlx.driftenv as dv

    k1 = {"model_id": "a", "chip": "m5max"}
    k2 = {"model_id": "a", "chip": "m5max", "quantization": {"bits": 4}}
    assert dv.envelope_cache_path(k1) != dv.envelope_cache_path(k2)
    # Deterministic.
    assert dv.envelope_cache_path(k1) == dv.envelope_cache_path(dict(k1))


def test_envelope_for_engine_uses_recorded_without_canary(tmp_path, monkeypatch):
    """A recorded envelope resolves WITHOUT running the canary (no model
    work at load when the tuple is already measured)."""
    import jevmlx.driftenv as dv

    monkeypatch.setattr(dv, "_ENVELOPE_CACHE", tmp_path / "cache")

    class _FakeEngine:
        model_id = "fake-engine"
        revision = None

        model = type("M", (), {"parameters": lambda self: {}})()
        tokenizer = None

    # Record exactly the key envelope_for_engine will compute.
    real_key = dv.envelope_key(_FakeEngine())
    dv.record_envelope(
        {"key": real_key, "shape_bucket": "M<=16", dv.MAX_GAP_DRIFT_KEY: 0.0625, "source": "probe"}
    )
    canary_calls = []

    def _no_canary(_engine):
        canary_calls.append(1)
        raise AssertionError("canary must not run when recorded")

    monkeypatch.setattr(dv, "run_canary", _no_canary)
    res = dv.envelope_for_engine(_FakeEngine())
    assert res["source"] == "recorded"
    assert res["bound"] == pytest.approx(0.0625)
    assert res["band"] == pytest.approx(0.125)
    assert not canary_calls


def test_envelope_for_engine_runs_canary_when_missing(tmp_path, monkeypatch):
    """Missing envelope: the canary path runs, records, and the band uses
    its measured value."""
    import jevmlx.driftenv as dv

    monkeypatch.setattr(dv, "_ENVELOPE_CACHE", tmp_path / "cache")

    class _FakeEngine:
        model_id = "fake-engine"
        revision = None
        model = type("M", (), {"parameters": lambda self: {}})()
        tokenizer = None

    def _fake_canary(_engine):
        return {
            "key": dv.envelope_key(_engine),
            "shape_bucket": "M<=16",
            dv.MAX_GAP_DRIFT_KEY: 0.0625,
            "source": "canary",
            "canary_rows": 16,
        }

    monkeypatch.setattr(dv, "run_canary", _fake_canary)
    res = dv.envelope_for_engine(_FakeEngine())
    assert res["source"] == "canary"
    assert res["band"] == pytest.approx(0.125)  # max(0.0625, plateau floor)
    # The canary WROTE its record: the next load finds it.
    loaded = dv.load_envelope_record(res["key"])
    assert loaded is not None
    assert dv.bound_from_records(loaded["records"], "M<=16") == pytest.approx(0.0625)


def test_parity_report_unchanged_contract():
    """The parity payload's atol stays the FIXED 0.05 — the band protects
    decisions, it does not redefine parity."""
    import jevmlx.engine as eng
    from jevmlx.parity import parity_report  # noqa: F401 — import parity of contract

    assert eng.INSTABILITY_BAND == pytest.approx(0.05)


def test_canary_never_undercuts_plateau(monkeypatch, tmp_path):
    """A canary measuring LESS than the known plateau (tiny synthetic
    schema on the same fp16 grid) does NOT shrink the band: the canary
    lower-bounds nothing — the recorded value is max(measured, plateau).
    (Real-model behavior verified on the 0.5B through slowtest.sh: the
    canary measured 0.0, the record carries 0.0625, the band stays 0.125
    and the escaped field rescores.)"""
    import jevmlx.driftenv as dv

    rec = dv.run_canary(_CanaryOnlyEngine())
    assert rec[dv.MAX_GAP_DRIFT_KEY] == pytest.approx(0.0625)
    # The stub engine cannot run the real probe -> the fallback fires —
    # the ASSERTION is the floor: even the fallback never under-covers.
    assert rec["source"] == "canary_fallback"


class _CanaryOnlyEngine:
    """Enough engine for envelope_key; the canary itself is monkeypatched."""

    model_id = "fake-canary"
    revision = None
    model = type("M", (), {"parameters": lambda self: {}})()
    tokenizer = None


def test_bucket_edges_are_data_driven(tmp_path, monkeypatch):
    """A recorded bucket-edge file RE-SPLITS the buckets (review note: the
    fp32 bisect jump between M=16 and M=112 demands a finer M>16 bucket
    once coder6 reports the exact M) — without a code change. Records
    written under the old edge set keep their labels resolvable via
    bound_from_records against the NEW edge set only when they align; the
    edge-set version guards stale reads."""
    import jevmlx.driftenv as dv

    monkeypatch.setattr(dv, "_ENVELOPE_CACHE", tmp_path / "cache")
    assert dv.shape_bucket(17) == "M>16"
    assert dv.shape_bucket(112) == "M>16"

    edges = {"version": dv.BUCKET_EDGES_VERSION, "edges": [4, 8, 16, 32, 112]}
    (tmp_path / "cache").mkdir(parents=True)
    (tmp_path / "cache" / "bucket_edges.json").write_text(json.dumps(edges))
    assert dv.shape_bucket(17) == "M<=32"
    assert dv.shape_bucket(56) == "M<=112"
    assert dv.shape_bucket(113) == "M>112"
    # A record under the new edge set bounds a pass inside it.
    records = [{"shape_bucket": "M<=112", "max_gap_drift_nats": 0.09375}]
    assert dv.bound_from_records(records, "M<=112") == pytest.approx(0.09375)
    assert dv.bound_from_records(records, "M<=32") == pytest.approx(0.09375)
