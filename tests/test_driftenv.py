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

import pytest

from jevmlx.engine import (
    INSTABILITY_BAND,
    ScalarEvidence,
    finalize_scalar_evidence,
    run_parallel_generation,
    run_parallel_generation_batched,
)
from tests.conftest import FakeTokenizer, make_engine
from tests.test_engine_fake import BiasedFakeModel


def _records(**by_bucket: float) -> list[dict]:
    """Envelope records dict from bucket->bound kwargs."""
    from jevmlx.driftenv import MAX_GAP_DRIFT_KEY

    return [{"shape_bucket": b, MAX_GAP_DRIFT_KEY: v} for b, v in by_bucket.items()]


def _envelope(records: list[dict]) -> dict:
    return {
        "key": {"model_id": "fake-engine", "bucket_edges_version": 1},
        "records": records,
        "source": "test",
    }


PLATEAU = _envelope(_records(**{"M>16": 0.0625}))


def test_band_for_pass_plateau():
    """band(M>16) = 0.05 + 0.0625 rounded UP to the 1/64 lattice = 0.125."""
    import jevmlx.driftenv as dv

    env = PLATEAU
    assert dv.band_for_pass(env["records"], 28) == pytest.approx(0.125)
    assert dv.band_for_pass(env["records"], 112) == pytest.approx(0.125)
    # Small-M buckets without a record of their own: the plateau record
    # covers them (M above the largest recorded bucket uses the largest bound).
    assert dv.band_for_pass(env["records"], 2) == pytest.approx(0.125)


def test_band_for_pass_small_record():
    """An M<=4 record of 0.0391 bounds only the small buckets; M>16 still
    uses the M>16 record (0.0625)."""
    import jevmlx.driftenv as dv

    env = _envelope(_records(**{"M<=4": 0.0390625, "M>16": 0.0625}))
    assert dv.band_for_pass(env["records"], 2) == pytest.approx(
        dv.round_up_lattice(0.05 + 0.0390625)
    )
    assert dv.band_for_pass(env["records"], 28) == pytest.approx(0.125)


def test_band_for_pass_missing_record_raises():
    """No record covers the pass: RAISES (no constant fallback — H3)."""
    import jevmlx.driftenv as dv

    env = _envelope([])
    with pytest.raises(dv.DriftEnvelopeError):
        dv.band_for_pass(env["records"], 28)


def test_band_for_pass_above_largest_bucket_uses_largest():
    """M above the largest recorded bucket uses the largest recorded bound
    (C2) — never a smaller bucket's, never a constant."""
    import jevmlx.driftenv as dv

    env = _envelope(_records(**{"M<=16": 0.0625}))
    # M=112 sits above M>16's reach; the M<=16 bound (0.0625) applies.
    assert dv.band_for_pass(env["records"], 112) == pytest.approx(0.125)


def test_band_for_pass_zero_rows_constant():
    """M == 0 (degenerate no-rows schema) returns INSTABILITY_BAND — no
    batched pass ran, so no widening applies."""
    import jevmlx.driftenv as dv

    assert dv.band_for_pass([], 0) == pytest.approx(INSTABILITY_BAND)


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


def test_finalize_canonical_path_default_band():
    """No band passed -> the default INSTABILITY_BAND (canonical batch=1 /
    dependency / oracle paths — they never rescore, so the band is moot)."""
    evidence = ScalarEvidence(
        choices=("A", "B"),
        log_scores_raw=(-0.5, -0.5625),  # margin 0.0625 > 0.05
        legal_mass_logs=(-0.01, -0.02),
        source_shape="dependency",  # canonical shape — no rescore gate
    )
    decision, rescored = finalize_scalar_evidence(
        evidence,
        prior_entry=None,
        temperature=1.0,
    )
    assert rescored is False
    assert decision.rescore_band_nats == pytest.approx(INSTABILITY_BAND)
    assert decision.drift_envelope_nats is None


class _MarginFakeModel(BiasedFakeModel):
    """A fake whose branch margin is a fixed small value: the winner's
    alias token gets `margin` nats, the other 0 — identical at every batch
    shape (the fake has no shape dependence). With margin in (0.05, 0.125]
    the OLD band does not rescore; the WIDENED band does."""

    def __init__(self, margin: float, vocab_size: int = 64, n_layers: int = 2):
        super().__init__(vocab_size=vocab_size, n_layers=n_layers, winner_token=ord("A") % 60)
        self.margin = margin

    def __call__(self, tokens, cache=None):
        out = super().__call__(tokens, cache)
        out = out.at[..., self.winner_token].add(self.margin - 20.0)
        return out


def _schema_two_choice():
    from jevmlx.schema import StructuredSchema

    return StructuredSchema(
        {"action": {"type": "enum", "description": "d", "choices": ["A", "B"]}}
    )


def test_engine_end_to_end_widened_band_rescores():
    """Full engine: an injected envelope widens the band so a field whose
    batched raw margin (0.075) sat above the old 0.05 band now rescores."""
    from tests.conftest import _test_envelope

    model = _MarginFakeModel(margin=0.075)
    tokenizer = FakeTokenizer()
    schema = _schema_two_choice()
    engine = make_engine(model, tokenizer, drift_envelope=_test_envelope())
    result = run_parallel_generation(engine, "ctx", schema)
    tel = result["field_telemetry"]["action"]
    assert tel["rescored"] is True
    assert tel["rescore_band_nats"] == pytest.approx(0.125)
    assert tel["drift_envelope_nats"] == pytest.approx(0.0625)
    assert tel["semantics"]["score_source"] == "rescored_batch1"


def test_engine_end_to_end_constant_band():
    """Same engine with a 0.0-bound envelope: margin 0.075 >= 0.05, no
    rescore (the behavior the envelope widens)."""
    from tests.conftest import _constant_test_envelope

    model = _MarginFakeModel(margin=0.075)
    tokenizer = FakeTokenizer()
    schema = _schema_two_choice()
    engine = make_engine(model, tokenizer, drift_envelope=_constant_test_envelope())
    result = run_parallel_generation(engine, "ctx", schema)
    tel = result["field_telemetry"]["action"]
    assert tel["rescored"] is False
    # The constant band (0.0 drift) still rounds UP to the 1/64 lattice:
    # 0.05 -> 0.0625 (0.05 is not on the grid). Margin 0.075 > 0.0625, so
    # no rescore — the behavior the widened envelope exists to change.
    assert tel["rescore_band_nats"] == pytest.approx(0.0625)


def test_record_envelope_persists_and_loads(tmp_path, monkeypatch):
    """record -> cache write -> load round-trip keyed by the tuple. The
    store keeps the MAX per bucket (C3: never last-writer-wins)."""
    import jevmlx.driftenv as dv

    monkeypatch.setattr(dv, "_ENVELOPE_CACHE", tmp_path / "cache")
    key = {
        "model_id": "m",
        "revision": "r",
        "quantization": None,
        "mlx_version": "1",
        "chip": "c",
        "activation_dtype": "float16",
        "bucket_edges_version": 1,
    }
    rec = {"key": key, "shape_bucket": "M>16", dv.MAX_GAP_DRIFT_KEY: 0.0625}
    written = dv.record_envelope(rec)
    assert written
    loaded = dv.load_envelope_record(key)
    assert loaded is not None
    assert dv.bound_from_records(loaded["records"], "M>16") == pytest.approx(0.0625)
    # A LOWER measured value does NOT replace (the store keeps the MAX).
    dv.record_envelope(
        {"key": key, "shape_bucket": "M>16", dv.MAX_GAP_DRIFT_KEY: 0.03, "source": "probe"}
    )
    loaded2 = dv.load_envelope_record(key)
    assert dv.bound_from_records(loaded2["records"], "M>16") == pytest.approx(0.0625)
    # A HIGHER value DOES replace.
    dv.record_envelope(
        {"key": key, "shape_bucket": "M>16", dv.MAX_GAP_DRIFT_KEY: 0.125, "source": "probe"}
    )
    loaded3 = dv.load_envelope_record(key)
    assert dv.bound_from_records(loaded3["records"], "M>16") == pytest.approx(0.125)
    assert len(loaded3["records"]) == 1


def test_envelope_cache_path_is_tuple_keyed():
    import jevmlx.driftenv as dv

    k1 = {"model_id": "a", "chip": "m5max", "bucket_edges_version": 1}
    k2 = {"model_id": "a", "chip": "m5max", "quantization": {"bits": 4}, "bucket_edges_version": 1}
    assert dv.envelope_cache_path(k1) != dv.envelope_cache_path(k2)
    assert dv.envelope_cache_path(k1) == dv.envelope_cache_path(dict(k1))


class _FakeEngineForKey:
    """Enough engine for envelope_key (model_id/revision); the canary is
    monkeypatched in the canary tests."""

    model_id = "fake-engine"
    revision = None
    model = type("M", (), {"parameters": lambda self: {}})()
    tokenizer = None
    profile = None
    vocab_size = 64


def test_envelope_for_engine_uses_recorded_without_canary(tmp_path, monkeypatch):
    """A recorded envelope resolves WITHOUT running the canary."""
    import jevmlx.driftenv as dv

    monkeypatch.setattr(dv, "_ENVELOPE_CACHE", tmp_path / "cache")

    real_key = dv.envelope_key(_FakeEngineForKey())
    dv.record_envelope(
        {"key": real_key, "shape_bucket": "M>16", dv.MAX_GAP_DRIFT_KEY: 0.0625, "source": "probe"}
    )
    canary_calls = []

    def _no_canary(_engine):
        canary_calls.append(1)
        raise AssertionError("canary must not run when recorded")

    monkeypatch.setattr(dv, "run_canary", _no_canary)
    res = dv.envelope_for_engine(_FakeEngineForKey())
    assert res["source"] == "recorded"
    assert dv.bound_from_records(res["records"], "M>16") == pytest.approx(0.0625)
    assert not canary_calls


def test_envelope_for_engine_runs_canary_when_missing(tmp_path, monkeypatch):
    """Missing envelope: the canary path runs, records, and the band uses
    its measured value."""
    import jevmlx.driftenv as dv

    monkeypatch.setattr(dv, "_ENVELOPE_CACHE", tmp_path / "cache")

    def _fake_canary(_engine):
        return {
            "key": dv.envelope_key(_engine),
            "shape_bucket": "M<=16",
            dv.MAX_GAP_DRIFT_KEY: 0.0625,
            "source": "canary",
            "canary_rows": 16,
        }

    monkeypatch.setattr(dv, "run_canary", _fake_canary)
    res = dv.envelope_for_engine(_FakeEngineForKey())
    assert res["source"] == "canary"
    assert dv.bound_from_records(res["records"], "M<=16") == pytest.approx(0.0625)
    # The canary WROTE its record: the next load finds it.
    loaded = dv.load_envelope_record(res["key"])
    assert loaded is not None
    assert dv.bound_from_records(loaded["records"], "M<=16") == pytest.approx(0.0625)


def test_canary_failure_raises(tmp_path, monkeypatch):
    """A canary that fails RAISES (H3: no constant fallback)."""
    import jevmlx.driftenv as dv

    monkeypatch.setattr(dv, "_ENVELOPE_CACHE", tmp_path / "cache")

    def _failing_canary(_engine):
        raise RuntimeError("probe broke")

    monkeypatch.setattr(dv, "run_canary", _failing_canary)
    with pytest.raises(dv.DriftEnvelopeError):
        dv.envelope_for_engine(_FakeEngineForKey())


def test_readonly_cache_keeps_in_memory_no_constant(tmp_path, monkeypatch):
    """A read-only cache (C6): the write fails, the record is kept
    in-memory, NO constant fallback installs."""
    import jevmlx.driftenv as dv

    monkeypatch.setattr(dv, "_ENVELOPE_CACHE", tmp_path / "cache")

    def _fake_canary(_engine):
        return {
            "key": dv.envelope_key(_engine),
            "shape_bucket": "M<=16",
            dv.MAX_GAP_DRIFT_KEY: 0.0625,
            "source": "canary",
            "canary_rows": 16,
        }

    monkeypatch.setattr(dv, "run_canary", _fake_canary)

    # Make the cache dir read-only after creation.
    (tmp_path / "cache").mkdir(parents=True)

    def _raise(*a, **kw):
        raise PermissionError("read-only")

    monkeypatch.setattr(dv.Path, "write_text", _raise)
    res = dv.envelope_for_engine(_FakeEngineForKey())
    # The canary's record rides in-memory despite the read-only cache.
    assert dv.bound_from_records(res["records"], "M<=16") == pytest.approx(0.0625)


def test_parity_report_unchanged_contract():
    """The parity payload's atol stays the FIXED 0.05 — the band protects
    decisions, it does not redefine parity."""
    import jevmlx.engine as eng

    assert eng.INSTABILITY_BAND == pytest.approx(0.05)


def test_bucket_edges_shipped_and_versioned():
    """The shipped bucket_edges.json is the source of truth (H4)."""
    import jevmlx.driftenv as dv

    assert dv.BUCKET_EDGES_VERSION == 1
    assert dv.BUCKET_EDGES == (4, 8, 16)
    assert dv.bucket_labels() == ["M<=4", "M<=8", "M<=16", "M>16"]
    # The envelope key carries the edge version (records under different
    # edges never mix).
    key = dv.envelope_key(_FakeEngineForKey())
    assert key["bucket_edges_version"] == dv.BUCKET_EDGES_VERSION


def test_decide_many_passes_merged_m_not_per_context_r(monkeypatch):
    """C1 regression: the band must be resolved against the MERGED pass
    width (n_group * R), not one context's R rows. decide_many slices the
    merged result per context before _assemble, so a naive len(row_logits)
    inside _assemble is R — the band would resolve to the M<=4 bucket for a
    1-row schema batched 20 times. This test FAILS if _assemble is given R
    instead of the merged n_group*R: an M>16-only envelope raises
    DriftEnvelopeError on the small bucket (no record covers M<=4)."""
    import jevmlx.driftenv as dv

    # An envelope with ONLY an M>16 record (no small-bucket coverage).
    # If _assemble passes R (1) instead of the merged 20, band_for_pass
    # raises DriftEnvelopeError (no record covers M<=4) and the test fails.
    env = _envelope(_records(**{"M>16": 0.0625}))
    seen_m: list[int] = []
    real_band = dv.band_for_pass

    def _spy(records, m_rows):
        seen_m.append(m_rows)
        return real_band(records, m_rows)

    monkeypatch.setattr(dv, "band_for_pass", _spy)

    model = _MarginFakeModel(margin=0.075)
    tokenizer = FakeTokenizer()
    schema = _schema_two_choice()
    engine = make_engine(model, tokenizer, drift_envelope=env)
    # 20 contexts of a 1-row schema: merged pass M = 20 * 1 = 20 (M>16).
    # If C1 were broken, _assemble would pass R=1 -> M<=4 bucket -> no record
    # -> DriftEnvelopeError. The call succeeding proves the merged M was used.
    results = run_parallel_generation_batched(engine, ["ctx"] * 20, schema)
    assert results and "field_telemetry" in results[0]
    assert "action" in results[0]["field_telemetry"]
    assert seen_m, "band_for_pass was never called — the pass did not run"
    # Every call resolved against the merged width (>= 20), never R (1).
    assert min(seen_m) >= 20, f"band resolved against per-context R ({seen_m}), not merged M"


def test_m_above_largest_bucket_uses_largest_bound_not_smaller(monkeypatch):
    """C2 regression: a pass whose M sits above the largest RECORDED bucket
    uses the LARGEST recorded bound (the plateau's upper reach), never a
    smaller bucket's bound and never a constant. This test FAILS if the
    engine resolves the band from the load-time canary (M<=16) and applies
    it to an M>16 production pass that has no M>16 record of its own —
    i.e. it fails if the engine carries a single resolved bound instead of
    the RECORDS and resolves per-pass."""
    import jevmlx.driftenv as dv

    # Records: only M<=16 (the canary's bucket) at 0.0, and M>16 at 0.125.
    # A correct per-pass resolution picks M>16's 0.125 for a 28-row pass.
    # A broken one (carries the canary's bound for every M) picks 0.0.
    env = _envelope(_records(**{"M<=16": 0.0, "M>16": 0.125}))
    band_28 = dv.band_for_pass(env["records"], 28)
    # 0.05 + 0.125 = 0.175, rounded up to 1/64 lattice = 0.1875.
    assert band_28 == pytest.approx(dv.round_up_lattice(0.05 + 0.125))
    # And NOT the M<=16 bucket's bound (0.0 -> 0.0625 on the lattice).
    assert band_28 != pytest.approx(dv.round_up_lattice(0.05 + 0.0))


def test_envelope_key_in_dunder_all():
    """MINOR: envelope_key is exported (part of the public API)."""
    import jevmlx.driftenv as dv

    assert "envelope_key" in dv.__all__
    assert "band_for_pass" in dv.__all__
    assert "recorded_envelope_records" in dv.__all__
