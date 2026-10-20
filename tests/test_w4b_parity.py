"""W4-B: the bench's parity.json producer.

jevmlx/parity.py factors the W1-A slow parity test's logic into one shared
implementation; the bench runs it after each model load and writes
<model folder>/parity.json. A model whose parity fails is marked
parity_failed in SUMMARY.md and cannot enter the README compat table
(PR #29's check_parity reads the same file).
"""

import json

import pytest
from conftest import PARITY_ATOL, make_engine, skip_on_shared_runner
from conftest import YNLogitModel as _StableModel
from conftest import _Mod97Tokenizer as _CountTokenizer


class _DriftingModel(_StableModel):
    """Batch-shape-dependent logits: the batched suffix call (batch > 1)
    boosts the alias "B" token, so the batched pass disagrees with the
    row-per-pass (batch=1) scoring — winners flip, parity must FAIL
    regardless of atol. Needs >= 2 rows so a batched call happens."""

    def __call__(self, tokens, cache=None):
        out = super().__call__(tokens, cache=cache)
        if tokens.shape[0] > 1:  # batched suffix pass: boost alias "B"
            out = out.at[:, :, 67].add(5.0)
        return out


def _cases():
    """Two ad-hoc cases with 2+ fields each (2 rows per field -> a batched
    suffix call happens even with the tiny fake model; a single-field case
    scores each row in its own pass either way)."""
    # W5-D finding 41: parity cases carry the WHOLE preset dict (schema +
    # real context), matching bundled_preset_specs' new shape.
    return [
        (
            "mini",
            {
                "schema": {
                    "flag": {"type": "boolean", "description": "d"},
                    "pick": {"type": "enum", "description": "d", "choices": ["yes", "no"]},
                },
                "context": "Parity mini context: routine, all checks passed.",
            },
        ),
        (
            "mini2",
            {
                "schema": {
                    "grade": {"type": "enum", "description": "d", "choices": ["a", "b", "c"]},
                    "ok": {"type": "boolean", "description": "d"},
                },
                "context": "Parity mini2 context: escalation with two checks failing.",
            },
        ),
    ]


def test_parity_report_passes_on_stable_fake(tmp_path):
    from jevmlx.parity import bundled_preset_specs, parity_report, write_parity_json

    payload = parity_report(make_engine(_StableModel(), _CountTokenizer()), "fake/stable", _cases())
    assert payload["passed"] is True
    assert payload["status"] == "PASS"  # P4/I7
    assert payload["winners_identical"] is True
    assert payload["max_abs_drift_nats"] == 0.0
    assert payload["atol"] == PARITY_ATOL
    assert payload["max_raw_row_drift_nats"] == 0.0  # diagnostic only
    assert payload["max_gap_drift_nats"] == 0.0  # the gate (fail-closed at atol)
    assert payload["max_margin_drift_nats"] == 0.0
    assert payload["max_batched_drift_nats"] == 0.0  # recorded, not gated
    assert payload["rescore_gate"]["escaped_near_tie"] == []  # empty = pass
    assert "environment" in payload
    assert payload["model"] == "fake/stable"
    assert payload["prompt_version"].startswith("jevmlx-parallel-")
    assert payload["test"] == "test_w1a_scoring_parity_batch_vs_chunked_real_model"
    assert payload["cases"] == ["mini", "mini2"]
    assert payload["run_at"].endswith("Z")

    # Default cases = every bundled preset (6 shipped schemas).
    default_cases = bundled_preset_specs()
    assert [cid for cid, _ in default_cases] == [
        "code_security",
        "content_moderation",
        "fintech_fraud",
        "high_cardinality_255",
        "inbound_email",
        "support_triage",
    ]

    # write_parity_json writes the same payload to <dir>/parity.json.
    written = write_parity_json(
        make_engine(_StableModel(), _CountTokenizer()), "fake/stable", tmp_path, _cases()
    )
    on_disk = json.loads((tmp_path / "parity.json").read_text(encoding="utf-8"))
    assert on_disk == written
    assert on_disk["passed"] is True


def test_parity_report_fails_when_winners_flip(tmp_path):
    from jevmlx.parity import write_parity_json

    payload = write_parity_json(
        make_engine(_DriftingModel(), _CountTokenizer()),
        "fake/drift",
        tmp_path,
        _cases(),
        scoring="slots",
    )
    assert payload["passed"] is False
    assert payload["status"] == "FAIL"  # P4/I7: winners changed
    assert payload["winners_identical"] is False
    # F9: one artifact per scorer — a slots payload writes parity-slots.json.
    on_disk = json.loads((tmp_path / "parity-slots.json").read_text(encoding="utf-8"))
    assert on_disk["passed"] is False
    assert on_disk["scoring"] == "slots"


def test_parity_report_drift_when_winners_identical_and_large_drift(monkeypatch, tmp_path):
    """D6: DRIFT = winners identical + no escaped near-tie, regardless of
    the drift size. The old envelope-band term was self-referential (the
    band was round_up(0.05 + THIS run's own max gap drift), so the band
    check could never fail) — removed; the measured drift is reported as
    is. The drift is injected at the check boundary: the branch under test
    is the status decision, not the drift mechanism (covered by
    test_batched_parity_matrix_catches_row_drift)."""
    import jevmlx.parity as parity_mod
    from jevmlx.parity import write_parity_json

    big_drift = {
        "winners_identical": True,
        "max_abs_drift_nats": 0.078,
        "max_raw_row_drift_nats": 0.09,
        "max_gap_drift_nats": 0.078,
        "max_logp_drift_nats": 0.078,
        "max_margin_drift_nats": 0.063,
        "max_merged_rows": 8,
        "per_case": {},
    }
    monkeypatch.setattr(parity_mod, "check_batched_parity", lambda engine, cases, **kw: big_drift)

    payload = write_parity_json(
        make_engine(_StableModel(), _CountTokenizer()), "fake/drift", tmp_path, _cases()
    )
    assert payload["winners_identical"] is True
    assert (
        max(
            payload["max_abs_drift_nats"],
            payload["max_gap_drift_nats"],
            payload["max_margin_drift_nats"],
        )
        >= PARITY_ATOL
    )
    assert payload["passed"] is False
    assert payload["status"] == "DRIFT"
    on_disk = json.loads((tmp_path / "parity.json").read_text(encoding="utf-8"))
    # D6: the payload no longer persists the band field.
    assert "band" not in (on_disk.get("drift_envelope") or {})


def test_parity_report_drift_without_band_term_accepts_old_payload(tmp_path):
    """D6: an OLD parity.json that still carries drift_envelope.band parses
    fine through check_results — the field is ignored, never a failure.
    (Leaderboard acceptance of the same old shape is covered by
    test_leaderboard_includes_drift_model, whose fixture carries a band.)"""
    from benchmarks.check_results import check_parity
    from jevmlx.parity import PARITY_TEST_NAME

    old = {
        "model": "old/model",
        "test": PARITY_TEST_NAME,
        "scoring": "labels",  # F9: payloads record their scorer
        "passed": False,
        "status": "DRIFT",
        "max_abs_drift_nats": 0.078,
        "max_raw_row_drift_nats": 0.03,
        "max_gap_drift_nats": 0.078,
        "max_margin_drift_nats": 0.063,
        "max_batched_drift_nats": 0.078,
        "winners_identical": True,
        "atol": PARITY_ATOL,
        "drift_envelope": {"band": 0.14, "shape_bucket": "M>16"},
        "cases": ["a"],
        "run_at": "2026-09-20T12:00:00Z",
    }
    (tmp_path / "parity.json").write_text(json.dumps(old), encoding="utf-8")
    ok, problems = check_parity(tmp_path)
    assert ok, problems
    assert "envelope band" not in problems[0]


def test_parity_report_fails_on_escaped_near_tie(monkeypatch, tmp_path):
    """D6: an escaped near-tie (a field inside the 0.05 band at batch=1
    that the batched path did NOT rescore) is FAIL, not DRIFT — winners
    being identical is not enough when the rescore gate was bypassed.
    The escape is injected at the parity module boundary: constructing a
    real escape needs a rescore-capable model, and the branch under test
    is the status decision, not the rescore mechanism (covered in
    test_engine_fake.py)."""
    import jevmlx.parity as parity_mod

    escape = {
        "case": "mini",
        "field": "flag",
        "batch_one_margin": 0.01,
        "batched_margin": 0.01,
        "reference_near_tie": True,
        "production_rescored": False,
        "winner_differs": False,
        "escaped_near_tie": True,
    }
    monkeypatch.setattr(
        parity_mod,
        "_rescore_gate_safety_assertion",
        lambda engine, cases: {"escaped_near_tie": [escape], "per_field": [escape]},
    )
    payload = parity_mod.parity_report(
        make_engine(_StableModel(), _CountTokenizer()), "fake/escaped", _cases()
    )
    assert payload["winners_identical"] is True
    assert payload["rescore_gate"]["escaped_near_tie"]
    assert payload["passed"] is False
    assert payload["status"] == "FAIL"  # not DRIFT: the escape is a divergence


def test_summarize_gates_parity_failed_rows(tmp_path):
    """A model folder with failing parity.json: every row's accuracy is
    replaced by the parity_failed note. A passing parity.json leaves rows
    untouched. Missing parity.json gates too."""
    from benchmarks.summarize_results import summarize

    out = tmp_path / "results"
    model_dir = out / "m5-32gb-testslug"
    model_dir.mkdir(parents=True)
    for combo in ("parallel-slots-quality",):
        d = model_dir / combo
        d.mkdir()
        (d / "report.json").write_text(
            json.dumps({"metrics": {"accuracy": 0.9, "case_exact_match": 0.5}}),
            encoding="utf-8",
        )
    (model_dir / "parity.json").write_text(
        json.dumps(
            {
                "passed": False,
                "max_abs_drift_nats": 0.9,
                "atol": 0.05,
                "max_gap_drift_nats": 0.01,
                "max_margin_drift_nats": 0.01,
                "winners_identical": False,
            }
        ),
        encoding="utf-8",
    )
    summarize(out)
    summary = (out / "SUMMARY.md").read_text(encoding="utf-8")
    assert "parity_failed" in summary
    assert "max_abs_drift_nats=0.9" in summary
    assert "| 0.9 |" not in summary  # the accuracy number is gated

    # Passing parity: numbers survive.
    (model_dir / "parity.json").write_text(
        json.dumps(
            {
                "passed": True,
                "max_abs_drift_nats": 0.01,
                "atol": 0.05,
                "max_gap_drift_nats": 0.01,
                "winners_identical": True,
            }
        ),
        encoding="utf-8",
    )
    summarize(out)
    summary = (out / "SUMMARY.md").read_text(encoding="utf-8")
    assert "parity_failed" not in summary

    # Missing parity.json: gated too.
    (model_dir / "parity.json").unlink()
    summarize(out)
    summary = (out / "SUMMARY.md").read_text(encoding="utf-8")
    assert "parity.json missing" in summary


@pytest.mark.slow
@skip_on_shared_runner()
def test_parity_real_model_twin(engine):
    """Slow twin: the SAME check on a real model must pass with drift
    within PARITY_ATOL — this is exactly the W1-A slow test's invariant,
    now through the shared producer (all four bundled presets).

    Default-model-only: PARITY_ATOL (== INSTABILITY_BAND = 5e-2) is the
    batch-vs-chunked log_score drift measured on the 0.5B dev model. Larger
    models (1.5B, 7B) see a higher raw drift from Metal batch-shape matmul
    variation; the M5 parity step injects those models via MODEL_ID env,
    and this 0.5B-calibrated tolerance would flunk them. Skip unless the
    slow suite is running against the default model."""
    from conftest import MODEL_ID as _MODEL_ID

    if _MODEL_ID != "mlx-community/Qwen2.5-0.5B-Instruct-4bit":
        pytest.skip(f"PARITY_ATOL is 0.5B-calibrated; skipping for {_MODEL_ID}")

    from jevmlx.parity import parity_report

    payload = parity_report(engine, "twin/real-model")
    # Winners stay identical (the decision invariant holds).
    assert payload["winners_identical"] is True
    # The single-context W1-A gate still passes (measured ~0.044 < 0.05).
    assert payload["max_abs_drift_nats"] < PARITY_ATOL
    # W5c-1 review section 2: the batched pairwise GAP drift (0.070) and
    # margin drift (0.063) EXCEED the fixed 0.05 contract, so the batched
    # gate FAILS CLOSED — parity_passed is False for the 0.5B. The raw
    # drift (0.125) is a diagnostic, not the gate. This is the honest
    # result: the 0.5B does NOT pass batched parity at the fixed band.
    assert (
        payload["max_gap_drift_nats"] >= PARITY_ATOL
        or payload["max_margin_drift_nats"] >= PARITY_ATOL
    )
    assert payload["passed"] is False
    assert payload["cases"] == [
        "code_security",
        "content_moderation",
        "fintech_fraud",
        "high_cardinality_255",
        "inbound_email",
        "support_triage",
    ]


def test_parity_report_carries_v2_keys_and_passes_check_parity(monkeypatch, tmp_path):
    """W5c-1: parity_report runs BOTH checks — the payload carries the v2
    key set (the gated decomposition: max_gap_drift_nats +
    max_margin_drift_nats) and check_results.check_parity accepts it (no
    pre-v2 rejection, no failed stages). A winners-flip payload names its
    stage and is rejected."""
    from benchmarks.check_results import _parity_failed_stages, check_parity
    from jevmlx.parity import write_parity_json

    payload = write_parity_json(
        make_engine(_StableModel(), _CountTokenizer()), "fake/stable", tmp_path, _cases()
    )
    # v2 key set present.
    assert "max_raw_row_drift_nats" in payload  # diagnostic
    assert "max_abs_drift_nats" in payload
    assert "max_batched_drift_nats" in payload  # recorded, not gated
    assert "max_gap_drift_nats" in payload  # the gate
    assert "max_margin_drift_nats" in payload
    assert "rescore_gate" in payload
    assert "environment" in payload
    assert payload["winners_identical"] is True
    # check_parity accepts the file (not pre-v2, not failed).
    ok, problems = check_parity(tmp_path)
    assert ok is True, problems
    # And the stage classifier finds no failing stage.
    assert _parity_failed_stages(payload) == []

    # A winners-flip payload: the stage classifier names the stage and
    # check_parity rejects the folder. (The _DriftingModel produces DRIFT —
    # batch-shape noise inside the envelope band, winners identical — which
    # is now publishable, so check_parity returns OK for it. A real FAIL
    # requires winners_identical=False; that path is covered by
    # test_failed_parity_fails in test_check_results.py.) The drift is
    # injected at the check boundary (same as the D6 test): the branch
    # under test is check_parity's DRIFT handling, not the drift mechanism.
    import jevmlx.parity as parity_mod

    big_drift = {
        "winners_identical": True,
        "max_abs_drift_nats": 0.078,
        "max_raw_row_drift_nats": 0.09,
        "max_gap_drift_nats": 0.078,
        "max_logp_drift_nats": 0.078,
        "max_margin_drift_nats": 0.063,
        "max_merged_rows": 8,
        "per_case": {},
    }
    monkeypatch.setattr(parity_mod, "check_batched_parity", lambda engine, cases, **kw: big_drift)
    drift_dir = tmp_path.parent / "drift"
    drift_dir.mkdir()
    write_parity_json(
        make_engine(_StableModel(), _CountTokenizer()),
        "fake/drift",
        drift_dir,
        _cases(),
    )
    on_disk = json.loads((drift_dir / "parity.json").read_text(encoding="utf-8"))
    assert on_disk["passed"] is False
    # DRIFT (not FAIL): winners identical, inside envelope band.
    assert on_disk["status"] == "DRIFT"
    stages = _parity_failed_stages(on_disk)
    assert stages and stages != ["unspecified stage (payload carries no stage keys)"]
    # parity-gates: DRIFT is publishable — check_parity returns OK with a note.
    ok, problems = check_parity(drift_dir)
    assert ok is True
    assert any("DRIFT" in p for p in problems)
