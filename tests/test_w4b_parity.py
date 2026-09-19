"""W4-B: the bench's parity.json producer.

jevmlx/parity.py factors the W1-A slow parity test's logic into one shared
implementation; the bench runs it after each model load and writes
<model folder>/parity.json. A model whose parity fails is marked
parity_failed in SUMMARY.md and cannot enter the README compat table
(PR #29's check_parity reads the same file).
"""

import json

import pytest
from conftest import PARITY_ATOL, make_engine
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
    assert payload["winners_identical"] is True
    assert payload["max_abs_drift_nats"] == 0.0
    assert payload["atol"] == PARITY_ATOL
    assert payload["model"] == "fake/stable"
    assert payload["prompt_version"].startswith("jevmlx-parallel-")
    assert payload["test"] == "test_w1a_scoring_parity_batch_vs_chunked_real_model"
    assert payload["cases"] == ["mini", "mini2"]
    assert payload["run_at"].endswith("Z")

    # Default cases = every bundled preset (4 shipped schemas).
    default_cases = bundled_preset_specs()
    assert [cid for cid, _ in default_cases] == [
        "code_security",
        "fintech_fraud",
        "high_cardinality_255",
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
        make_engine(_DriftingModel(), _CountTokenizer()), "fake/drift", tmp_path, _cases()
    )
    assert payload["passed"] is False
    assert payload["winners_identical"] is False
    on_disk = json.loads((tmp_path / "parity.json").read_text(encoding="utf-8"))
    assert on_disk["passed"] is False


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
            {"passed": False, "max_abs_drift_nats": 0.9, "atol": 0.05, "winners_identical": False}
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
            {"passed": True, "max_abs_drift_nats": 0.01, "atol": 0.05, "winners_identical": True}
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
    assert payload["winners_identical"] is True
    assert payload["max_abs_drift_nats"] < PARITY_ATOL
    assert payload["passed"] is True
    assert payload["cases"] == [
        "code_security",
        "fintech_fraud",
        "high_cardinality_255",
        "support_triage",
    ]
