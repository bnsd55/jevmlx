"""P11 / F9: one parity artifact per scorer.

- T1: scoring flows through EVERY parity call — raw rows, final
  decisions, batched and single-context. Never a hardcoded/default scorer
  inside parity.
- T2: the batched raw-row comparison covers EVERY context copy of the
  merged pass, not only the first (main read rows 0..R-1 only).
- T3: multi fields contribute per-option corrected log-odds drift with
  the same atol contract as scalar fields (main skipped them: ls_full is
  None -> continue).

All three FAIL on main.
"""

from __future__ import annotations

from conftest import YNLogitModel, make_engine

from tests.test_w4b_parity import _cases, _CountTokenizer


def test_t1_scoring_flows_through_every_parity_call(monkeypatch):
    """T1: with scoring="slots", every engine call parity makes — batched,
    chunked, single-context, final decisions — receives scoring="slots"."""
    import jevmlx.engine as engine_mod

    captured: list[tuple[str, dict]] = []
    real_rpg = engine_mod.run_parallel_generation
    real_rpgb = engine_mod.run_parallel_generation_batched

    def spy_rpg(engine, context, schema, **kwargs):
        captured.append(("run_parallel_generation", dict(kwargs)))
        return real_rpg(engine, context, schema, **kwargs)

    def spy_rpgb(engine, contexts, schema, **kwargs):
        captured.append(("run_parallel_generation_batched", dict(kwargs)))
        return real_rpgb(engine, contexts, schema, **kwargs)

    monkeypatch.setattr(engine_mod, "run_parallel_generation", spy_rpg)
    monkeypatch.setattr(engine_mod, "run_parallel_generation_batched", spy_rpgb)

    from jevmlx.parity import check_batched_parity, check_scoring_parity

    engine = make_engine(YNLogitModel(), _CountTokenizer())
    check_batched_parity(engine, _cases(), scoring="slots")
    check_scoring_parity(engine, _cases(), scoring="slots")
    assert captured, "parity made no engine calls"
    for name, kwargs in captured:
        assert kwargs.get("scoring") == "slots", (
            f"{name} called without the payload scorer (kwargs={kwargs})"
        )


def test_t2_raw_row_comparison_covers_every_context_copy(monkeypatch):
    """T2: raw-row drift injected ONLY in the 2nd context copy of the
    merged pass is detected. Main compares only rows 0..R-1 (the first
    copy) and reports 0.0."""
    import mlx.core as mx

    import jevmlx.engine as engine_mod
    from jevmlx.parity import check_batched_parity

    real_score_rows = engine_mod._score_rows
    N_ROWS = 2  # the fixture: 2 rows per case (R), copies 1..3 get the drift

    def drifting_score_rows(
        model,
        cache,
        rows,
        row_decision,
        vocab_size,
        pad_id,
        auto_max_rows,
        ledger,
        cache_slots=None,
    ):
        result = real_score_rows(
            model,
            cache,
            rows,
            row_decision,
            vocab_size,
            pad_id,
            auto_max_rows,
            ledger,
            cache_slots=cache_slots,
        )
        if cache_slots is not None and len(cache_slots) > N_ROWS:
            # Drift every NON-first context copy: +1 nat on the first child
            # logit of each row >= N_ROWS (copies 1..n_ctx-1). The first
            # copy (rows 0..R-1) stays identical to the batch=1 reference.
            for ridx in list(result.row_logits):
                if ridx >= N_ROWS:
                    vals = result.row_logits[ridx]
                    result.row_logits[ridx] = [vals[0] + 1.0, *vals[1:]]
        return result

    monkeypatch.setattr(engine_mod, "_score_rows", drifting_score_rows)
    # The raw section consumes mx only via the parity module's own imports;
    # reference mx here so a missing backend fails loudly, not silently.
    assert mx is not None

    result = check_batched_parity(make_engine(YNLogitModel(), _CountTokenizer()), _cases())
    assert result["max_raw_row_drift_nats"] > 0.0, (
        "drift in the 2nd+ context copies went undetected"
    )


def test_t3_multi_option_drift_fails_parity():
    """T3: a multi field whose per-option corrected log-odds drift above
    atol (batched vs chunked) FAILS parity. Main skipped multi fields
    entirely (log_scores is None -> continue) and the payload passed."""
    from jevmlx.parity import parity_report

    class _MultiDriftModel(YNLogitModel):
        """Boosts the multi YES token only in batched suffix calls
        (>1-row chunks): per-option corrected log-odds drift >= atol."""

        def __call__(self, tokens, cache=None):
            out = super().__call__(tokens, cache=cache)
            if tokens.shape[0] > 1 and tokens.shape[1] > 1:
                # the multi option's YES token id (_mod97(ord('Y'))).
                out = out.at[:, :, 90].add(3.0)
            return out

    multi_case = (
        "multi_only",
        {
            "schema": {
                "tags": {"type": "multi", "description": "d", "choices": ["x", "y"]},
            },
            "context": "Parity multi context: routine, all checks passed.",
        },
    )
    payload = parity_report(
        make_engine(_MultiDriftModel(), _CountTokenizer()), "fake/multi-drift", [multi_case]
    )
    assert payload["max_abs_drift_nats"] >= payload["atol"], (
        "multi per-option drift was not measured (main silently skipped it)"
    )
    assert payload["passed"] is False


def test_t1b_payload_records_its_scorer():
    """F9: the payload records which scorer it covers, and a slots payload
    is a DIFFERENT artifact from the labels one."""
    import json
    import tempfile
    from pathlib import Path

    from jevmlx.parity import write_parity_json

    with tempfile.TemporaryDirectory() as td:
        out = Path(td)
        labels = write_parity_json(
            make_engine(YNLogitModel(), _CountTokenizer()), "fake/stable", out, _cases()
        )
        assert labels["scoring"] == "labels"
        assert json.loads((out / "parity.json").read_text())["scoring"] == "labels"
        slots = write_parity_json(
            make_engine(YNLogitModel(), _CountTokenizer()),
            "fake/stable",
            out,
            _cases(),
            scoring="slots",
        )
        assert slots["scoring"] == "slots"
        # A non-default scorer gets its OWN artifact, never overwrites the
        # default scorer's parity.json.
        assert json.loads((out / "parity-slots.json").read_text())["scoring"] == "slots"
        assert json.loads((out / "parity.json").read_text())["scoring"] == "labels"


def test_check_parity_rejects_payload_without_scorer(tmp_path):
    """F9: a parity payload with no scoring key is rejected — no silent
    default (the pre-F9 artifact proved parity for neither scorer)."""
    import json

    from benchmarks.check_results import check_parity
    from jevmlx.parity import PARITY_TEST_NAME

    legacy = {
        "model": "old/model",
        "test": PARITY_TEST_NAME,
        "passed": True,
        "status": "PASS",
        "max_abs_drift_nats": 0.01,
        "max_raw_row_drift_nats": 0.01,
        "max_gap_drift_nats": 0.01,
        "max_margin_drift_nats": 0.01,
        "winners_identical": True,
        "atol": 0.05,
        "cases": ["a"],
        "run_at": "2026-09-20T12:00:00Z",
    }
    (tmp_path / "parity.json").write_text(json.dumps(legacy), encoding="utf-8")
    ok, problems = check_parity(tmp_path)
    assert not ok
    assert any("no scoring key" in p for p in problems)


def test_check_parity_requires_every_scorer_artifact_to_pass(tmp_path):
    """F9: parity.json passes but parity-slots.json FAILs -> the folder is
    rejected (one bad scorer is a bad model)."""
    import json

    from benchmarks.check_results import check_parity
    from jevmlx.parity import PARITY_TEST_NAME

    def payload(scoring: str, passed: bool) -> dict:
        return {
            "model": "m",
            "test": PARITY_TEST_NAME,
            "scoring": scoring,
            "passed": passed,
            "status": "PASS" if passed else "FAIL",
            "max_abs_drift_nats": 0.01,
            "max_raw_row_drift_nats": 0.01,
            "max_gap_drift_nats": 0.01,
            "max_margin_drift_nats": 0.01,
            "winners_identical": passed,
            "atol": 0.05,
            "cases": ["a"],
            "run_at": "2026-09-20T12:00:00Z",
        }

    (tmp_path / "parity.json").write_text(json.dumps(payload("labels", True)))
    (tmp_path / "parity-slots.json").write_text(json.dumps(payload("slots", False)))
    ok, problems = check_parity(tmp_path)
    assert not ok
    assert any("parity-slots.json" in p for p in problems)


def test_leaderboard_unverified_for_pre_f9_parity(tmp_path):
    """F9 decision (c): a legacy parity payload (no scoring key, recorded
    status DRIFT) keeps the model row in the table — the Parity column
    reads UNVERIFIED (pre-F9) with no drift number. A FAIL status stays
    excluded under any shape (a winner change is real under any scorer)."""
    import json
    from pathlib import Path

    from benchmarks.leaderboard import _local_rows

    def write_model(root: Path, name: str, parity: dict | None) -> None:
        model_dir = root / f"m1-8gb-{name}"
        combo = model_dir / "parallel-slots-typesafe"
        combo.mkdir(parents=True)
        (combo / "run.json").write_text(
            json.dumps(
                {
                    "config": {
                        "track": "parallel",
                        "dataset_path": "typesafe",
                        "model": "fake-1b",
                    }
                }
            )
        )
        (combo / "report.json").write_text(
            json.dumps(
                {
                    "metrics": {
                        "agreement": {"agreement_common_subset": 0.9, "n_cases": 45},
                        "by_workflow": {},
                    }
                }
            )
        )
        (combo / "timing.json").write_text(
            json.dumps({"median": {"per_item_end_to_end_ms": 100.0}})
        )
        (model_dir / "parity.json").write_text(json.dumps(parity), encoding="utf-8")

    legacy_drift = {
        "model": "m",
        "test": "w1a",
        "passed": False,
        "status": "DRIFT",
        "max_abs_drift_nats": 0.078,
        "max_raw_row_drift_nats": 0.03,
        "max_gap_drift_nats": 0.078,
        "max_margin_drift_nats": 0.063,
        "atol": 0.05,
        "winners_identical": True,
        "cases": ["a"],
        "run_at": "2026-09-20T12:00:00Z",
    }
    legacy_fail = {**legacy_drift, "passed": False, "status": "FAIL", "winners_identical": False}

    root = tmp_path / "results"
    root.mkdir()
    write_model(root, "legacy", legacy_drift)
    write_model(root, "fail", legacy_fail)

    rows = _local_rows(root)
    models = {r["model"]: r["parity_status"] for r in rows}
    assert models["fake-1b"].count("UNVERIFIED (pre-F9)") == 1 or any(
        r["parity_status"] == "UNVERIFIED (pre-F9)" for r in rows
    )
    # The FAIL-status legacy model is excluded entirely.
    assert all(r["parity_status"] != "FAIL" for r in rows)
    # The UNVERIFIED row carries no drift number.
    unv = [r for r in rows if r["parity_status"] == "UNVERIFIED (pre-F9)"]
    assert unv and all(r["parity_max_drift"] == 0.0 for r in unv)


def test_check_parity_rejects_unknown_scorer(tmp_path):
    """F9: a payload tagged with an unknown scorer is neither legacy nor a
    valid per-scorer artifact — it fails closed."""
    import json

    from benchmarks.check_results import check_parity
    from jevmlx.parity import PARITY_TEST_NAME

    weird = {
        "model": "m",
        "test": PARITY_TEST_NAME,
        "scoring": "llm-judge",
        "passed": True,
        "status": "PASS",
        "max_abs_drift_nats": 0.01,
        "max_raw_row_drift_nats": 0.01,
        "max_gap_drift_nats": 0.01,
        "max_margin_drift_nats": 0.01,
        "winners_identical": True,
        "atol": 0.05,
        "cases": ["a"],
        "run_at": "2026-09-20T12:00:00Z",
    }
    (tmp_path / "parity.json").write_text(json.dumps(weird), encoding="utf-8")
    ok, problems = check_parity(tmp_path)
    assert not ok
    assert any("unknown scoring" in p for p in problems)
