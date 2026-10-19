"""W2-E step 2: pooled logistic calibration for multi fields.

- fit_logistic on a synthetic separable set recovers sign and ordering.
- Decision path without calibration: fixed P(yes) >= 0.5 rule (as before).
- Decision path with calibration: calibrated_log_odds > 0 selects.
- multi_threshold is deleted from the API, CLI and docs (no dual path).
"""

import json
import math
import random

import pytest
from conftest import YNLogitModel as _BiasedMultiModel
from conftest import _Mod97Tokenizer as FakeTokenizer
from conftest import make_engine

from jevmlx.calibrate import CalibrationBundle, calibrated_log_odds, fit_logistic
from jevmlx.engine import PROMPT_VERSION, run_parallel_generation
from jevmlx.schema import StructuredSchema


def _multi_schema():
    return StructuredSchema(
        {
            "flags": {
                "type": "multi",
                "description": "d",
                "choices": ["x", "y", "z"],
            }
        }
    )


# ---------------------------------------------------------------- fit tests


def test_fit_logistic_separable_set_recovers_sign_and_ordering():
    """Positives at high log-odds, negatives at low: the fit must give
    calibrated(positive) > calibrated(negative) for every pair, the sign of
    a must be positive (more log-odds -> more yes), and the calibrated
    boundary must separate the sets."""
    rng = random.Random(11)
    samples = []
    for _ in range(200):
        y = rng.randrange(2)
        x = rng.gauss(3.0, 0.8) if y else rng.gauss(-3.0, 0.8)
        samples.append((x, y))
    a, b = fit_logistic(samples)
    assert a > 0.5  # positive slope: log-odds up -> yes
    # Sign/ordering: a monotone increasing map preserves the ordering.
    pos = [x for x, y in samples if y]
    neg = [x for x, y in samples if not y]
    assert calibrated_log_odds(a, b, max(neg)) < calibrated_log_odds(a, b, min(pos))
    # Perfectly separable: every positive calibrates above 0, negative below.
    assert all(calibrated_log_odds(a, b, x) > 0 for x in pos)
    assert all(calibrated_log_odds(a, b, x) < 0 for x in neg)


def test_fit_logistic_flipped_labels_give_negative_slope():
    rng = random.Random(11)
    samples = []
    for _ in range(200):
        y = rng.randrange(2)
        x = rng.gauss(3.0, 0.8) if y else rng.gauss(-3.0, 0.8)
        samples.append((x, 1 - y))  # labels flipped
    a, _b = fit_logistic(samples)
    assert a < -0.5


def test_fit_logistic_separable_pool_stays_bounded():
    """F2: on a perfectly separable pool the unregularized MLE diverges, so
    the default l2=1e-3 must keep (a, b) finite and bounded while the
    calibrated sign still separates the classes."""
    rng = random.Random(7)
    samples = []
    for _ in range(400):
        y = rng.randrange(2)
        x = rng.gauss(4.0, 1.0) if y else rng.gauss(-4.0, 1.0)
        samples.append((x, y))
    a, b = fit_logistic(samples)  # default l2
    assert math.isfinite(a) and math.isfinite(b)
    assert abs(a) < 100 and abs(b) < 100  # bounded, not diverging
    # Decision boundary still separates: sign of calibrated log-odds.
    pos = [x for x, y in samples if y]
    neg = [x for x, y in samples if not y]
    assert all(calibrated_log_odds(a, b, x) > 0 for x in pos)
    assert all(calibrated_log_odds(a, b, x) < 0 for x in neg)


def test_fit_logistic_validation():
    with pytest.raises(ValueError, match="at least one"):
        fit_logistic([])
    with pytest.raises(ValueError, match="labels must be"):
        fit_logistic([(1.0, 2)])


# ---------------------------------------------------- engine decision paths


def test_engine_without_calibration_fixed_half_rule():
    """No calibration: P(yes) = sigmoid(yes_logit - no_logit) per option;
    selection at P(yes) >= 0.5; telemetry calibrated is None and there is
    NO threshold key anywhere."""
    model = _BiasedMultiModel(yes_logit=1.0, no_logit=-1.0)
    result = run_parallel_generation(make_engine(model, FakeTokenizer()), "ctx", _multi_schema())
    telemetry = result["field_telemetry"]["flags"]
    assert result["parsed_json"]["flags"]["value"] == ["x", "y", "z"]  # P(yes)~0.88
    assert telemetry["calibrated"] is None
    assert "threshold" not in telemetry
    assert telemetry["per_option"]["x"] == pytest.approx(1.0 / (1.0 + math.exp(-2.0)), rel=1e-6)


def test_engine_with_calibration_bundle_selects_by_sign():
    """a=1, b=0: calibrated log-odds = raw log-odds = 2 > 0 -> all selected
    (same as uncalibrated here); a=1, b=-3: calibrated = -1 < 0 -> none."""
    model = _BiasedMultiModel(yes_logit=1.0, no_logit=-1.0)
    tok = FakeTokenizer()
    keep = run_parallel_generation(
        make_engine(model, tok),
        "ctx",
        _multi_schema(),
        calibration=CalibrationBundle.from_payload({"multi": {"a": 1.0, "b": 0.0}}),
    )
    assert keep["parsed_json"]["flags"]["value"] == ["x", "y", "z"]
    assert keep["field_telemetry"]["flags"]["calibrated"] == {"a": 1.0, "b": 0.0}
    drop = run_parallel_generation(
        make_engine(model, tok),
        "ctx",
        _multi_schema(),
        calibration=CalibrationBundle.from_payload({"multi": {"a": 1.0, "b": -3.0}}),
    )
    assert drop["parsed_json"]["flags"]["value"] == []
    assert drop["field_telemetry"]["flags"]["calibrated"] == {"a": 1.0, "b": -3.0}
    # Margin in probability units (F1): calibrated c = -1 per option ->
    # |sigmoid(-1) - 0.5|; raw log-odds ride calibrated_log_odds.
    assert drop["field_telemetry"]["flags"]["margin"] == pytest.approx(
        abs(1.0 / (1.0 + math.exp(1.0)) - 0.5)
    )
    assert drop["field_telemetry"]["flags"]["calibrated_log_odds"] == {
        "x": -1.0,
        "y": -1.0,
        "z": -1.0,
    }


def test_engine_calibration_from_file(tmp_path):
    """F3: run_parallel_generation ALSO takes the bundle only — the file is
    loaded at the boundary (CalibrationBundle.load), then passed in."""
    model = _BiasedMultiModel(yes_logit=1.0, no_logit=-1.0)
    path = tmp_path / "calib.json"
    # W5-C finding 22: ONE bundle shape — the same one the CLI writes.
    path.write_text(
        json.dumps(
            {
                "model_revision": "test",
                "prompt_version": PROMPT_VERSION,
                "scoring": "slots",
                "prior_mode": "off",
                "scalar": {"temperature": 1.0},
                "multi": {"a": 2.0, "b": -5.0},
            }
        )
    )
    result = run_parallel_generation(
        make_engine(model, FakeTokenizer()),
        "ctx",
        _multi_schema(),
        scoring="slots",
        calibration=CalibrationBundle.load(str(path)),
    )
    # calibrated = 2*2 - 5 = -1 < 0 -> nothing selected
    assert result["parsed_json"]["flags"]["value"] == []
    assert result["field_telemetry"]["flags"]["calibrated"] == {"a": 2.0, "b": -5.0}


def test_engine_calibration_payload_validation():
    """F3: the engine takes CalibrationBundle | None ONLY.

    A path is loaded at the public boundary (api._resolve_calibration) —
    the engine-level run_parallel_generation raises TypeError for a str.
    The boundary behaviors (missing file, bad dict) are asserted in
    tests/test_api.py against decide(); here we assert the engine-level
    contract: wrong types -> TypeError naming the rule, a bad bundle's
    provenance/payload -> ValueError from from_payload before the engine.
    """
    from jevmlx.calibrate import CalibrationBundle

    model = _BiasedMultiModel()
    tok = FakeTokenizer()
    schema = _multi_schema()
    with pytest.raises(TypeError, match="calibration must be a CalibrationBundle or None"):
        run_parallel_generation(
            make_engine(model, tok), "ctx", schema, calibration="/nope/missing.json"
        )
    with pytest.raises(TypeError, match="calibration must be a CalibrationBundle or None"):
        run_parallel_generation(
            make_engine(model, tok), "ctx", schema, calibration={"multi": {"a": 1}}
        )
    with pytest.raises(ValueError, match='must carry numeric "a" and "b"'):
        CalibrationBundle.from_payload({"multi": {"a": "x", "b": 0}})


# ------------------------------------------------------------- API surface


def test_api_multi_threshold_is_deleted():
    """decide/decide_many/run_parallel_generation have no multi_threshold
    parameter anywhere (no dual path)."""
    import inspect

    from jevmlx import api
    from jevmlx.engine import run_parallel_generation as rpg

    for fn in (api.decide, api.decide_many, rpg):
        params = inspect.signature(fn).parameters
        assert "multi_threshold" not in params, fn.__name__
        assert "calibration" in params, fn.__name__


def test_cli_multi_threshold_flag_is_deleted():
    """The CLI has no --multi-threshold; --calibration exists; the calibrate
    subcommand writes --out."""

    # Parse args through main()'s parser via a --help probe: main() builds
    # the parser inline, so exercise it through argparse errors instead.
    import io
    from contextlib import redirect_stderr, redirect_stdout

    from jevmlx import cli

    buf_out, buf_err = io.StringIO(), io.StringIO()
    with pytest.raises(SystemExit), redirect_stdout(buf_out), redirect_stderr(buf_err):
        cli.main(["decide", "--help"])
    assert "--multi-threshold" not in buf_out.getvalue()
    assert "--calibration" in buf_out.getvalue()
    with pytest.raises(SystemExit), redirect_stdout(buf_out), redirect_stderr(buf_err):
        cli.main(["calibrate", "--help"])
    assert "--out" in buf_out.getvalue()


# ---------------------------------------------------- openai parity (calib)


def test_openai_multi_calibrated_selection_and_no_threshold_key():
    """decide_openai's multi telemetry mirrors the native contract:
    calibrated (a, b) or None, margin in calibrated nats, no threshold key."""
    from unittest.mock import patch

    from jevmlx.openai_slots import _decide_multi_field

    schema = _multi_schema()
    field = schema.fields["flags"]

    class FakeResponse:
        def __init__(self, top):
            self.top = top

    def fake_chat(
        base_url, model, messages, api_key=None, timeout=None, temperature=None, extra_payload=None
    ):
        # chat_completions_raw returns choices[0]; P(Y) = 0.9 among top-2
        return {
            "logprobs": {
                "content": [
                    {
                        "top_logprobs": [
                            {"token": '"Y"', "logprob": math.log(0.9)},
                            {"token": '"N"', "logprob": math.log(0.1)},
                        ]
                    }
                ]
            }
        }

    with patch("jevmlx.openai_slots.chat_completions_raw", side_effect=fake_chat):
        _parsed, telemetry, _n = _decide_multi_field(
            "http://x",
            "m",
            None,
            schema,
            "ctx",
            "flags",
            field,
            5.0,
            # W5-C finding 22: the typed CalibrationBundle (the ONE shape
            # from_payload parses; fields the CLI does not set stay None).
            CalibrationBundle.from_payload(
                {
                    "model_revision": "test",
                    "prompt_version": PROMPT_VERSION,
                    "scoring": "slots",
                    "prior_mode": "off",
                    "multi": {"a": 1.0, "b": 0.0},
                }
            ),
            FakeTokenizer(),
        )
    assert "threshold" not in telemetry
    assert telemetry["calibrated"] == {"a": 1.0, "b": 0.0}
    # P(yes)=0.9 -> log-odds ~2.197 > 0 -> all selected
    assert telemetry["value"] == ["x", "y", "z"]
    assert telemetry["margin"] == pytest.approx(2.197, rel=1e-2)

    with patch("jevmlx.openai_slots.chat_completions_raw", side_effect=fake_chat):
        _parsed, telemetry_off, _n = _decide_multi_field(
            "http://x", "m", None, schema, "ctx", "flags", field, 5.0, None, FakeTokenizer()
        )
    assert telemetry_off["calibrated"] is None
    assert telemetry_off["value"] == ["x", "y", "z"]  # 0.9 >= 0.5 uncalibrated


# ------------------------------------------- P8: one decision coordinate --


def _built_multi(model, tokenizer, choices):
    """A real built plan + batched dispatch + prefill cache for one multi
    field, for decision-coordinate tests (the same machinery parity's raw
    stage uses)."""
    from jevmlx.engine import (
        _build_schema_rows,
        _prefill,
        _score_rows,
        dispatch_rows,
        field_rows_of,
    )
    from jevmlx.timing import Ledger

    schema = StructuredSchema({"flags": {"type": "multi", "description": "d", "choices": choices}})
    engine = make_engine(model, tokenizer)
    built = _build_schema_rows(schema, engine.tokenizer, "slots")
    pf = _prefill(engine.model, engine.tokenizer, "ctx", schema, Ledger(), "slots", engine.profile)
    scored = _score_rows(
        engine.model,
        pf.cache,
        built["rows"],
        built["row_decision"],
        engine.vocab_size,
        built["pad_id"],
        max(1, len(built["rows"])),
        Ledger(),
    )
    dispatch = dispatch_rows(built, scored)
    return engine, schema, built, dispatch, field_rows_of(built)["flags"], pf.cache


def _run_multi_field(
    engine, schema, built, dispatch, fr, cache, *, prior=None, calib=None, temperature=1.0
):
    """score_multi_field over the hand-built dispatch (the signature is
    stable on main — the P8 failures there are behavioral, not import
    errors)."""
    from jevmlx.engine import score_multi_field
    from jevmlx.timing import Ledger

    return score_multi_field(
        engine.model,
        cache,
        built,
        dispatch,
        fr,
        schema.fields["flags"],
        built["field_plans"]["flags"],
        prior=prior,
        calib=calib,
        scoring="slots",
        temperature=temperature,
        vocab_size=engine.vocab_size,
        pad_id=built["pad_id"],
        ledger=Ledger(),
        drift_band=(0.125, None),
    )


def test_T1_calibrated_selection_uses_corrected_log_odds():
    """P8 T1 (the P6 probe, mid-level): the corrected evidence says NO but
    the raw pair says YES — with a calibrator the option must NOT be
    selected. Raw [1.0, -1.0] (log-odds 2.0), prior [2.0, -1.0] (difference
    3.0): z = 2.0 - 3.0 = -1.0 < 0. Main's calibrated branch keys on the
    RAW pair (c = 2.0 > 0) and selects the option — the F6 bug."""
    engine, schema, built, dispatch, fr, cache = _built_multi(
        _BiasedMultiModel(vocab_size=128), FakeTokenizer(), ["A", "B"]
    )
    ridx_a, ridx_b = fr.option_idxs
    dispatch.option_pair[ridx_a] = [1.0, -1.0]  # raw says YES (2.0)
    dispatch.option_pair[ridx_b] = [-2.0, 0.0]  # decisively no under any rule
    prior = {"flags": {"option_pairs": {"A": [2.0, -1.0]}}}  # prior diff 3.0
    calib = CalibrationBundle.from_payload({"multi": {"a": 1.0, "b": 0.0}})
    outcome = _run_multi_field(engine, schema, built, dispatch, fr, cache, prior=prior, calib=calib)
    telemetry = outcome.telemetry
    assert outcome.parsed["value"] == []  # corrected z = -1.0: NOT selected
    assert telemetry["per_option"]["A"] == pytest.approx(1.0 / (1.0 + math.exp(1.0)))
    assert telemetry["prior_corrected"] is True


def test_T2_reported_probability_is_the_selection_number():
    """P8 T2: for EVERY option the reported P(yes) is the exact number the
    selection consumed — calibrated p = sigmoid(a*z + b) on the CORRECTED
    log-odds. Main reports the uncalibrated corrected softmax there (a raw
    a=2.0/b=0.5 calibrator never reaches the reported number)."""
    engine, schema, built, dispatch, fr, cache = _built_multi(
        _BiasedMultiModel(vocab_size=128), FakeTokenizer(), ["A", "B"]
    )
    ridx_a, ridx_b = fr.option_idxs
    dispatch.option_pair[ridx_a] = [1.0, 0.0]  # raw log-odds 1.0
    dispatch.option_pair[ridx_b] = [0.2, 0.0]  # raw log-odds 0.2
    prior = {"flags": {"option_pairs": {"A": [0.5, 0.0]}}}  # z_A = 0.5
    calib = CalibrationBundle.from_payload({"multi": {"a": 2.0, "b": 0.5}})
    outcome = _run_multi_field(engine, schema, built, dispatch, fr, cache, prior=prior, calib=calib)
    telemetry = outcome.telemetry
    z_a, z_b = 0.5, 0.2  # corrected log-odds
    expected = {
        "A": 1.0 / (1.0 + math.exp(-(2.0 * z_a + 0.5))),
        "B": 1.0 / (1.0 + math.exp(-(2.0 * z_b + 0.5))),
    }
    for option, p_yes in expected.items():
        assert telemetry["per_option"][option] == pytest.approx(p_yes, rel=1e-9)
        # Selection used the same number: p > 0.5 exactly on the selected side.
        selected = option in outcome.parsed["value"]
        assert selected == (p_yes > 0.5)
    assert telemetry["calibrated_log_odds"] == {"A": 2.0 * z_a + 0.5, "B": 2.0 * z_b + 0.5}

    # Uncalibrated: the reported p is sigmoid(z / T) — the exact number the
    # z >= 0 threshold used (identical to main's corrected softmax).
    engine2, schema2, built2, dispatch2, fr2, cache2 = _built_multi(
        _BiasedMultiModel(vocab_size=128), FakeTokenizer(), ["A", "B"]
    )
    ridx_a2, ridx_b2 = fr2.option_idxs
    dispatch2.option_pair[ridx_a2] = [1.0, 0.0]
    dispatch2.option_pair[ridx_b2] = [0.2, 0.0]
    outcome2 = _run_multi_field(
        engine2,
        schema2,
        built2,
        dispatch2,
        fr2,
        cache2,
        prior={"flags": {"option_pairs": {"A": [0.5, 0.0]}}},
        temperature=5.0,
    )
    telemetry2 = outcome2.telemetry
    assert telemetry2["per_option"]["A"] == pytest.approx(1.0 / (1.0 + math.exp(-0.1)))
    assert telemetry2["per_option"]["B"] == pytest.approx(1.0 / (1.0 + math.exp(-0.04)))
    assert outcome2.parsed["value"] == ["A", "B"]  # both z > 0


def test_T3_joint_solver_scores_the_same_coordinate():
    """P8 T3: the trusted-count joint solver must score options in the SAME
    corrected coordinate as the threshold path. Raw ordering: A (1.0) > B
    (0.9); the prior flips the corrected ordering: z_A = 1.0 - 3.0 = -2.0,
    z_B = 0.9. A trusted count of exactly 1 must select top-1 by z = B.
    Main scores the solver by the RAW pair and selects A."""
    engine, schema, built, dispatch, fr, cache = _built_multi(
        _BiasedMultiModel(vocab_size=128), FakeTokenizer(), ["A", "B"]
    )
    ridx_a, ridx_b = fr.option_idxs
    dispatch.option_pair[ridx_a] = [1.0, 0.0]
    dispatch.option_pair[ridx_b] = [0.9, 0.0]
    # Trusted count: bucket '1' wins its row by 3.0 nats (> COUNT_MARGIN_MIN).
    c_ridx = fr.count_idxs[0]
    c_bi = next(iter(dispatch.count_node_logits[c_ridx]))
    dispatch.count_node_logits[c_ridx] = {c_bi: [0.0, 3.0, 0.0, 0.0, 0.0]}
    prior = {"flags": {"option_pairs": {"A": [3.0, 0.0]}}}
    outcome = _run_multi_field(engine, schema, built, dispatch, fr, cache, prior=prior)
    telemetry = outcome.telemetry
    assert telemetry["count_choice"] == "1"
    assert telemetry["reconciled_by"] == "count"
    assert outcome.parsed["value"] == ["B"]  # top-1 by corrected z, not raw


def test_T4_no_prior_prior_corrected_false():
    """P8 T4: with no prior at all the multi telemetry carries
    prior_corrected = False (main: the key is ABSENT)."""
    engine, schema, built, dispatch, fr, cache = _built_multi(
        _BiasedMultiModel(vocab_size=128), FakeTokenizer(), ["A", "B"]
    )
    ridx_a, ridx_b = fr.option_idxs
    dispatch.option_pair[ridx_a] = [1.0, 0.0]
    dispatch.option_pair[ridx_b] = [0.2, 0.0]
    outcome = _run_multi_field(engine, schema, built, dispatch, fr, cache)
    assert outcome.telemetry["prior_corrected"] is False
    assert "prior_option_pairs" not in outcome.telemetry


def test_T5_collect_multi_corrected_z_matches_engine():
    """P8 T5 (calibration fit): collect_multi(prior_correction=True) folds
    each option's z to the SAME corrected log-odds the engine applies the
    (a, b) to — raw difference minus the prior difference from the
    telemetry's prior_option_pairs. Fitting on a different quantity than
    the engine applies is the F6 root cause."""
    from jevmlx.calibrate import collect_multi

    engine = make_engine(
        _BiasedMultiModel(yes_logit=1.0, no_logit=-1.0, vocab_size=128), FakeTokenizer()
    )
    cases = [
        {
            "schema": {"flags": {"type": "multi", "description": "d", "choices": ["x", "y"]}},
            "context": "ctx",
            "labels": {"flags": ["x"]},
        }
    ]
    samples_off = collect_multi(engine, cases, prior_correction=False)
    assert samples_off == [(2.0, 1), (2.0, 0)]  # raw log-odds: 1.0 - (-1.0)
    samples_on = collect_multi(engine, cases, prior_correction=True)
    # The neutral prior here is the SAME evidence (deterministic fake): the
    # corrected z is ~0 for both options.
    assert samples_on[0][0] == pytest.approx(0.0, abs=1e-9)
    assert samples_on[1][0] == pytest.approx(0.0, abs=1e-9)
    assert [y for _, y in samples_on] == [1, 0]
