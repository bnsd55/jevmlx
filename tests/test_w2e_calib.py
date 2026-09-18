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

from jevmlx.calibrate import calibrated_log_odds, fit_logistic
from jevmlx.engine import run_parallel_generation
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
    result = run_parallel_generation(model, FakeTokenizer(), "ctx", _multi_schema())
    telemetry = result["field_telemetry"]["flags"]
    assert result["parsed_json"]["flags"]["value"] == ["x", "y", "z"]  # P(yes)~0.88
    assert telemetry["calibrated"] is None
    assert "threshold" not in telemetry
    assert telemetry["per_option"]["x"] == pytest.approx(1.0 / (1.0 + math.exp(-2.0)), rel=1e-6)


def test_engine_with_calibration_dict_selects_by_sign():
    """a=1, b=0: calibrated log-odds = raw log-odds = 2 > 0 -> all selected
    (same as uncalibrated here); a=1, b=-3: calibrated = -1 < 0 -> none."""
    model = _BiasedMultiModel(yes_logit=1.0, no_logit=-1.0)
    tok = FakeTokenizer()
    keep = run_parallel_generation(
        model, tok, "ctx", _multi_schema(), calibration={"multi": {"a": 1.0, "b": 0.0}}
    )
    assert keep["parsed_json"]["flags"]["value"] == ["x", "y", "z"]
    assert keep["field_telemetry"]["flags"]["calibrated"] == {"a": 1.0, "b": 0.0}
    drop = run_parallel_generation(
        model, tok, "ctx", _multi_schema(), calibration={"multi": {"a": 1.0, "b": -3.0}}
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
    model = _BiasedMultiModel(yes_logit=1.0, no_logit=-1.0)
    path = tmp_path / "calib.json"
    path.write_text(json.dumps({"temperature": 1.0, "multi": {"a": 2.0, "b": -5.0}}))
    result = run_parallel_generation(
        model, FakeTokenizer(), "ctx", _multi_schema(), calibration=str(path)
    )
    # calibrated = 2*2 - 5 = -1 < 0 -> nothing selected
    assert result["parsed_json"]["flags"]["value"] == []
    assert result["field_telemetry"]["flags"]["calibrated"] == {"a": 2.0, "b": -5.0}


def test_engine_calibration_payload_validation():
    model = _BiasedMultiModel()
    tok = FakeTokenizer()
    schema = _multi_schema()
    with pytest.raises(ValueError, match="calibration file not found"):
        run_parallel_generation(model, tok, "ctx", schema, calibration="/nope/missing.json")
    with pytest.raises(ValueError, match='must carry numeric "a" and "b"'):
        run_parallel_generation(model, tok, "ctx", schema, calibration={"multi": {"a": 1}})
    with pytest.raises(ValueError, match="must be finite"):
        run_parallel_generation(
            model, tok, "ctx", schema, calibration={"multi": {"a": 1e999, "b": 0}}
        )


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
            {"multi": {"a": 1.0, "b": 0.0}},
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
