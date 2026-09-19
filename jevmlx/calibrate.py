"""Calibration for the parallel decision engine.

Two calibrators, one module:

- Temperature: one scalar T fitted on labeled cases minimizes the mean NLL
  of the true choice under softmax(scores / T). For enum/boolean fields.
- Pooled logistic: a single (a, b) pair fitted on raw multi option
  log-odds (yes_logit - no_logit from ``option_logit_pairs``) against
  yes/no labels — calibrated_log_odds = a * log_odds + b, an option is
  selected when calibrated_log_odds > 0. One pool across ALL multi fields
  and options: multi options are homogeneous binary decisions, so they
  share one calibrator. No scipy, no numpy — stdlib math only
  (gradient-descent on the logistic NLL).
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass

Sample = tuple[list[float], int]  # (per-choice scores at T=1, labeled choice index)
MultiSample = tuple[float, int]  # (raw log-odds yes-no, label 1=yes 0=no)


def collect(engine, cases: Sequence[dict]) -> list[Sample]:
    """Run run_parallel_generation once per case at T=1 and keep raw scores
    for every labeled field Takes the loaded :class:`Engine`.
    Cases: [{"schema": {...}, "context": str, "labels": {field: value}}]
    (labels may cover a subset of fields).

    multi fields are skipped with a logged warning: their scores are
    per-option p_true values, not a choice distribution, so NLL fitting
    does not apply to them. Use :func:`collect_multi` for them."""
    import logging

    from jevmlx.engine import run_parallel_generation
    from jevmlx.schema import StructuredSchema

    logger = logging.getLogger(__name__)

    samples: list[Sample] = []
    for case in cases:
        schema = StructuredSchema(case["schema"])
        result = run_parallel_generation(engine, case["context"], schema, temperature=1.0)
        for fname, label in case["labels"].items():
            telemetry = result["field_telemetry"][fname]
            fdef = schema[fname]
            if fdef.field_type == "multi":
                logger.warning("calibrate.collect: skipping multi field '%s'", fname)
                continue
            choices = ["true", "false"] if fdef.field_type == "boolean" else fdef.choices
            label_str = (
                str(label) if fdef.field_type != "boolean" else ("true" if label else "false")
            )
            if label_str not in choices:
                raise ValueError(f"case label {fname}={label!r} not in choices {choices}")
            log_scores = telemetry["log_scores"]
            samples.append(([log_scores[choice] for choice in choices], choices.index(label_str)))
    return samples


def _nll(scores: Sequence[float], label_idx: int, t: float) -> float:
    scaled = [s / t for s in scores]
    m = max(scaled)
    log_z = m + math.log(sum(math.exp(s - m) for s in scaled))
    return log_z - scaled[label_idx]


def fit_temperature(
    samples: Sequence[Sample], lo: float = 0.05, hi: float = 20.0, iters: int = 60
) -> float:
    """Golden-section search for the T in [lo, hi] minimizing mean NLL."""

    def mean_nll(t: float) -> float:
        return sum(_nll(s, y, t) for s, y in samples) / len(samples)

    inv_phi = (math.sqrt(5) - 1) / 2
    a, b = lo, hi
    c = b - inv_phi * (b - a)
    d = a + inv_phi * (b - a)
    fc, fd = mean_nll(c), mean_nll(d)
    for _ in range(iters):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - inv_phi * (b - a)
            fc = mean_nll(c)
        else:
            a, c, fc = c, d, fd
            d = a + inv_phi * (b - a)
            fd = mean_nll(d)
    return round((a + b) / 2, 4)


def _probs(scores: Sequence[float], t: float) -> list[float]:
    scaled = [s / t for s in scores]
    m = max(scaled)
    exps = [math.exp(s - m) for s in scaled]
    z = sum(exps)
    return [e / z for e in exps]


def ece(samples: Sequence[Sample], t: float = 1.0, bins: int = 10) -> float:
    """Expected calibration error: mean |max-prob - accuracy| over equal-width
    confidence bins, weighted by bin size."""
    if not samples:
        return 0.0
    bin_conf = [0.0] * bins
    bin_acc = [0.0] * bins
    bin_n = [0] * bins
    for scores, label_idx in samples:
        probs = _probs(scores, t)
        p_max = max(probs)
        correct = 1.0 if max(range(len(probs)), key=probs.__getitem__) == label_idx else 0.0
        b = min(bins - 1, int(p_max * bins))
        bin_conf[b] += p_max
        bin_acc[b] += correct
        bin_n[b] += 1
    total = len(samples)
    return sum(
        (bin_n[i] / total) * abs(bin_conf[i] / bin_n[i] - bin_acc[i] / bin_n[i])
        for i in range(bins)
        if bin_n[i]
    )


def load_cases(path: str) -> list:
    cases = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def collect_multi(engine, cases: Sequence[dict]) -> list[MultiSample]:
    """Raw multi option log-odds + yes/no labels for the pooled logistic fit.

    Same cases shape as :func:`collect`; runs the engine once per case and,
    for every labeled multi field, takes ``option_logit_pairs`` — the raw
    [yes, no] logits per option (bug 8, T=1, caller-temperature-agnostic) —
    folds each to log-odds log P(yes) - log P(no), and pairs it with the
    label (1 = option in the labeled set, 0 = not).

    Raises ValueError naming the field when a labeled value is not one of
    the field's choices (or not a list/set of them).
    """
    from jevmlx.engine import run_parallel_generation
    from jevmlx.schema import StructuredSchema

    samples: list[MultiSample] = []
    for case in cases:
        schema = StructuredSchema(case["schema"])
        result = run_parallel_generation(engine, case["context"], schema, temperature=1.0)
        for fname, label in case["labels"].items():
            fdef = schema[fname]
            if fdef.field_type != "multi":
                continue
            if isinstance(label, str):
                labeled_options = {label}
            elif isinstance(label, (list, set, tuple)):
                labeled_options = set(label)
            else:
                raise ValueError(
                    f"case label {fname}={label!r} for a multi field must be "
                    "an option name or a list/set of option names"
                )
            unknown = labeled_options - set(fdef.choices)
            if unknown:
                raise ValueError(
                    f"case label {fname}={sorted(unknown)!r} not in choices {fdef.choices}"
                )
            pairs = result["field_telemetry"][fname]["option_logit_pairs"]
            for option, (yes_logit, no_logit) in pairs.items():
                log_odds = yes_logit - no_logit
                samples.append((log_odds, 1 if option in labeled_options else 0))
    return samples


def fit_logistic(
    samples: Sequence[MultiSample], *, iters: int = 300, lr: float = 0.5, l2: float = 1e-3
) -> tuple[float, float]:
    """Pooled logistic calibration: maximize the Bernoulli log-likelihood of
    the labels under sigmoid(a * log_odds + b) with plain gradient descent
    (stdlib only). Returns (a, b).

    ``l2`` is an L2 penalty weight on ``(a, b)`` (b is penalized too — the
    fit is data-driven, not shrunk toward a prior). It defaults to 1e-3, not
    0: on a perfectly separable pool the unregularized MLE diverges (the
    coefficients grow without bound as iterations climb), so the default
    keeps the fit finite and bounded while leaving the decision boundary
    (the calibrated sign) untouched.
    """
    if not samples:
        raise ValueError("fit_logistic needs at least one (log_odds, label) sample")
    for _, label in samples:
        if label not in (0, 1):
            raise ValueError(f"labels must be 0 or 1, got {label!r}")
    a, b = 1.0, 0.0
    n = len(samples)
    for _ in range(iters):
        ga = gb = 0.0
        for x, y in samples:
            z = a * x + b
            # Stable sigmoid
            p = 1.0 / (1.0 + math.exp(-z)) if z >= 0 else math.exp(z) / (1.0 + math.exp(z))
            ga += (p - y) * x
            gb += p - y
        ga = ga / n + l2 * a
        gb = gb / n + l2 * b
        a -= lr * ga
        b -= lr * gb
    return round(a, 6), round(b, 6)


def calibrated_log_odds(a: float, b: float, log_odds: float) -> float:
    """Apply the pooled calibration: a * log_odds + b. Selection rule:
    calibrated value > 0."""
    return a * log_odds + b


@dataclass(frozen=True)
class CalibrationBundle:
    """Typed calibration artifact (W5-C finding 22).

    One object carries every fitted calibrator plus the provenance it is
    valid under. The engine derives the scalar temperature and the multi
    (a, b) from the bundle and REJECTS a request the bundle does not
    describe (wrong prompt version, scoring mode, prior mode, or model
    revision) instead of silently applying numbers fitted elsewhere.

    ``prior_mode`` records which input the multi calibrator was fitted on
    (finding 21): ``"off"`` = raw evidence log-odds; ``"neutral_v1"`` =
    prior-corrected log-odds (the neutral-context pass subtracted). Fit and
    apply must use the same input — the engine enforces it.

    ``load`` reads the ONE bundle shape ``jevmlx calibrate --out`` writes:
    ``{"model_revision", "prompt_version", "scoring", "prior_mode", "scalar":
    {"temperature"}, "multi": {"a", "b"}}`` — the CLI writes it and the
    engine reads it; there is no other shape.
    """

    temperature: float | None = None
    multi_a: float | None = None
    multi_b: float | None = None
    model_revision: str | None = None
    prompt_version: str | None = None
    scoring: str | None = None
    prior_mode: str = "off"

    def __post_init__(self) -> None:
        if self.temperature is not None and (
            not math.isfinite(self.temperature) or self.temperature <= 0
        ):
            raise ValueError(f"bundle temperature must be finite > 0, got {self.temperature!r}")
        for name, val in (("multi_a", self.multi_a), ("multi_b", self.multi_b)):
            if val is not None and not math.isfinite(val):
                raise ValueError(f"bundle {name} must be finite, got {val!r}")
        if (self.multi_a is None) != (self.multi_b is None):
            raise ValueError("bundle multi calibration needs BOTH a and b")
        if self.prior_mode not in ("off", "neutral_v1"):
            raise ValueError(
                f"bundle prior_mode must be 'off' or 'neutral_v1', got {self.prior_mode!r}"
            )
        if self.scoring is not None and self.scoring not in ("slots", "labels"):
            raise ValueError(f"bundle scoring must be 'slots' or 'labels', got {self.scoring!r}")

    @classmethod
    def from_payload(cls, payload: dict) -> CalibrationBundle:
        """Parse THE bundle JSON shape (model_revision, prompt_version,
        scoring, prior_mode, scalar.temperature, multi.a/b) — the one shape
        ``jevmlx calibrate --out`` writes."""
        if not isinstance(payload, dict):
            raise ValueError(f"calibration payload must be a dict, got {type(payload).__name__}")
        model_revision = payload.get("model_revision")
        prompt_version = payload.get("prompt_version")
        scoring = payload.get("scoring")
        prior_mode = payload.get("prior_mode", "off")
        temperature = None
        scalar = payload.get("scalar")
        if isinstance(scalar, dict):
            try:
                temperature = float(scalar["temperature"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f'calibration["scalar"] must carry numeric "temperature": {exc}'
                ) from exc
        elif "scalar" in payload and scalar is not None:
            raise ValueError('calibration["scalar"] must be a dict {"temperature": ..}')
        if temperature is None and "temperature" in payload:
            # The legacy top-level-temperature shape is NOT the bundle shape —
            # reject it loudly so a stale file never half-loads.
            raise ValueError(
                'calibration["temperature"] is the retired CLI shape; the bundle '
                'shape is {"scalar": {"temperature": ..}} (jevmlx calibrate --out writes it)'
            )
        multi_a = multi_b = None
        multi = payload.get("multi")
        if multi is not None:
            if not isinstance(multi, dict):
                raise ValueError('calibration["multi"] must be a dict {"a": .., "b": ..}')
            try:
                multi_a, multi_b = float(multi["a"]), float(multi["b"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f'calibration["multi"] must carry numeric "a" and "b": {exc}'
                ) from exc
        return cls(
            temperature=temperature,
            multi_a=multi_a,
            multi_b=multi_b,
            model_revision=str(model_revision) if model_revision is not None else None,
            prompt_version=str(prompt_version) if prompt_version is not None else None,
            scoring=str(scoring) if scoring is not None else None,
            prior_mode=str(prior_mode),
        )

    @classmethod
    def load(cls, path: str) -> CalibrationBundle:
        """Load and validate a calibration JSON file (file I/O lives here,
        at the boundary — never inside the engine's hot path)."""
        try:
            with open(path, encoding="utf-8") as f:
                payload = json.load(f)
        except FileNotFoundError as exc:
            raise ValueError(f"calibration file not found: {path}") from exc
        except json.JSONDecodeError as exc:
            raise ValueError(f"calibration file is not valid JSON: {path}: {exc}") from exc
        try:
            return cls.from_payload(payload)
        except ValueError as exc:
            raise ValueError(f"calibration file {path}: {exc}") from exc

    @property
    def has_multi(self) -> bool:
        return self.multi_a is not None

    @property
    def has_scalar(self) -> bool:
        return self.temperature is not None

    def identity(self) -> str:
        """Stable short provenance id for telemetry: revision + prompt +
        scoring + prior mode. Empty segments collapse to 'anon' so the id
        is still informative for ad-hoc inline bundles."""
        parts = [
            self.model_revision or "anon",
            self.prompt_version or "anon",
            self.scoring or "anon",
            self.prior_mode,
        ]
        return "/".join(parts)


def accuracy(samples: Sequence[Sample]) -> float:
    """Fraction of samples whose argmax choice matches the label."""
    if not samples:
        return 0.0
    hits = sum(1 for s, y in samples if max(range(len(s)), key=lambda i: s[i]) == y)
    return hits / len(samples)
