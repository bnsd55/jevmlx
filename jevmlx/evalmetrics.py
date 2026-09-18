"""Eval metrics over predictions.jsonl (see E-contracts).

Pure Python + math, no third-party dependencies. Every metric treats an
invalid prediction as wrong (unconditional field accuracy); ``correct`` is
respected when present (null label -> excluded from accuracy denominators).
"""

from __future__ import annotations

import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

__all__ = [
    "balanced_accuracy",
    "load_predictions",
    "macro_f1",
    "perturbation_flip_rate",
    "compute_metrics",
    "typesafe_agreement",
    "tvd_vs_consensus",
    "exact_record_accuracy",
    "constraint_violation_rate",
    "child_accuracy_given_parent_correct",
    "order_flip_rate",
]


def load_predictions(path: str | Path) -> list[dict]:
    """Read a predictions.jsonl file (one JSON object per non-blank line)."""
    records = []
    for line in Path(path).read_text().splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def _valid_correct(record: dict) -> bool:
    """True iff this prediction counts as correct (invalid => wrong)."""
    if record.get("correct") is not None:
        return bool(record["correct"])
    return bool(record.get("valid")) and record.get("prediction") is not None


def _labelled(records: list[dict]) -> list[dict]:
    """Records that have a label (correct may be False when invalid)."""
    return [r for r in records if r.get("label") is not None or r.get("correct") is not None]


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _tvd(dist_a: dict[str, float], dist_b: dict[str, float]) -> float:
    """Total variation distance between two distributions over a shared support."""
    keys = set(dist_a) | set(dist_b)
    return 0.5 * sum(abs(dist_a.get(k, 0.0) - dist_b.get(k, 0.0)) for k in keys)


def _finite(*values: float) -> bool:
    return all(isinstance(v, (int, float)) and math.isfinite(v) for v in values)


# ---------------------------------------------------------------- accuracies


def field_accuracy(records: list[dict]) -> float | None:
    """Unconditional field accuracy: correct / labelled, invalid counts wrong."""
    labelled = _labelled(records)
    if not labelled:
        return None
    return sum(1 for r in labelled if _valid_correct(r)) / len(labelled)


def case_exact_match(records: list[dict]) -> float | None:
    """Share of cases where every labelled field is correct (invalid = wrong)."""
    by_case: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_case[record["case_id"]].append(record)
    if not by_case:
        return None
    perfect = 0
    for case_records in by_case.values():
        labelled = _labelled(case_records)
        if labelled and all(_valid_correct(r) for r in labelled):
            perfect += 1
    return perfect / len(by_case)


def macro_by(records: list[dict], key: str) -> dict[str, float]:
    """Macro (per-group unweighted mean) field accuracy grouped by ``key``.

    Records whose ``key`` is None are skipped (a null workflow is no group).
    """
    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        if record.get(key) is None:
            continue
        grouped[str(record[key])].append(record)
    macro = {}
    for group, group_records in grouped.items():
        accuracy = field_accuracy(group_records)
        if accuracy is not None:
            macro[group] = accuracy
    return dict(sorted(macro.items()))


def majority_class_baseline(records: list[dict]) -> dict[str, float]:
    """Per-field accuracy of always predicting the modal labelled value."""
    by_field: dict[str, list[dict]] = defaultdict(list)
    for record in _labelled(records):
        by_field[record["field"]].append(record)
    baseline: dict[str, float] = {}
    for field, field_records in by_field.items():
        counts = Counter(
            json.dumps(r["label"], sort_keys=True)
            if isinstance(r["label"], (list, dict))
            else r["label"]
            for r in field_records
        )
        baseline[field] = counts.most_common(1)[0][1] / len(field_records)
    return dict(sorted(baseline.items()))


# ------------------------------------------------------------- probabilistic


def brier_score(records: list[dict]) -> float | None:
    """Multiclass Brier: mean over fields of sum over choices of (p - y)^2."""
    scores = []
    for record in records:
        distribution = _record_distribution(record)
        if distribution is None:
            continue
        label = record.get("label")
        scores.append(
            sum(
                (prob - (1.0 if choice == label else 0.0)) ** 2
                for choice, prob in distribution.items()
            )
        )
    return _mean(scores)


def log_loss(records: list[dict]) -> float | None:
    """Multiclass true-label log loss (natural log)."""
    losses = []
    for record in records:
        distribution = _record_distribution(record)
        if distribution is None:
            continue
        label = record.get("label")
        prob = distribution.get(label)
        if prob is None or not _finite(prob) or prob <= 0:
            return None  # unmeasurable distribution; never paper over it
        losses.append(-math.log(prob))
    return _mean(losses)


def _record_distribution(record: dict) -> dict[str, float] | None:
    """Best available choice distribution: log_scores softmaxed at T=1."""
    log_scores = record.get("log_scores")
    if not isinstance(log_scores, dict) or not log_scores:
        return None
    if not all(_finite(v) for v in log_scores.values()):
        return None
    peak = max(log_scores.values())
    exp = {choice: math.exp(score - peak) for choice, score in log_scores.items()}
    total = sum(exp.values())
    if total <= 0:
        return None
    return {choice: value / total for choice, value in exp.items()}


def correctness_auroc(records: list[dict]) -> float | None:
    """AUROC of confidence predicting correctness (rank-statistic, no numpy).

    Labelled records only; pairs with equal confidence count 0.5. Returns
    None without both classes.
    """
    scored = [
        (float(r["probability"]), _valid_correct(r))
        for r in _labelled(records)
        if r.get("probability") is not None and _finite(r["probability"])
    ]
    positive = sum(1 for _, correct in scored if correct)
    negative = len(scored) - positive
    if not positive or not negative:
        return None
    wins = 0.0
    for confidence_a, correct_a in scored:
        for confidence_b, correct_b in scored:
            if not correct_a or correct_b:
                continue  # only positive-vs-negative pairs contribute
            if confidence_a > confidence_b:
                wins += 1.0
            elif confidence_a == confidence_b:
                wins += 0.5
    return wins / (positive * negative)


def ece_equal_mass(records: list[dict], bins: int = 5) -> float | None:
    """Descriptive ECE over ``bins`` equal-mass confidence bins.

    Empty-sample safety: returns None when there is nothing to measure.
    """
    scored = [
        (float(r["probability"]), 1.0 if _valid_correct(r) else 0.0)
        for r in _labelled(records)
        if r.get("probability") is not None and _finite(r["probability"])
    ]
    if not scored:
        return None
    scored.sort()
    total = len(scored)
    bin_size = math.ceil(total / bins)
    ece = 0.0
    for start in range(0, total, bin_size):
        chunk = scored[start : start + bin_size]
        mean_confidence = sum(c for c, _ in chunk) / len(chunk)
        mean_correct = sum(y for _, y in chunk) / len(chunk)
        ece += (len(chunk) / total) * abs(mean_confidence - mean_correct)
    return ece


# ------------------------------------------------------------------ set fields


def multi_jaccard(records: list[dict]) -> float | None:
    """Mean Jaccard similarity of multi predictions (invalid => 0.0)."""
    scores = []
    for record in records:
        if record.get("type") != "multi":
            continue
        if record.get("label") is None:
            continue
        predicted = record.get("prediction")
        if not record.get("valid") or not isinstance(predicted, list):
            scores.append(0.0)
            continue
        predicted_set, label_set = set(predicted), set(record["label"])
        union = predicted_set | label_set
        scores.append(len(predicted_set & label_set) / len(union) if union else 1.0)
    return _mean(scores)


# ------------------------------------------------- balanced accuracy / macro-F1


def _prediction_label_pair(record: dict) -> tuple[str, str] | None:
    """(label, prediction) as comparable strings, or None when not scoreable.

    Lists (multi fields) are compared as sorted JSON — order-insensitive, so
    ``["a", "b"]`` and ``["b", "a"]`` are the same value. Non-str labels are
    stringified to a stable JSON form.
    """
    label = record.get("label")
    prediction = record.get("prediction")
    if label is None:
        return None
    if not record.get("valid", prediction is not None) or prediction is None:
        prediction = "<invalid>"
    if isinstance(label, list | dict):
        label = json.dumps(label, sort_keys=True)
    if isinstance(prediction, list | dict):
        prediction = json.dumps(prediction, sort_keys=True)
    return str(label), str(prediction)


def balanced_accuracy(records: list[dict]) -> dict[str, float | None]:
    """Per field: mean recall over classes (balanced accuracy).

    Recall is computed per class among that class's labelled records; classes
    with no labelled records are skipped, and a field whose every class lacks
    support yields None. Predictions are counted for the predicted class
    (invalid predictions never contribute to any class's recall).
    """
    by_field: dict[str, Counter] = defaultdict(Counter)
    support: dict[str, Counter] = defaultdict(Counter)
    for record in records:
        pair = _prediction_label_pair(record)
        if pair is None:
            continue
        label, prediction = pair
        by_field[record["field"]][label] += 0  # register the class
        support[record["field"]][label] += 1
        if prediction == label:
            by_field[record["field"]][label] += 1
    result: dict[str, float | None] = {}
    for field in sorted(by_field):
        recalls = [
            by_field[field][cls] / support[field][cls]
            for cls in by_field[field]
            if support[field][cls] > 0
        ]
        result[field] = _mean(recalls)
    return result


def macro_f1(records: list[dict]) -> dict[str, float | None]:
    """Per field: macro-averaged F1 across the label classes present.

    For each class: precision = TP / predicted, recall = TP / labelled, F1 =
    2PR/(P+R) (0 when both are 0). The class mean ignores classes with no
    labelled records; a field with none yields None.
    """
    tp: dict[str, Counter] = defaultdict(Counter)
    predicted: dict[str, Counter] = defaultdict(Counter)
    labelled_n: dict[str, Counter] = defaultdict(Counter)
    for record in records:
        pair = _prediction_label_pair(record)
        if pair is None:
            continue
        label, prediction = pair
        field = record["field"]
        labelled_n[field][label] += 1
        if prediction == label:
            tp[field][label] += 1
        elif record.get("valid", True) and prediction is not None:
            predicted[field][prediction] += 1
    result: dict[str, float | None] = {}
    for field in sorted(labelled_n):
        f1s = []
        for cls in labelled_n[field]:
            recall = tp[field][cls] / labelled_n[field][cls]
            precision_den = tp[field][cls] + predicted[field][cls]
            precision = tp[field][cls] / precision_den if precision_den else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            f1s.append(f1)
        result[field] = _mean(f1s)
    return result


# ------------------------------------------------------- perturbation flips


def _canonical_form(value) -> str:
    """Comparison form of a prediction: lists order-insensitive, dicts stable.

    Multi predictions are lists; ["a", "b"] and ["b", "a"] are the same
    value, so list elements are sorted before serialising.
    """
    if isinstance(value, list):
        return json.dumps(sorted(value), sort_keys=True)
    return json.dumps(value, sort_keys=True)


def perturbation_flip_rate(records: list[dict]) -> float | None:
    """Share of (original, perturbed) pairs whose prediction differs.

    A pair is (group_id, field, case_id) where one line carries a
    ``perturbation`` (the variant) and the other does not (the original).
    Only canonical lines count: ``permutation`` must be ``"canonical"`` or
    missing, so choice-rotation rows never pollute the pairs. Requires
    meta.perturbation to have flowed through to the line (see evalrun).
    Returns None when no such pair exists.
    """
    originals: dict[tuple[str, str], object] = {}
    variants: dict[tuple[str, str], list[tuple[str, object]]] = defaultdict(list)
    for record in records:
        if record.get("permutation") not in (None, "canonical"):
            continue  # rotation/fieldperm rows are order probes, not perturbations
        kind = record.get("perturbation") or (record.get("meta") or {}).get("perturbation")
        key = (record.get("group_id") or record["case_id"], record["field"])
        if kind is None:
            originals.setdefault(key, record.get("prediction"))
        else:
            variants[key].append((record["case_id"], record.get("prediction")))
    pairs = 0
    flipped = 0
    for key, variant_list in variants.items():
        if key not in originals:
            continue
        base = _canonical_form(originals[key])
        for _case_id, prediction in variant_list:
            pairs += 1
            if _canonical_form(prediction) != base:
                flipped += 1
    if not pairs:
        return None
    return flipped / pairs


# -------------------------------------------------------- position-bias (permutation)


def any_flip_rate(records: list[dict]) -> dict[str, float]:
    """Per field: share of permuted runs whose chosen value differs from canonical."""
    canonical: dict[tuple[str, str], object] = {}
    permuted: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        key = (record["case_id"], record["field"])
        if (record.get("permutation") or "canonical") == "canonical":
            canonical[key] = record.get("prediction")
        else:
            permuted[record["field"]].append(record)
    flips: dict[str, float] = {}
    for field, field_records in permuted.items():
        comparable = [r for r in field_records if (r["case_id"], field) in canonical]
        if not comparable:
            continue
        flipped = sum(
            1
            for r in comparable
            if json.dumps(r.get("prediction"), sort_keys=True)
            != json.dumps(canonical[(r["case_id"], field)], sort_keys=True)
        )
        flips[field] = flipped / len(comparable)
    return dict(sorted(flips.items()))


def mean_tvd_across_permutations(records: list[dict]) -> dict[str, float]:
    """Per field: mean TVD between each permutation's distribution and canonical's.

    Uses the softmaxed log_scores distribution; requires a canonical record
    per (case, field). Skips permutations lacking one.
    """
    canonical: dict[tuple[str, str], dict] = {}
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for record in records:
        if record.get("log_scores") is None:
            continue
        key = (record["case_id"], record["field"])
        if record.get("permutation") == "canonical":
            canonical[key] = record
        else:
            grouped[key].append(record)
    tvds: dict[str, list[float]] = defaultdict(list)
    for key, permuted_records in grouped.items():
        base = canonical.get(key)
        if base is None:
            continue
        base_distribution = _record_distribution(base)
        if base_distribution is None:
            continue
        for record in permuted_records:
            distribution = _record_distribution(record)
            if distribution is None:
                continue
            tvds[record["field"]].append(_tvd(base_distribution, distribution))
    return {field: _mean(values) for field, values in sorted(tvds.items()) if values}


def tie_rate(records: list[dict]) -> float | None:
    """Share of scored records whose top-two log_scores are exactly equal."""
    scored = [r for r in records if isinstance(r.get("log_scores"), dict) and r["log_scores"]]
    if not scored:
        return None
    ties = 0
    for record in scored:
        values = sorted(record["log_scores"].values(), reverse=True)
        if len(values) >= 2 and values[0] == values[1]:
            ties += 1
    return ties / len(scored)


# -------------------------------------------------------------- risk-coverage


def risk_coverage_curve(records: list[dict], points: int = 10) -> list[dict]:
    """Risk (1 - accuracy) at ``points`` evenly spaced coverage levels.

    Records are sorted by confidence descending (tie-break: stable order);
    at each coverage the prefix risk is computed. Uncovered records count as
    neither right nor wrong — they are simply not yet included.
    """
    scored = sorted(
        (
            (float(r["probability"]), 1.0 if _valid_correct(r) else 0.0)
            for r in _labelled(records)
            if r.get("probability") is not None and _finite(r["probability"])
        ),
        key=lambda pair: -pair[0],
    )
    total = len(scored)
    if total == 0:
        return []
    curve = []
    for i in range(1, points + 1):
        coverage = i / points
        take = max(1, min(total, math.ceil(coverage * total)))
        prefix = scored[:take]
        accuracy = sum(y for _, y in prefix) / len(prefix)
        curve.append({"coverage": round(coverage, 2), "risk": 1.0 - accuracy, "n": take})
    return curve


# ------------------------------------------------------------------ bootstrap


def accuracy_cluster_bootstrap(
    records: list[dict],
    draws: int = 1000,
    seed: int = 0,
    confidence: float = 0.95,
) -> dict | None:
    """Case-level cluster bootstrap CI for field accuracy.

    Resamples whole cases (all their fields move together) ``draws`` times and
    returns point estimate plus the central percentile interval.
    """
    by_case: dict[str, list[dict]] = defaultdict(list)
    for record in _labelled(records):
        by_case[record["case_id"]].append(record)
    case_ids = sorted(by_case)
    if not case_ids:
        return None
    point = field_accuracy(records)
    rng = random.Random(seed)
    stats = []
    for _ in range(draws):
        sample = [r for case in rng.choices(case_ids, k=len(case_ids)) for r in by_case[case]]
        stats.append(field_accuracy(sample))
    stats.sort()
    alpha = (1.0 - confidence) / 2
    lower_index = math.floor(alpha * (draws - 1))
    upper_index = math.ceil((1 - alpha) * (draws - 1))
    return {
        "accuracy": point,
        "ci_low": stats[lower_index],
        "ci_high": stats[upper_index],
        "n_cases": len(case_ids),
        "n_fields": sum(len(v) for v in by_case.values()),
        "draws": draws,
        "seed": seed,
    }


# ------------------------------------------------------------------- assembly


# ------------------------------------------------- TypeSafe-comparable metrics


def _typesafe_lines(records: list[dict]) -> list[dict]:
    """Records from TypeSafe-derived cases (source == 'typesafe')."""
    return [r for r in records if r.get("source") == "typesafe"]


def _is_ambiguous(record: dict) -> bool:
    """Whether the line's field was flagged ambiguous by the fetcher.

    The line-level ``ambiguous`` key (a list of field names, carried from
    meta when requested) or a per-line ``field_ambiguous`` flag both count.
    """
    field = record.get("field")
    flag = record.get("field_ambiguous")
    if flag is not None:
        return bool(flag)
    ambiguous = record.get("ambiguous")
    if isinstance(ambiguous, list):
        return field in ambiguous
    return False


def typesafe_agreement(records: list[dict]) -> dict:
    """Agreement with TypeSafe's published consensus (their headline metric).

    For records whose case source is ``typesafe``: the share of labelled
    fields whose prediction equals the consensus label — numerically the
    accuracy restricted to that source, reported under TypeSafe's name.
    Returns::

        {"overall": float, "by_workflow": {workflow: float},
         "agreement_common_subset": float, "n_fields": int,
         "n_cases": int}

    ``agreement_common_subset`` is computed only on fields that are not
    flagged ambiguous (meta.ambiguous from the fetcher), the subset closest
    to TypeSafe's own presentation; it is None when every field is flagged.
    Records with no label are excluded from all rates; ``n_cases`` counts
    distinct case_ids among the labelled lines.
    """
    lines = [r for r in _typesafe_lines(records) if _labelled([r])]
    if not lines:
        return {}
    by_workflow: dict[str, list[dict]] = defaultdict(list)
    common: list[dict] = []
    for record in lines:
        by_workflow[str(record.get("workflow"))].append(record)
        if not _is_ambiguous(record):
            common.append(record)
    overall = field_accuracy(lines)
    result = {
        "overall": overall,
        "by_workflow": macro_by(lines, "workflow"),
        "agreement_common_subset": field_accuracy(common),
        "n_fields": len(lines),
        "n_cases": len({record.get("case_id") for record in lines}),
    }
    return result


def tvd_vs_consensus(records: list[dict]) -> dict:
    """Mean TVD between our choice distribution and the consensus one.

    Our distribution is the same one the calibration metrics use (softmax
    over ``log_scores`` at T=1, or ``per_option`` for multi); the reference
    is the fetcher's ``consensus`` distribution carried on the line. Lines
    without either side are skipped. Returns ``{"overall": float,
    "by_workflow": {...}}`` (both absent when no line qualifies).
    """
    scored: list[tuple[str, float]] = []
    for record in _typesafe_lines(records):
        reference = record.get("consensus")
        if not isinstance(reference, dict) or not reference:
            continue
        ours = _record_distribution(record)
        if ours is None:
            continue
        workflow = str(record.get("workflow"))
        scored.append((workflow, _tvd(ours, reference)))
    if not scored:
        return {}
    by_workflow: dict[str, list[float]] = defaultdict(list)
    for workflow, value in scored:
        by_workflow[workflow].append(value)
    result = {
        "overall": sum(value for _w, value in scored) / len(scored),
        "by_workflow": {
            workflow: sum(values) / len(values) for workflow, values in sorted(by_workflow.items())
        },
    }
    return result


# ----------------------------------------------------- dependent-schema metrics


def _case_predictions(records: list[dict]) -> dict[str, dict[str, object]]:
    """Group records by case_id -> {field: prediction}.

    Only fields with a label are kept (constraints only apply to labelled
    fields). Multi predictions are normalized to a list (or empty list).
    """
    by_case: dict[str, dict[str, object]] = defaultdict(dict)
    for r in records:
        case_id = r.get("case_id")
        if case_id is None or r.get("label") is None:
            continue
        pred = r.get("prediction")
        if r.get("type") == "multi":
            if isinstance(pred, list):
                pass
            elif isinstance(pred, str):
                pred = [pred] if pred else []
            else:
                pred = []
        by_case[case_id][r["field"]] = pred
    return by_case


def _case_labels(records: list[dict]) -> dict[str, dict[str, object]]:
    """Group records by case_id -> {field: label}."""
    by_case: dict[str, dict[str, object]] = defaultdict(dict)
    for r in records:
        case_id = r.get("case_id")
        if case_id is None or r.get("label") is None:
            continue
        by_case[case_id][r["field"]] = r["label"]
    return by_case


def _case_constraints(records: list[dict]) -> dict[str, list[dict]]:
    """Group records by case_id -> constraints list (from the first record
    that carries them)."""
    by_case: dict[str, list[dict]] = {}
    for r in records:
        case_id = r.get("case_id")
        if case_id is None:
            continue
        constraints = r.get("constraints")
        if isinstance(constraints, list) and constraints and case_id not in by_case:
            by_case[case_id] = constraints
    return by_case


def _check_constraint(constraint: dict, preds: dict[str, object]) -> bool:
    """Return True if the constraint is SATISFIED, False if VIOLATED."""
    ctype = constraint.get("type")
    if ctype == "implies":
        parent = constraint.get("parent")
        child = constraint.get("child")
        mapping = constraint.get("mapping", {})
        parent_val = preds.get(parent)
        child_val = preds.get(child)
        if parent_val is None or child_val is None:
            return True  # unmeasured field can't violate
        allowed = mapping.get(parent_val, [])
        return child_val in allowed
    if ctype == "excludes":
        field = constraint.get("field")
        value = constraint.get("value")
        other = constraint.get("other")
        field_val = preds.get(field)
        other_val = preds.get(other)
        if field_val is None or other_val is None:
            return True
        if field_val == value:
            # When field==value, other must be empty/falsy
            if isinstance(other_val, list):
                return len(other_val) == 0
            return other_val in (None, "", False)
        return True
    if ctype == "requires_parent":
        parent = constraint.get("parent")
        child = constraint.get("child")
        mapping = constraint.get("mapping", {})
        parent_val = preds.get(parent)
        child_val = preds.get(child)
        if parent_val is None or child_val is None:
            return True
        allowed = mapping.get(parent_val, [])
        return child_val in allowed
    if ctype == "exclusivity":
        field = constraint.get("field")
        options = set(constraint.get("options", []))
        field_val = preds.get(field)
        if field_val is None:
            return True
        if not isinstance(field_val, list):
            field_val = [field_val] if field_val else []
        selected = set(field_val) & options
        return len(selected) <= 1  # at most one from the exclusivity group
    return True  # unknown constraint type: assume satisfied


def constraint_violation_rate(records: list[dict]) -> dict | None:
    """Fraction of cases with ≥1 constraint violation, and per-constraint-type
    breakdown. Returns None when no case carries constraints."""
    preds = _case_predictions(records)
    constraints_by_case = _case_constraints(records)
    if not constraints_by_case:
        return None
    n_cases = len(constraints_by_case)
    violated = 0
    by_type: dict[str, int] = defaultdict(int)
    total_by_type: dict[str, int] = defaultdict(int)
    for case_id, constraints in constraints_by_case.items():
        case_preds = preds.get(case_id, {})
        case_violated = False
        for c in constraints:
            ctype = c.get("type", "unknown")
            total_by_type[ctype] += 1
            if not _check_constraint(c, case_preds):
                by_type[ctype] += 1
                case_violated = True
        if case_violated:
            violated += 1
    result = {"overall": violated / n_cases if n_cases else 0.0}
    if total_by_type:
        result["by_type"] = {
            t: by_type[t] / total_by_type[t] if total_by_type[t] else 0.0
            for t in sorted(total_by_type)
        }
    return result


def child_accuracy_given_parent_correct(records: list[dict]) -> dict | None:
    """For each implies/requires_parent constraint, child field accuracy
    among cases where the parent was predicted correctly. Returns None when
    no such constraints exist."""
    preds = _case_predictions(records)
    labels = _case_labels(records)
    constraints_by_case = _case_constraints(records)
    if not constraints_by_case:
        return None
    # Collect (parent, child) pairs from constraints.
    pairs: dict[tuple[str, str], list[bool]] = defaultdict(list)
    for case_id, constraints in constraints_by_case.items():
        case_preds = preds.get(case_id, {})
        case_labels = labels.get(case_id, {})
        for c in constraints:
            if c.get("type") not in ("implies", "requires_parent"):
                continue
            parent = c.get("parent")
            child = c.get("child")
            parent_pred = case_preds.get(parent)
            parent_label = case_labels.get(parent)
            child_pred = case_preds.get(child)
            child_label = case_labels.get(child)
            if parent_pred is None or parent_label is None:
                continue
            if parent_pred != parent_label:
                continue  # parent wrong: skip
            if child_pred is None or child_label is None:
                continue
            pairs[(parent, child)].append(child_pred == child_label)
    if not pairs:
        return None
    return {
        f"{parent}→{child}": sum(v) / len(v) if v else 0.0
        for (parent, child), v in sorted(pairs.items())
    }


def exact_record_accuracy(records: list[dict]) -> float | None:
    """Fraction of cases where EVERY labelled field is correct AND no
    constraint is violated. Stricter than case_exact_match (which ignores
    constraints). Returns None when no cases carry labels."""
    by_case: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        if r.get("label") is not None or r.get("correct") is not None:
            by_case[r["case_id"]].append(r)
    if not by_case:
        return None
    preds = _case_predictions(records)
    constraints_by_case = _case_constraints(records)
    perfect = 0
    for case_id, case_records in by_case.items():
        labelled = _labelled(case_records)
        if not labelled or not all(_valid_correct(r) for r in labelled):
            continue
        # If this case has constraints, they must all be satisfied too.
        constraints = constraints_by_case.get(case_id, [])
        if constraints:
            case_preds = preds.get(case_id, {})
            if not all(_check_constraint(c, case_preds) for c in constraints):
                continue
        perfect += 1
    return perfect / len(by_case)


def order_flip_rate(records: list[dict]) -> dict | None:
    """Fraction of cases where the prediction for a field flips when field
    order changes (permutation != 'canonical'). Reuses any_flip_rate when
    permutation data is present; returns None otherwise."""
    existing = any_flip_rate(records)
    if existing:
        return existing
    return None


def compute_metrics(records: list[dict]) -> dict:
    """Assemble the metrics dict for evalreport.write_report.

    Missing-data safety: a metric that cannot be computed (no labelled rows,
    no confidences, no log_scores) is simply absent rather than zero-filled.
    """
    metrics: dict = {}
    accuracy = field_accuracy(records)
    if accuracy is not None:
        metrics["accuracy"] = accuracy
    exact = case_exact_match(records)
    if exact is not None:
        metrics["case_exact_match"] = exact
    majority = majority_class_baseline(records)
    if majority:
        metrics["majority_class_baseline"] = majority
    jaccard = multi_jaccard(records)
    if jaccard is not None:
        metrics["multi_jaccard"] = jaccard
    brier = brier_score(records)
    if brier is not None:
        metrics["brier"] = brier
    nll = log_loss(records)
    if nll is not None:
        metrics["log_loss"] = nll
    auroc = correctness_auroc(records)
    if auroc is not None:
        metrics["correctness_auroc"] = auroc
    ece = ece_equal_mass(records, bins=5)
    if ece is not None:
        metrics["ece_5bin_equal_mass"] = ece
    ties = tie_rate(records)
    if ties is not None:
        metrics["tie_rate"] = ties
    curve = risk_coverage_curve(records, points=10)
    if curve:
        metrics["risk_coverage"] = curve
    flips = any_flip_rate(records)
    if flips:
        metrics["any_flip_rate"] = flips
    tvds = mean_tvd_across_permutations(records)
    if tvds:
        metrics["mean_tvd"] = tvds
    bootstrap = accuracy_cluster_bootstrap(records)
    if bootstrap:
        metrics["accuracy_cluster_bootstrap"] = bootstrap
    balanced = balanced_accuracy(records)
    if any(value is not None for value in balanced.values()):
        metrics["balanced_accuracy"] = balanced
    f1 = macro_f1(records)
    if any(value is not None for value in f1.values()):
        metrics["macro_f1"] = f1
    flips = perturbation_flip_rate(records)
    if flips is not None:
        metrics["perturbation_flip_rate"] = flips
    agreement = typesafe_agreement(records)
    if agreement:
        metrics["agreement"] = agreement
    tvd = tvd_vs_consensus(records)
    if tvd:
        metrics["tvd_vs_consensus"] = tvd
    # Dependent-schema metrics (EV1: review Q1 'The correct diagnostic').
    exact_record = exact_record_accuracy(records)
    if exact_record is not None:
        metrics["exact_record_accuracy"] = exact_record
    violations = constraint_violation_rate(records)
    if violations:
        metrics["constraint_violation_rate"] = violations
    child_acc = child_accuracy_given_parent_correct(records)
    if child_acc:
        metrics["child_accuracy_given_parent_correct"] = child_acc
    # oracle_parent_gap: needs a conditioned rerun (W3-D); None until then.
    flips = order_flip_rate(records)
    if flips:
        metrics["order_flip_rate"] = flips
    return metrics
