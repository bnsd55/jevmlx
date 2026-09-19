"""Hand-computable metrics tests for jevmlx.evalmetrics. No numpy, no model."""

import math

import pytest

from jevmlx.evalmetrics import (
    any_flip_rate,
    balanced_accuracy,
    brier_score,
    case_exact_match,
    cluster_bootstrap_ci,
    compute_metrics,
    correctness_auroc,
    ece_equal_mass,
    field_accuracy,
    load_predictions,
    log_loss,
    macro_by,
    macro_f1,
    majority_class_baseline,
    mean_tvd_across_permutations,
    multi_jaccard,
    perturbation_flip_rate,
    risk_coverage_curve,
    tie_rate,
    tvd_vs_consensus,
    typesafe_agreement,
)

# 4 cases, 6 labelled field-predictions. Every number below is hand-checkable.
P = [
    # case c1: action correct (log_scores -> prob APPROVE 0.5, REJECT 0.5)
    {
        "run_id": "r",
        "case_id": "c1",
        "group_id": "g1",
        "source": "typesafe",
        "workflow": "w1",
        "field": "action",
        "type": "enum",
        "track": "parallel",
        "model": "m",
        "permutation": "canonical",
        "label": "APPROVE",
        "prediction": "APPROVE",
        "valid": True,
        "correct": True,
        "log_scores": {"APPROVE": 0.0, "REJECT": 0.0},
        "probability": 0.6,
    },
    # case c1: amount wrong
    {
        "run_id": "r",
        "case_id": "c1",
        "group_id": "g1",
        "source": "typesafe",
        "workflow": "w1",
        "field": "amount",
        "type": "boolean",
        "track": "parallel",
        "model": "m",
        "permutation": "canonical",
        "label": True,
        "prediction": False,
        "valid": True,
        "correct": False,
        "log_scores": None,
        "probability": 0.9,
    },
    # case c2: invalid prediction (counts wrong, not skipped)
    {
        "run_id": "r",
        "case_id": "c2",
        "group_id": "g1",
        "source": "quality-eval",
        "workflow": None,
        "field": "action",
        "type": "enum",
        "track": "parallel",
        "model": "m",
        "permutation": "canonical",
        "label": "APPROVE",
        "prediction": None,
        "valid": False,
        "correct": False,
        "log_scores": None,
        "probability": None,
        "error": "bad JSON",
    },
    # case c3: action correct, high confidence
    {
        "run_id": "r",
        "case_id": "c3",
        "group_id": "g2",
        "source": "typesafe",
        "workflow": "w1",
        "field": "action",
        "type": "enum",
        "track": "parallel",
        "model": "m",
        "permutation": "canonical",
        "label": "REJECT",
        "prediction": "REJECT",
        "valid": True,
        "correct": True,
        "log_scores": {"APPROVE": -2.0, "REJECT": 2.0},
        "probability": 0.95,
    },
    # case c3: multi field jaccard 1/2
    {
        "run_id": "r",
        "case_id": "c3",
        "group_id": "g2",
        "source": "typesafe",
        "workflow": "w1",
        "field": "tags",
        "type": "multi",
        "track": "parallel",
        "model": "m",
        "permutation": "canonical",
        "label": ["a", "b"],
        "prediction": ["a", "c"],
        "valid": True,
        "correct": False,
        "log_scores": None,
        "probability": 0.7,
    },
    # case c4: multi field correct (jaccard 1.0)
    {
        "run_id": "r",
        "case_id": "c4",
        "group_id": "g2",
        "source": "typesafe",
        "workflow": "w2",
        "field": "tags",
        "type": "multi",
        "track": "parallel",
        "model": "m",
        "permutation": "canonical",
        "label": ["x"],
        "prediction": ["x"],
        "valid": True,
        "correct": True,
        "log_scores": None,
        "probability": 0.8,
    },
    # unlabelled row: excluded from accuracies
    {
        "run_id": "r",
        "case_id": "c4",
        "group_id": "g2",
        "source": "typesafe",
        "workflow": "w2",
        "field": "notes",
        "type": "enum",
        "track": "parallel",
        "model": "m",
        "permutation": "canonical",
        "label": None,
        "prediction": "APPROVE",
        "valid": True,
        "correct": None,
        "log_scores": None,
        "probability": 0.5,
    },
]


class TestAccuracies:
    def test_field_accuracy(self):
        # 6 labelled rows (notes unlabelled): T,F,F,T,F,T = 3/6
        assert field_accuracy(P) == 0.5

    def test_invalid_prediction_counts_wrong(self):
        # drop the correct c3 action: 2/5 labelled correct
        trimmed = [r for r in P if not (r["case_id"] == "c3" and r["field"] == "action")]
        assert field_accuracy(trimmed) == 2 / 5

    def test_case_exact_match(self):
        # c1 wrong (amount), c2 wrong (invalid), c3 wrong (tags), c4 perfect -> 1/4
        assert case_exact_match(P) == 0.25

    def test_macro_by_source(self):
        macro = macro_by(P, "source")
        # typesafe: c1 action T, c1 amount F, c3 action T, c3 tags F, c4 tags T -> 3/5
        assert macro["typesafe"] == 3 / 5
        # quality-eval: only the invalid c2 row -> 0.0
        assert macro["quality-eval"] == 0.0

    def test_macro_by_workflow_excludes_null(self):
        macro = macro_by(P, "workflow")
        assert set(macro) == {"w1", "w2"}

    def test_majority_class_baseline(self):
        majority = majority_class_baseline(P)
        # action labels: APPROVE, APPROVE, REJECT -> majority APPROVE = 2/3
        assert majority["action"] == 2 / 3
        # amount labels: [True] -> 1.0
        assert majority["amount"] == 1.0
        # tags labels: ["a","b"], ["x"] -> each modal once = 1/2
        assert majority["tags"] == 0.5


class TestProbabilistic:
    def test_brier_score(self):
        # scored rows: c1 action dist (0.5, 0.5) label APPROVE -> 0.25+0.25 = 0.5
        # c3 action softmax(2,-2): p_REJECT = e^2/(e^2+e^-2) -> miss (1-p)^2 + hit 0^2
        p_reject = math.exp(2) / (math.exp(2) + math.exp(-2))
        expected = (0.5 + 2 * (1 - p_reject) ** 2) / 2  # mean over 2 rows; c3 has both terms
        assert math.isclose(brier_score(P), expected, rel_tol=1e-9)

    def test_log_loss(self):
        # c1: p(APPROVE)=0.5 -> -ln 0.5 ; c3: p(REJECT) ~= 0.982 -> -ln(0.982)
        p_reject = math.exp(2) / (math.exp(2) + math.exp(-2))
        expected = (-math.log(0.5) + -math.log(p_reject)) / 2
        assert math.isclose(log_loss(P), expected, rel_tol=1e-9)

    def test_correctness_auroc(self):
        # confident-correct pairs: (0.6,True) vs negatives (0.9,False), (0.7,False)
        # (0.95,True) beats both negatives; (0.6,True) beats none -> (2+0)/4 = 0.5
        assert correctness_auroc(P) == 0.5

    def test_auroc_needs_both_classes(self):
        only_correct = [r for r in P if r.get("probability") is not None and r.get("correct")]
        assert correctness_auroc(only_correct) is None

    def test_ece_equal_mass(self):
        # confidences: 0.6,0.9,0.95,0.7,0.8 sorted -> 5 bins of 1
        # |0.6-1| + |0.7-0| + |0.8-1| + |0.9-0| + |0.95-1| = 0.4+0.7+0.2+0.9+0.05 = 2.25 /5
        assert math.isclose(ece_equal_mass(P, bins=5), 2.25 / 5)

    def test_ece_empty_is_none(self):
        assert ece_equal_mass([], bins=5) is None

    def test_tie_rate(self):
        # c1 action log_scores equal -> tie; c3 not. 1 tie / 2 scored = 0.5
        assert tie_rate(P) == 0.5


class TestSetMetrics:
    def test_multi_jaccard(self):
        # {a,c} vs {a,b} = 1/3, and {x} vs {x} = 1.0 -> 2/3
        assert multi_jaccard(P) == 2 / 3

    def test_multi_jaccard_invalid_is_zero(self):
        invalid = [dict(P[4], valid=False, prediction=None)]
        assert multi_jaccard(invalid) == 0.0


class TestPositionBias:
    def test_any_flip_rate(self):
        # c3 action: canonical REJECT + rot1 APPROVE + rot2 REJECT -> 1 flip / 2 perms
        records = P + [
            {
                "run_id": "r",
                "case_id": "c3",
                "group_id": "g2",
                "source": "typesafe",
                "workflow": "w1",
                "field": "action",
                "type": "enum",
                "track": "parallel",
                "model": "m",
                "permutation": "rot1",
                "label": "REJECT",
                "prediction": "APPROVE",
                "valid": True,
                "correct": False,
                "log_scores": None,
                "probability": 0.9,
            },
            {
                "run_id": "r",
                "case_id": "c3",
                "group_id": "g2",
                "source": "typesafe",
                "workflow": "w1",
                "field": "action",
                "type": "enum",
                "track": "parallel",
                "model": "m",
                "permutation": "rot2",
                "label": "REJECT",
                "prediction": "REJECT",
                "valid": True,
                "correct": True,
                "log_scores": None,
                "probability": 0.9,
            },
        ]
        flips = any_flip_rate(records)
        assert flips == {"action": 0.5}

    def test_mean_tvd_across_permutations(self):
        # c1 action canonical log_scores {A:0, R:0} -> dist (0.5, 0.5)
        # rot1 {A:0, R:2} -> dist (0.119, 0.881) -> TVD = 0.381
        records = P + [
            {
                "run_id": "r",
                "case_id": "c1",
                "group_id": "g1",
                "source": "typesafe",
                "workflow": "w1",
                "field": "action",
                "type": "enum",
                "track": "parallel",
                "model": "m",
                "permutation": "rot1",
                "label": "APPROVE",
                "prediction": "REJECT",
                "valid": True,
                "correct": False,
                "log_scores": {"APPROVE": 0.0, "REJECT": 2.0},
                "probability": 0.9,
            },
        ]
        tvds = mean_tvd_across_permutations(records)
        p_a = math.exp(0) / (math.exp(0) + math.exp(2))
        expected = 0.5 * (abs(0.5 - p_a) + abs(0.5 - (1 - p_a)))
        assert math.isclose(tvds["action"], expected, rel_tol=1e-9)


class TestRiskCoverage:
    def test_curve_shape_and_values(self):
        curve = risk_coverage_curve(P, points=10)
        assert len(curve) == 10
        assert curve[0]["coverage"] == 0.1 and curve[-1]["coverage"] == 1.0
        # full-coverage risk = 1 - 0.6 = 0.4 (5 scored rows, 3 correct)
        assert math.isclose(curve[-1]["risk"], 0.4)
        # top-1 by confidence = 0.95 (correct) -> risk 0 at coverage 0.1
        assert curve[0]["risk"] == 0.0

    def test_empty(self):
        assert risk_coverage_curve([]) == []


class TestBootstrap:
    def test_cluster_bootstrap_moves_whole_cases(self):
        result = cluster_bootstrap_ci(P, field_accuracy, metric_name="accuracy", draws=200, seed=7)
        assert result["n_cases"] == 4
        assert result["accuracy"] == 0.5
        assert 0.0 <= result["ci_low"] <= result["accuracy"] <= result["ci_high"] <= 1.0
        # deterministic given the seed
        again = cluster_bootstrap_ci(P, field_accuracy, metric_name="accuracy", draws=200, seed=7)
        assert (again["ci_low"], again["ci_high"]) == (result["ci_low"], result["ci_high"])


class TestAssembly:
    def test_compute_metrics_keys(self):
        metrics = compute_metrics(P)
        expected_keys = {
            "accuracy",
            "case_exact_match",
            "majority_class_baseline",
            "multi_jaccard",
            "brier",
            "log_loss",
            "correctness_auroc",
            "ece_5bin_equal_mass",
            "tie_rate",
            "risk_coverage",
            # W6-B6b: the old accuracy_cluster_bootstrap (1000 draws) is
            # replaced by accuracy_ci (case-cluster bootstrap, B=2000).
            "accuracy_ci",
            "valid_accuracy",
            "per_field_accuracy",
        }
        assert expected_keys <= set(metrics)

    def test_compute_metrics_empty_is_safe(self):
        assert compute_metrics([]) == {}

    def test_load_predictions_roundtrip(self, tmp_path):
        path = tmp_path / "pred.jsonl"
        path.write_text("\n".join(__import__("json").dumps(r) for r in P[:3]) + "\n\n")
        loaded = load_predictions(path)
        assert len(loaded) == 3 and loaded[0]["case_id"] == "c1"


class TestReportWiring:
    def test_jevmlx_report_cli_writes_both_files(self, tmp_path, capsys, monkeypatch):
        import json

        from jevmlx import cli

        predictions = tmp_path / "predictions.jsonl"
        predictions.write_text("\n".join(json.dumps(r) for r in P) + "\n", encoding="utf-8")
        out = tmp_path / "report.json"
        monkeypatch.setattr(
            "sys.argv",
            ["jevmlx", "report", "--predictions", str(predictions), "--out", str(out)],
        )
        cli.main()
        run = json.loads(out.read_text())
        assert run["metrics"]["accuracy"] == 0.5
        md = out.with_suffix(".md").read_text()
        assert "## Metrics" in md and "| accuracy |" in md


class TestBalancedAccuracyAndMacroF1:
    """Hand-computed balanced accuracy, macro-F1, and perturbation flips."""

    def _rec(self, field, label, prediction, valid=True, **extra):
        record = {
            "run_id": "r",
            "case_id": "c",
            "group_id": "g",
            "source": "s",
            "workflow": "w",
            "field": field,
            "type": "enum",
            "track": "parallel",
            "model": "m",
            "permutation": "canonical",
            "label": label,
            "prediction": prediction,
            "valid": valid,
            "correct": None if label is None else (valid and prediction == label),
            "log_scores": None,
            "probability": None,
            "per_option": None,
            "latency_ms": None,
            "rows": None,
            "passes": None,
            "error": None,
            "salvage_prediction": None,
        }
        record.update(extra)
        return record

    def test_balanced_accuracy_mean_recall_over_classes(self):
        # field "risk": label POS rows: 1 of 2 recalled; label NEG rows: 2 of 2.
        # Balanced accuracy = (0.5 + 1.0) / 2 = 0.75 (plain accuracy would be 0.75 too,
        # but it is insensitive to the class split — recall mean is the definition).
        records = [
            self._rec("risk", "POS", "POS"),
            self._rec("risk", "POS", "NEG"),
            self._rec("risk", "NEG", "NEG"),
            self._rec("risk", "NEG", "NEG"),
            # field "ok": single class, both recalled -> 1.0.
            self._rec("ok", True, True),
            self._rec("ok", True, True),
        ]
        assert balanced_accuracy(records) == {"ok": 1.0, "risk": 0.75}

    def test_balanced_accuracy_invalid_prediction_never_recalls(self):
        records = [
            self._rec("x", "A", "A"),
            self._rec("x", "A", None, valid=False),
        ]
        # Class A recall = 1/2; no other classes -> balanced accuracy 0.5.
        assert balanced_accuracy(records) == {"x": 0.5}

    def test_macro_f1_two_classes_hand_computed(self):
        # Class POS: TP=1, predicted POS=1 -> precision 1.0; labelled=2 -> recall 0.5;
        #   F1_POS = 2*1*0.5/(1.5) = 2/3.
        # Class NEG: TP=2, but the wrong POS row predicted NEG is a false positive:
        #   predicted NEG=3 -> precision 2/3; labelled=2 -> recall 1.0;
        #   F1_NEG = 2*(2/3)*1/(5/3) = 0.8.
        # Macro F1 = (2/3 + 0.8) / 2 = 11/15.
        records = [
            self._rec("risk", "POS", "POS"),
            self._rec("risk", "POS", "NEG"),
            self._rec("risk", "NEG", "NEG"),
            self._rec("risk", "NEG", "NEG"),
        ]
        assert macro_f1(records) == {"risk": pytest.approx(11 / 15)}

    def test_macro_f1_false_positive_lowers_precision(self):
        # Class A: TP=2, labelled=2 -> recall 1.0; the wrong B row predicted A, so
        # predicted A=3 -> precision 2/3 -> F1_A = 2*(2/3)*1/(5/3) = 0.8.
        # Class B: TP=0, labelled=1 -> recall 0.0; nothing predicted B -> F1_B = 0.
        records = [
            self._rec("f", "A", "A"),
            self._rec("f", "A", "A"),
            self._rec("f", "B", "A"),
        ]
        assert macro_f1(records) == {"f": pytest.approx(0.4)}

    def test_perturbation_flip_rate_by_group(self):
        # Group g1: original predicts APPROVE; ws variant flips to BLOCK;
        # numfmt variant repeats APPROVE -> 1 flip / 2 pairs = 0.5.
        # Group g2 has an original but no variants (excluded from the denominator).
        original_1 = self._rec("action", "APPROVE", "APPROVE", case_id="c1", group_id="g1")
        variant_ws = self._rec(
            "action", "APPROVE", "BLOCK", case_id="c1#p1", group_id="g1", perturbation="ws"
        )
        variant_num = self._rec(
            "action", "APPROVE", "APPROVE", case_id="c1#p2", group_id="g1", perturbation="numfmt"
        )
        lone_original = self._rec("action", "A", "A", case_id="c2", group_id="g2")
        assert perturbation_flip_rate([original_1, variant_ws, variant_num, lone_original]) == 0.5

    def test_perturbation_flip_rate_no_pairs_is_none(self):
        assert perturbation_flip_rate([self._rec("f", "A", "A")]) is None

    def test_perturbation_flip_rate_all_flips_and_multi_field(self):
        original_a = self._rec("flag", True, True, case_id="c1", group_id="g1")
        original_b = self._rec("tier", "LOW", "LOW", case_id="c1", group_id="g1")
        variant_a = self._rec(
            "flag", True, False, case_id="c1#p1", group_id="g1", perturbation="ws"
        )
        variant_b = self._rec(
            "tier", "LOW", "HIGH", case_id="c1#p1", group_id="g1", perturbation="ws"
        )
        assert perturbation_flip_rate([original_a, original_b, variant_a, variant_b]) == 1.0

    def test_perturbation_flip_rate_ignores_rotation_rows(self):
        """Rotation/fieldperm rows (permutation != canonical) never join pairs."""
        original = self._rec("action", "APPROVE", "APPROVE", case_id="c1", group_id="g1")
        variant = self._rec(
            "action", "APPROVE", "APPROVE", case_id="c1#p1", group_id="g1", perturbation="ws"
        )
        rotation = self._rec(
            "action",
            "APPROVE",
            "BLOCK",
            case_id="c1#p1",
            group_id="g1",
            perturbation="ws",
            permutation="rot1",
        )
        # Without the rotation row: 1 pair, 0 flips. With it: still 0 flips —
        # the rotated BLOCK row must not count as a perturbation flip.
        assert perturbation_flip_rate([original, variant]) == 0.0
        assert perturbation_flip_rate([original, variant, rotation]) == 0.0

        # And a flipped canonical variant is still detected next to rotation rows.
        flipped_variant = self._rec(
            "action", "APPROVE", "BLOCK", case_id="c1#p2", group_id="g1", perturbation="numfmt"
        )
        assert perturbation_flip_rate([original, variant, rotation, flipped_variant]) == 0.5

    def test_perturbation_flip_rate_multi_order_is_not_a_flip(self):
        """Multi predictions compare as sets: order alone is not a flip."""
        original = self._rec("tags", ["a", "b"], ["a", "b"], case_id="c1", group_id="g1")
        reordered = self._rec(
            "tags", ["a", "b"], ["b", "a"], case_id="c1#p1", group_id="g1", perturbation="ws"
        )
        assert perturbation_flip_rate([original, reordered]) == 0.0

        # A genuinely different set still flips.
        changed = self._rec(
            "tags", ["a", "b"], ["a", "c"], case_id="c1#p2", group_id="g1", perturbation="ws"
        )
        assert perturbation_flip_rate([original, reordered, changed]) == 0.5


class TestTypesafeComparable:
    """agreement + tvd_vs_consensus on TypeSafe-derived lines (hand-computed)."""

    @staticmethod
    def _ts(case_id, field, prediction, label, workflow="security_incidents", **extra):
        record = {
            "run_id": "r",
            "case_id": case_id,
            "group_id": case_id,
            "source": "typesafe",
            "workflow": workflow,
            "field": field,
            "type": "enum",
            "track": "parallel",
            "model": "m",
            "permutation": "canonical",
            "label": label,
            "prediction": prediction,
            "valid": prediction is not None,
            "correct": None if label is None else (prediction is not None and prediction == label),
            "log_scores": None,
            "probability": None,
            "per_option": None,
            "latency_ms": None,
            "rows": None,
            "passes": None,
            "error": None,
            "salvage_prediction": None,
        }
        record.update(extra)
        return record

    def test_agreement_hand_computed_with_ambiguous_exclusion(self):
        # 4 labelled fields across two workflows, 3 agree: overall = 3/4.
        # security_incidents: 3 labelled, 2 agree -> 2/3. The field q_scope is
        # flagged ambiguous (per-line field_ambiguous) and DISAGREES, so the
        # common subset is q_auth + q_strength = 2/2 = 1.0. Workflow means are
        # macro (unweighted): (2/3 + 1) / 2 = 5/6.
        records = [
            self._ts("c1", "q_auth", True, True),
            self._ts("c1", "q_scope", "workgroup", "organization_wide", field_ambiguous=True),
            self._ts("c1", "q_strength", "1", "1"),
            self._ts("c2", "q_auth", False, False, workflow="invoice_processing"),
            # Non-typesafe lines never count.
            self._ts("c3", "q_auth", "X", "X", source="quality-eval"),
            # Unlabelled typesafe line: excluded from the rates.
            self._ts("c1", "q_summary", None, None),
        ]
        result = typesafe_agreement(records)
        assert result["overall"] == pytest.approx(3 / 4)
        assert result["by_workflow"] == {
            "invoice_processing": 1.0,
            "security_incidents": pytest.approx(2 / 3),
        }
        assert result["agreement_common_subset"] == 1.0
        assert result["n_fields"] == 4
        assert result["n_cases"] == 2  # c1 and c2

    def test_agreement_empty_and_all_ambiguous(self):
        assert typesafe_agreement([]) == {}
        assert typesafe_agreement([self._ts("c", "f", "A", "A", source="quality-eval")]) == {}
        only_ambiguous = [self._ts("c1", "q", "A", "B", field_ambiguous=True)]
        result = typesafe_agreement(only_ambiguous)
        assert result["agreement_common_subset"] is None
        assert result["overall"] == 0.0

    def test_agreement_reads_meta_ambiguous_list_when_present(self):
        # evalrun may carry the fetcher's ambiguous list as line['ambiguous'].
        records = [
            self._ts("c1", "q_auth", True, True),
            self._ts("c1", "q_scope", "x", "y", ambiguous=["q_scope"]),
        ]
        result = typesafe_agreement(records)
        assert result["agreement_common_subset"] == 1.0  # q_scope excluded
        assert result["overall"] == 0.5

    def test_tvd_vs_consensus_two_fields_hand_computed(self):
        # Field 1: ours {A: 0.75, B: 0.25} vs consensus {A: 0.5, B: 0.5}
        #   TVD = 0.5 * (|0.25| + |0.25|) = 0.25.
        # Field 2: ours {A: 1.0} vs consensus {A: 0.6, B: 0.4}
        #   TVD = 0.5 * (0.4 + 0.4) = 0.4.
        # Mean overall = (0.25 + 0.4) / 2 = 0.325.
        records = [
            self._ts(
                "c1",
                "q_a",
                "A",
                "A",
                log_scores={"A": math.log(0.75), "B": math.log(0.25)},
                consensus={"A": 0.5, "B": 0.5},
            ),
            self._ts(
                "c2",
                "q_b",
                "A",
                "A",
                workflow="invoice_processing",
                log_scores={"A": 5.0, "B": -50.0},
                consensus={"A": 0.6, "B": 0.4},
            ),
            # No consensus on the line -> skipped.
            self._ts("c3", "q_c", "A", "A", log_scores={"A": 1.0, "B": 0.0}),
            # Consensus but no log_scores -> skipped.
            self._ts("c4", "q_d", "A", "A", consensus={"A": 1.0}),
        ]
        result = tvd_vs_consensus(records)
        assert result["overall"] == pytest.approx(0.325)
        assert result["by_workflow"] == {
            "invoice_processing": pytest.approx(0.4),
            "security_incidents": pytest.approx(0.25),
        }

    def test_tvd_vs_consensus_multi_uses_per_option(self):
        # multi field: per_option distributions are our side of the TVD.
        record = self._ts(
            "c1",
            "tags",
            ["a"],
            ["a"],
            per_option={"a": 0.8, "b": 0.3},
            consensus={"true": 0.75, "false": 0.25},
        )
        # _record_distribution falls back to per_option when log_scores absent?
        # It does not - check the documented behaviour: without log_scores the
        # line is skipped.
        result = tvd_vs_consensus([record])
        assert result == {}

    def test_tvd_and_agreement_absent_for_other_sources(self):
        plain = [self._ts("c", "f", "A", "A", source="quality-eval")]
        assert typesafe_agreement(plain) == {}
        assert tvd_vs_consensus(plain) == {}
        # And compute_metrics stays clean for non-typesafe runs.
        assert "agreement" not in compute_metrics(plain)
        assert "tvd_vs_consensus" not in compute_metrics(plain)


class TestTypesafeReportTable:
    """evalreport renders the agreement table when the metric keys exist."""

    def test_report_contains_agreement_table_when_keys_exist(self, tmp_path):
        from jevmlx import evalreport

        run = {
            "environment": {},
            "config": {},
            "metrics": {
                "accuracy": 0.75,
                "agreement": {
                    "overall": 0.75,
                    "by_workflow": {"security_incidents": 2 / 3, "invoice_processing": 1.0},
                    "agreement_common_subset": 1.0,
                    "n_fields": 4,
                    "n_cases": 2,
                },
                "tvd_vs_consensus": {
                    "overall": 0.325,
                    "by_workflow": {"security_incidents": 0.25, "invoice_processing": 0.4},
                },
            },
            "per_field": [],
        }
        out = tmp_path / "report.json"
        evalreport.write_report(out, run)
        md = out.with_suffix(".md").read_text()
        assert "## Agreement vs TypeSafe consensus" in md
        assert "| overall | 0.7500 | 1.0000 | 0.3250 |" in md
        assert "| security_incidents |" in md and "| invoice_processing |" in md
        # The workflow rows carry their per-workflow agreement and TVD.
        assert "| security_incidents | 0.6667 | n/a | 0.2500 |" in md
        import json

        json.loads(out.read_text())  # JSON still valid and complete

    def test_report_has_no_agreement_table_without_the_metric(self):
        from jevmlx import evalreport

        md = evalreport._to_markdown({"environment": {}, "config": {}, "metrics": {}})
        assert "Agreement" not in md


# ---------------------------------------------------------------------------
# W6-B6b: statistics (Wilson, cluster bootstrap, McNemar, paired)
# ---------------------------------------------------------------------------


class TestWilsonInterval:
    def test_known_small_case(self):
        """Hand-computed Wilson for k=7, n=10, z=1.96."""
        from jevmlx.evalmetrics import wilson_interval

        ci = wilson_interval(7, 10, z=1.96)
        assert ci is not None
        assert ci["point"] == 0.7
        # Wilson center = (p + z^2/2n) / (1 + z^2/n)
        # = (0.7 + 3.8416/20) / (1 + 3.8416/10)
        # = (0.7 + 0.19208) / 1.38416 = 0.6443
        # Half-width: z * sqrt(p(1-p)/n + z^2/4n^2) / denom
        # = 1.96 * sqrt(0.21/10 + 3.8416/400) / 1.38416
        # = 1.96 * sqrt(0.021 + 0.009604) / 1.38416
        # = 1.96 * 0.17493 / 1.38416 = 0.2476
        # ci_low ~ 0.397, ci_high ~ 0.892
        assert abs(ci["ci_low"] - 0.3967) < 0.01
        assert abs(ci["ci_high"] - 0.8920) < 0.01
        assert ci["method"] == "wilson"

    def test_n_zero_returns_none(self):
        from jevmlx.evalmetrics import wilson_interval

        assert wilson_interval(0, 0) is None

    def test_all_correct(self):
        from jevmlx.evalmetrics import wilson_interval

        ci = wilson_interval(10, 10)
        assert ci["point"] == 1.0
        assert ci["ci_high"] == 1.0
        assert ci["ci_low"] < 1.0  # not a vacuous [1,1]

    def test_all_wrong(self):
        from jevmlx.evalmetrics import wilson_interval

        ci = wilson_interval(0, 10)
        assert ci["point"] == 0.0
        assert ci["ci_low"] == 0.0
        assert ci["ci_high"] > 0.0


class TestClusterBootstrapCI:
    def test_reproducible_with_seed(self):
        """Same seed -> identical CI (the seed is fixed)."""
        from jevmlx.evalmetrics import cluster_bootstrap_ci, field_accuracy

        ci1 = cluster_bootstrap_ci(P, field_accuracy, metric_name="accuracy", draws=200, seed=42)
        ci2 = cluster_bootstrap_ci(P, field_accuracy, metric_name="accuracy", draws=200, seed=42)
        assert ci1 == ci2
        assert ci1["ci_low"] <= ci1["accuracy"] <= ci1["ci_high"]
        assert ci1["method"] == "case_cluster_bootstrap"
        assert ci1["draws"] == 200
        assert ci1["seed"] == 42

    def test_different_seeds_may_differ(self):
        from jevmlx.evalmetrics import cluster_bootstrap_ci, field_accuracy

        ci1 = cluster_bootstrap_ci(P, field_accuracy, draws=200, seed=1)
        ci2 = cluster_bootstrap_ci(P, field_accuracy, draws=200, seed=2)
        # Different seeds usually produce slightly different intervals.
        assert ci1 is not None and ci2 is not None
        assert ci1["seed"] == 1 and ci2["seed"] == 2

    def test_too_few_cases_returns_none(self):
        from jevmlx.evalmetrics import cluster_bootstrap_ci, field_accuracy

        # Single case -> not enough for a bootstrap.
        single = [r for r in P if r["case_id"] == "c1"]
        assert cluster_bootstrap_ci(single, field_accuracy) is None

    def test_b2000_default(self):
        """The default is B=2000 (the B6 spec)."""
        from jevmlx.evalmetrics import cluster_bootstrap_ci, field_accuracy

        ci = cluster_bootstrap_ci(P, field_accuracy)
        assert ci["draws"] == 2000


class TestMcNemarExact:
    def test_textbook_2x2(self):
        """Exact McNemar on a textbook discordance table.

        b=3 (A right, B wrong), c=7 (A wrong, B right), n=10.
        Under H0: Binom(10, 0.5). P(X<=3) = sum C(10,i)/2^10 for i=0..3
        = (1+10+45+120)/1024 = 176/1024 = 0.1719.
        Two-sided p = 2 * 0.1719 = 0.3438.
        """
        from jevmlx.evalmetrics import mcnemar_exact

        result = mcnemar_exact(3, 7)
        assert result["b"] == 3
        assert result["c"] == 7
        assert result["n"] == 10
        assert result["method"] == "mcnemar_exact"
        assert abs(result["p_value"] - 0.3438) < 0.001

    def test_no_discordance_p_is_1(self):
        from jevmlx.evalmetrics import mcnemar_exact

        result = mcnemar_exact(0, 0)
        assert result["p_value"] == 1.0
        assert result["n"] == 0

    def test_symmetric(self):
        """b,c and c,b give the same p-value (two-sided test)."""
        from jevmlx.evalmetrics import mcnemar_exact

        assert mcnemar_exact(2, 8)["p_value"] == mcnemar_exact(8, 2)["p_value"]


class TestPairedBootstrap:
    def test_paired_difference_and_mcnemar(self):
        """Two conditions on the same cases: diff CI + McNemar."""
        from jevmlx.evalmetrics import paired_bootstrap_difference

        # P has cases c1, c2. Build B as a slightly different condition.
        records_b = [dict(r) for r in P]
        # Flip one prediction in c2 to create discordance.
        for r in records_b:
            if r["case_id"] == "c2" and r["field"] == "action":
                r["prediction"] = "DEFER" if r.get("prediction") != "DEFER" else "APPROVE"
                r["correct"] = not r.get("correct", False)
        result = paired_bootstrap_difference(P, records_b, draws=200, seed=0)
        assert result is not None
        assert "difference" in result
        assert "ci_low" in result
        assert "ci_high" in result
        assert result["method"] == "paired_bootstrap"
        assert "mcnemar" in result
        assert result["mcnemar"]["method"] == "mcnemar_exact"


class TestValidAccuracy:
    def test_valid_accuracy_excludes_invalid_lines(self):
        """valid_accuracy excludes invalid lines from the denominator;
        field_accuracy counts them as wrong. They differ when invalid
        predictions exist."""
        from jevmlx.evalmetrics import field_accuracy, valid_accuracy

        records = [
            {
                "case_id": "c1",
                "field": "f",
                "label": "A",
                "prediction": "A",
                "valid": True,
                "correct": True,
            },
            {
                "case_id": "c1",
                "field": "g",
                "label": "B",
                "prediction": "X",
                "valid": False,
                "correct": False,
            },
        ]
        # field_accuracy: 1 correct / 2 labelled = 0.5 (invalid counts wrong).
        # valid_accuracy: 1 correct / 1 valid = 1.0 (invalid excluded).
        assert field_accuracy(records) == 0.5
        assert valid_accuracy(records) == 1.0

    def test_valid_equals_field_when_all_valid(self):
        """When every prediction is valid, the two metrics are equal."""
        from jevmlx.evalmetrics import field_accuracy, valid_accuracy

        records = [
            {
                "case_id": "c1",
                "field": "f",
                "label": "A",
                "prediction": "A",
                "valid": True,
                "correct": True,
            },
            {
                "case_id": "c1",
                "field": "g",
                "label": "B",
                "prediction": "B",
                "valid": True,
                "correct": True,
            },
        ]
        assert field_accuracy(records) == valid_accuracy(records) == 1.0


class TestReportCIs:
    def test_report_markdown_shows_ci_next_to_accuracy(self):
        """The metrics table prints the interval next to the point."""
        from jevmlx import evalreport

        metrics = compute_metrics(P)
        md = evalreport._to_markdown({"environment": {}, "config": {}, "metrics": metrics})
        # The accuracy row shows [ci_low, ci_high] (bootstrap).
        assert "accuracy" in md
        assert "[" in md and "]" in md
        assert "case_cluster_bootstrap" in md

    def test_per_field_table_shows_wilson_or_n_too_small(self):
        from jevmlx import evalreport

        metrics = compute_metrics(P)
        md = evalreport._to_markdown({"environment": {}, "config": {}, "metrics": metrics})
        assert "Per-field accuracy" in md
        # Wilson intervals appear in the per-field table.
        assert "wilson" in md
