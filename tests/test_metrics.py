import copy
import math
import unittest

from jev_alpha.metrics import MetricsValidationError, evaluate_labels


def label(document="d1", question="q1", expected="yes", origin="human", split="development"):
    return {"document_id": document, "question_id": question, "expected": expected,
            "label_origin": origin, "split": split,
            "evidence": [{"source_url": "https://example.org/source", "excerpt": "Source evidence."}]}


def prediction(document="d1", question="q1", model="jev", probabilities=None, **kwargs):
    row = {"document_id": document, "question_id": question, "model": model,
           "input_hash": document, "kind": "choice",
           "probabilities": probabilities or {"yes": 0.8, "no": 0.2}}
    row.update(kwargs)
    return row


def pack(*records):
    return {"schema_version": "benchmark_labels_v1", "records": list(records)}


class MetricsTests(unittest.TestCase):
    def test_missing_abstained_and_unavailable_keep_full_denominator(self):
        labels = pack(*(label(document=f"d{i}") for i in range(4)))
        predictions = [prediction(document="d0"), prediction(document="d1", abstain=True),
                       prediction(document="d2", input_available=False)]
        result = evaluate_labels(labels, predictions)["groups"][0]
        self.assertEqual(result["label_count"], 4)
        self.assertEqual(result["accuracy"], 0.25)
        self.assertEqual(result["answered_accuracy"], 1)
        self.assertEqual(result["coverage"], 0.25)
        self.assertEqual(result["missing_predictions"], 1)
        self.assertEqual(result["abstentions"], 1)
        self.assertEqual(result["input_unavailable"], 1)
        self.assertAlmostEqual(result["choice_brier_mean"], 0.08)
        self.assertAlmostEqual(result["choice_brier_worst_case_mean"], 1.5203)

    def test_choice_brier_is_raw_sum_not_divided_by_options(self):
        result = evaluate_labels(pack(label()), [prediction(probabilities={"yes": 0, "no": 1})])
        self.assertEqual(result["groups"][0]["choice_brier_mean"], 2)

    def test_noul_binary_score_and_fixed_threshold(self):
        row = prediction(kind="noul", noul=0.5)
        row.pop("probabilities")
        result = evaluate_labels(pack(label(expected=True)), [row])["groups"][0]
        self.assertEqual(result["accuracy"], 1)
        self.assertEqual(result["noul_brier_mean"], 0.25)
        self.assertIsNone(result["choice_brier_mean"])

    def test_origin_and_split_groups_never_mix_gold_with_synthetic(self):
        labels = pack(label(), label("d2", origin="synthetic"),
                      label("d3", origin="agent_reviewed"), label("d4", split="holdout"))
        predictions = [prediction(document=f"d{i}") for i in range(1, 5)]
        result = evaluate_labels(labels, predictions)
        self.assertEqual(len(result["groups"]), 4)
        self.assertEqual(result["label_origins"], {"human": 2, "agent_reviewed": 1, "synthetic": 1})
        for group in result["groups"]:
            self.assertEqual(group["human_gold_labels"], group["label_origin"] == "human")
        synthetic = next(group for group in result["groups"] if group["label_origin"] == "synthetic")
        self.assertIn("not empirical", synthetic["interpretation"])

    def test_missing_predictions_are_counted_for_each_model(self):
        result = evaluate_labels(pack(label(), label("d2")),
                                 [prediction(), prediction("d2", model="baseline")])
        for group in result["groups"]:
            self.assertEqual(group["label_count"], 2)
            self.assertEqual(group["coverage"], 0.5)

    def test_empty_inputs_do_not_report_perfect_quality(self):
        result = evaluate_labels(pack(), [])
        self.assertEqual(result["status"], "no_labels")
        self.assertEqual(result["groups"], [])
        result = evaluate_labels(pack(label()), [])
        self.assertEqual(result["status"], "no_predictions")
        self.assertEqual(result["groups"], [])

    def test_zero_answered_rows_have_no_conditional_accuracy_or_brier(self):
        row = prediction(abstain=True)
        row.pop("probabilities")
        result = evaluate_labels(pack(label()), [row])["groups"][0]
        self.assertEqual(result["accuracy"], 0)
        self.assertIsNone(result["answered_accuracy"])
        self.assertIsNone(result["choice_brier_mean"])
        self.assertEqual(result["choice_brier_worst_case_mean"], 2.0004)

    def test_rounded_choice_uses_raw_values_and_reports_approximation(self):
        probabilities = {"historical_or_context": 0.23, "operative_change": 0.28,
                         "procedural": 0.33, "scope_or_exception": 0.01, "unclear": 0.14}
        row = prediction(probabilities=probabilities)
        original = copy.deepcopy(row)
        group = evaluate_labels(pack(label(expected="procedural")), [row])["groups"][0]
        self.assertEqual(group["accuracy"], 1)
        self.assertAlmostEqual(group["choice_brier_mean"], 0.5999)
        self.assertEqual(group["choice_rounded_probability_count"], 1)
        self.assertAlmostEqual(group["choice_rounded_probability_rows"][0]["probability_mass_delta"], -0.01)
        self.assertIn("without renormalization", group["choice_brier_basis"])
        self.assertEqual(row, original)

    def test_choice_rounding_does_not_accept_arbitrary_missing_mass(self):
        for probabilities in ({"yes": 0.5, "no": 0.4}, {"yes": 0.5, "no": 0.48},
                              {"yes": 0, "no": 0}, {"yes": 0.501, "no": 0.49}):
            with self.subTest(probabilities=probabilities), self.assertRaises(MetricsValidationError):
                evaluate_labels(pack(label()), [prediction(probabilities=probabilities)])

    def test_allowed_positive_mass_error_may_have_brier_slightly_above_two(self):
        group = evaluate_labels(pack(label()), [prediction(probabilities={
            "yes": 0, "no": 1, "other": 0.02, "unused": 0,
        })])["groups"][0]
        self.assertAlmostEqual(group["choice_brier_mean"], 2.0004)
        self.assertEqual(group["choice_rounded_probability_count"], 1)

    def test_duplicate_labels_rejected_even_with_different_origins(self):
        with self.assertRaisesRegex(MetricsValidationError, "Duplicate"):
            evaluate_labels(pack(label(), label(origin="synthetic")), [])

    def test_duplicate_predictions_rejected(self):
        with self.assertRaisesRegex(MetricsValidationError, "Duplicate"):
            evaluate_labels(pack(label()), [prediction(), prediction()])

    def test_nonfinite_or_invalid_probability_rejected(self):
        for value in (math.nan, math.inf, -math.inf, -0.1, 1.1, True, "0.5", 10**1000):
            with self.subTest(value=repr(value)[:30]):
                with self.assertRaises(MetricsValidationError):
                    evaluate_labels(pack(label()), [prediction(probabilities={"yes": value, "no": 0.2})])
                with self.assertRaises(MetricsValidationError):
                    evaluate_labels(pack(label(expected=True)), [prediction(kind="noul", noul=value)])

    def test_invalid_distribution_not_hidden_by_abstention(self):
        with self.assertRaises(MetricsValidationError):
            evaluate_labels(pack(label()), [prediction(probabilities={"yes": math.nan, "no": 0}, abstain=True)])

    def test_explicit_null_distribution_does_not_bypass_validation(self):
        row = prediction(abstain=True)
        row["probabilities"] = None
        with self.assertRaises(MetricsValidationError):
            evaluate_labels(pack(label()), [row])

    def test_missing_and_abstained_positive_labels_reduce_recall(self):
        labels = pack(label("d1", expected=True), label("d2", expected=True),
                      label("d3", expected=False), label("d4", expected=True))
        predictions = [prediction("d1", kind="noul", noul=0.8),
                       prediction("d3", kind="noul", noul=0.8),
                       prediction("d4", kind="noul", abstain=True, noul=0.9)]
        group = evaluate_labels(labels, predictions)["groups"][0]
        positive = next(row for row in group["per_question_class_metrics"] if row["class_label"] is True)
        self.assertEqual(positive["support"], 3)
        self.assertEqual(positive["recall"], 1 / 3)
        self.assertEqual(positive["precision"], 0.5)
        self.assertEqual(positive["false_negatives_including_unanswered"], 2)

    def test_choice_must_include_expected_label(self):
        with self.assertRaisesRegex(MetricsValidationError, "omits"):
            evaluate_labels(pack(label(expected="other")), [prediction()])

    def test_distribution_options_consistent_across_models(self):
        with self.assertRaisesRegex(MetricsValidationError, "consistent"):
            evaluate_labels(pack(label()), [prediction(), prediction(model="other", probabilities={"yes": 0.8, "maybe": 0.2})])

    def test_tied_prediction_does_not_use_the_expected_label(self):
        row = prediction(probabilities={"yes": 0.5, "no": 0.5})
        self.assertEqual(evaluate_labels(pack(label()), [row])["groups"][0]["accuracy"], 0)
        self.assertEqual(evaluate_labels(pack(label(expected="no")), [row])["groups"][0]["accuracy"], 1)

    def test_real_labels_require_source_excerpt_and_valid_url(self):
        for evidence in ([], [{"source_url": "file:///local", "excerpt": "a"}],
                         [{"source_url": "https://example.org", "excerpt": ""}],
                         [{"source_url": "https://", "excerpt": "a"}]):
            with self.subTest(evidence=evidence):
                row = label()
                row["evidence"] = evidence
                with self.assertRaises(MetricsValidationError):
                    evaluate_labels(pack(row), [])

    def test_synthetic_labels_can_omit_evidence_but_not_origin(self):
        row = label(origin="synthetic")
        row.pop("evidence")
        self.assertFalse(evaluate_labels(pack(row), [prediction()])["human_gold_available"])
        row.pop("label_origin")
        with self.assertRaises(MetricsValidationError):
            evaluate_labels(pack(row), [])

    def test_same_document_cannot_leak_between_splits(self):
        with self.assertRaisesRegex(MetricsValidationError, "both development and holdout"):
            evaluate_labels(pack(label(), label(question="q2", split="holdout")), [])

    def test_score_and_kind_mismatch_rejected(self):
        with self.assertRaisesRegex(MetricsValidationError, "score"):
            evaluate_labels(pack(label()), [prediction(kind="score")])
        with self.assertRaisesRegex(MetricsValidationError, "kind"):
            evaluate_labels(pack(label(expected=True)), [prediction()])

    def test_call_cost_and_latency_are_not_multiplied_by_question_count(self):
        labels = pack(label(), label(question="q2"))
        predictions = [prediction(question=q, run_id="run", cost_usd=0.01, latency_ms=12)
                       for q in ("q1", "q2")]
        result = evaluate_labels(labels, predictions)["model_totals"][0]
        self.assertEqual(result["unique_calls"], 1)
        self.assertEqual(result["cost_usd_total"], 0.01)
        self.assertEqual(result["latency_ms_total"], 12)
        self.assertTrue(result["cost_usd_complete"])

    def test_conflicting_call_metadata_rejected(self):
        with self.assertRaisesRegex(MetricsValidationError, "metadata"):
            evaluate_labels(pack(label(), label(question="q2")),
                            [prediction(cost_usd=0.01), prediction(question="q2", cost_usd=0.02)])

    def test_missing_cost_is_unknown_not_zero(self):
        result = evaluate_labels(pack(label()), [prediction()])["model_totals"][0]
        self.assertIsNone(result["cost_usd_total"])
        self.assertFalse(result["cost_usd_complete"])

    def test_extra_predictions_reported_not_counted_as_more_labels(self):
        result = evaluate_labels(pack(label()), [prediction(), prediction("extra")])
        self.assertEqual(result["groups"][0]["label_count"], 1)
        self.assertEqual(result["model_totals"][0]["unmatched_predictions"], 1)

    def test_input_hash_differences_visible_without_assuming_unequal_evidence(self):
        result = evaluate_labels(pack(label()), [prediction(), prediction(model="other", input_hash="different")])
        self.assertEqual(len(result["cross_model_input_hash_mismatches"]), 1)

    def test_input_not_mutated(self):
        labels, predictions = pack(label()), [prediction()]
        original = copy.deepcopy((labels, predictions))
        evaluate_labels(labels, predictions)
        self.assertEqual((labels, predictions), original)


if __name__ == "__main__":
    unittest.main()
