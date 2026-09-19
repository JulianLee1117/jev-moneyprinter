"""Offline exact-support accounting tests; all fixtures are synthetic."""

import copy
import unittest

from jev_alpha.coverage import (evaluate_coverage, freeze_support_labels,
                                selection_from_fanout, selection_from_state)


def source(doc="d1", text="Rate 5%. Except steel 鋼🙂."):
    return {"episode_manifest": {"document_id": doc, "source_sha256": "a" * 64,
                                  "passage_selection": "all"},
            "current_source_passages_with_ids": [
                {"passage_id": "p1", "text": text},
                {"passage_id": "p2", "text": "Routine contact information."}]}


def labels(states):
    return {"records": [{"document_id": s["episode_manifest"]["document_id"],
                          "question_id": "exception_limits_headline", "expected": True,
                          "label_origin": "agent_reviewed", "split": "development",
                          "evidence": [{"source_sha256": "a" * 64,
                                        "passage_ids": ["p1"],
                                        "excerpt": s["current_source_passages_with_ids"][0]["text"]}]}
                         for s in states]}


class CoverageTests(unittest.TestCase):
    def setUp(self):
        self.states = [source()]
        self.labels = labels(self.states)
        self.frozen = freeze_support_labels(self.labels, self.states)

    def report(self, selections, calls=None):
        return evaluate_coverage(self.frozen, self.states, {"test": {
            "selections": selections, "inference_calls": calls or []}})["arms"]["test"]

    def test_whole_source_is_explicit_coverage_ceiling(self):
        result = self.report([selection_from_state(self.states[0])])
        self.assertEqual(result["label_support_recall"], 1)
        self.assertEqual(result["review_character_fraction"], 1)
        self.assertEqual(result["unsafe_exclusion_candidates"], [])
        self.assertEqual(result["cost"]["total_reported_cost_usd"], 0)
        self.assertGreater(result["review_text_utf8_bytes"], result["review_text_characters"])

    def test_partial_parent_is_not_counted_as_whole_support(self):
        text = self.states[0]["current_source_passages_with_ids"][0]["text"]
        fanout = {"schema_version": "jev-passage-selection-v1", "document_id": "d1",
                  "source_sha256": "a" * 64, "status": "complete",
                  "selected_fragments": [{"parent_passage_id": "p1", "offset_start": 0,
                                          "offset_end": 8, "text": text[:8]}]}
        result = self.report([selection_from_fanout(fanout)])
        self.assertEqual(result["labels_fully_retained"], 0)
        self.assertEqual(result["distinct_support_characters_retained"], 8)
        self.assertEqual(result["exception_support_omitted_or_unresolved"], ["d1:exception_limits_headline"])
        self.assertEqual(result["labels"][0]["omitted_support"][0]["retained_characters"], 8)

    def test_overlaps_and_duplicate_spans_do_not_inflate_coverage(self):
        selection = selection_from_state(self.states[0])
        selection["spans"] += copy.deepcopy(selection["spans"])
        result = self.report([selection])
        self.assertEqual(result["review_character_fraction"], 1)
        self.assertEqual(result["distinct_support_character_recall"], 1)
        self.assertEqual(result["display_text_characters_with_duplicates"], 2*result["review_text_characters"])

    def test_fragment_union_handles_gap_and_adjacent_unicode_offsets(self):
        selection = selection_from_state(self.states[0])
        text = selection["spans"][0]["text"]
        selection["spans"] = [{"passage_id": "p1", "offset_start": a,
                                "offset_end": b, "text": text[a:b]}
                               for a, b in [(0, 5), (6, len(text))]]
        self.assertEqual(self.report([selection])["labels_fully_retained"], 0)
        selection["spans"].append({"passage_id": "p1", "offset_start": 5,
                                   "offset_end": 6, "text": text[5:6]})
        self.assertEqual(self.report([selection])["labels_fully_retained"], 1)

    def test_missing_method_documents_stay_in_denominator(self):
        self.states.append(source("d2"))
        self.frozen = freeze_support_labels(labels(self.states), self.states)
        result = self.report([selection_from_state(self.states[0])])
        self.assertEqual(result["label_count"], 2)
        self.assertEqual(result["label_support_recall"], .5)
        self.assertEqual(result["missing_selection_documents"], ["d2"])
        self.assertTrue(result["labels"][1]["selection_document_missing"])

    def test_missing_source_label_remains_unresolved_in_denominator(self):
        absent = labels([source("absent")])["records"][0]
        self.labels["records"].append(absent)
        self.frozen = freeze_support_labels(self.labels, self.states)
        result = self.report([selection_from_state(self.states[0])])
        self.assertEqual(result["label_count"], 2)
        self.assertEqual(result["resolved_label_count"], 1)
        self.assertEqual(result["label_support_recall"], .5)
        self.assertIn("source_document_missing", result["labels"][1]["source_issues"])

    def test_same_support_for_two_questions_is_not_two_independent_spans(self):
        label = copy.deepcopy(self.labels["records"][0])
        label["question_id"] = "treatment_time_basis"
        self.labels["records"].append(label)
        self.frozen = freeze_support_labels(self.labels, self.states)
        result = self.report([selection_from_state(self.states[0])])
        self.assertEqual(result["label_count"], 2)
        self.assertEqual(result["distinct_support_passage_count"], 1)
        self.assertEqual(result["distinct_support_characters"], len(label["evidence"][0]["excerpt"]))

    def test_incomplete_fallback_is_covered_but_not_complete_execution(self):
        selection = selection_from_state(self.states[0])
        selection["status"] = "incomplete"
        result = self.report([selection])
        self.assertEqual(result["label_support_recall"], 1)
        self.assertEqual(result["incomplete_selector_documents"], ["d1"])

    def test_cost_dedup_and_unknown_cost_do_not_turn_into_free_inference(self):
        result = self.report([], [{"call_id": "a", "reported_cost_usd": .2},
                                 {"call_id": "a", "reported_cost_usd": .2},
                                 {"call_id": "b"}])
        self.assertEqual(result["cost"]["unique_inference_calls"], 2)
        self.assertEqual(result["cost"]["known_reported_cost_usd"], .2)
        self.assertIsNone(result["cost"]["total_reported_cost_usd"])
        with self.assertRaisesRegex(ValueError, "Conflicting costs"):
            self.report([], [{"call_id": "a", "reported_cost_usd": 1},
                             {"call_id": "a", "reported_cost_usd": 2}])

    def test_invalid_cost_is_rejected(self):
        for cost in [True, float("nan"), float("inf"), -1, ".1"]:
            with self.subTest(cost=cost), self.assertRaises(ValueError):
                self.report([], [{"call_id": "a", "reported_cost_usd": cost}])

    def test_source_snapshot_changes_are_rejected(self):
        self.states[0]["current_source_passages_with_ids"][0]["text"] += "Changed"
        with self.assertRaisesRegex(ValueError, "exact source snapshot"):
            self.report([])

    def test_wrong_selection_hash_text_or_offset_is_rejected(self):
        for field, value in [("text", "invented"), ("offset_start", -1),
                             ("offset_end", 999), ("offset_start", True)]:
            selection = selection_from_state(self.states[0])
            selection["spans"][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.report([selection])
        selection = selection_from_state(self.states[0])
        selection["source_sha256"] = "b" * 64
        with self.assertRaises(ValueError):
            self.report([selection])

    def test_excerpt_mismatch_duplicate_labels_and_selected_documents_rejected(self):
        corrupted = copy.deepcopy(self.labels)
        corrupted["records"][0]["evidence"][0]["excerpt"] = "A guess"
        with self.assertRaisesRegex(ValueError, "excerpt"):
            freeze_support_labels(corrupted, self.states)
        self.labels["records"] *= 2
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            freeze_support_labels(self.labels, self.states)
        selection = selection_from_state(self.states[0])
        with self.assertRaisesRegex(ValueError, "Duplicate selected"):
            self.report([selection, selection])

    def test_partial_source_cannot_supply_full_source_denominator(self):
        self.states[0]["episode_manifest"]["original_passage_count"] = 99
        with self.assertRaisesRegex(ValueError, "original passage count"):
            freeze_support_labels(self.labels, self.states)
        self.states[0]["episode_manifest"]["passage_selection"] = "keyword-neighborhood-v1"
        with self.assertRaisesRegex(ValueError, "full source"):
            freeze_support_labels(self.labels, self.states)

    def test_freezing_does_not_mutate_source_or_labels(self):
        states, original = copy.deepcopy(self.states), copy.deepcopy(self.labels)
        freeze_support_labels(self.labels, self.states)
        self.assertEqual(states, self.states)
        self.assertEqual(original, self.labels)


if __name__ == "__main__":
    unittest.main()
