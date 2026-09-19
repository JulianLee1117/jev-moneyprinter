"""Synthetic consistency regressions; no labels/prompts inferred from returns."""

import copy
import json
from pathlib import Path
import unittest

from jev_alpha.domain import guard_panel
from jev_alpha.jev import MODEL, JevValidationError, build_request


PACK = json.loads((Path(__file__).resolve().parents[1]
                   / "research/jev-core-question-pack.v1.json").read_text(encoding="utf-8"))


def review():
    return {"status": "complete", "reviewer": "synthetic-reviewer",
            "reviewed_at": "2026-09-18T12:00:00Z"}


def state():
    return {
        "current_source_passages_with_ids": [{"passage_id": "p1", "text": "Synthetic operative source."}],
        "prior_public_case_state_with_evidence": {"completeness": "unknown", "documents": []},
        "validated_rates_and_dates": {}, "dated_candidate_issuer_exposures": [],
        "earlier_public_source_evidence": [],
        "episode_manifest": {
            "document_id": "2026-12345", "source_sha256": "a" * 64,
            "content_availability": {"candidate_timestamp": "2026-05-15T08:45:00-04:00",
                                     "content_version_matches_timestamp": False},
        },
    }


def fixture(evidence=None, choices=None):
    request = build_request(state() if evidence is None else evidence, PACK)
    choices = choices or {}
    answers = {}
    for name, question in request["questions"].items():
        if question["type"] == "choice":
            chosen = choices.get(name, "unknown")
            answers[name] = {"type": "choice", "choice": chosen,
                             "probabilities": {key: float(key == chosen) for key in question["criteria"]}}
        else:
            answers[name] = {"type": "noul", "noul": 0.5}
    return request, {"model": MODEL + "-synthetic", "id": "synthetic-only",
                     "usage": {"input_tokens": 100, "output_tokens": 100, "cost": 0.0001},
                     "answers": answers, "extra_metadata": {"retained": True}}


def exposure(issuer):
    return {"issuer_id": issuer, "as_of": "2026-04-01", "available_at": "2026-04-02T12:00:00Z",
            "review": review(),
            "evidence": [{"source_url": "https://example.test/exposure", "excerpt": "Synthetic dated exposure."}]}


class PanelConsistencyTests(unittest.TestCase):
    def test_empty_exposures_with_unknown_stay_unknown_without_contradictions(self):
        request, response = fixture()
        result = guard_panel(request, response)
        self.assertEqual("needs_review", result["status"])
        self.assertEqual([], result["contradictions"])
        self.assertEqual("unknown", result["raw_model_response"]["answers"]["nue_exposure_support"]["choice"])
        self.assertIn("dated_candidate_issuer_exposures", result["unresolved_evidence"])
        missing = [b["issuer_id"] for b in result["blockers"] if b["code"] == "issuer_exposure_missing"]
        self.assertEqual(["NUE", "STLD"], missing)

    def test_absence_labels_are_flagged_not_rewritten(self):
        choices = {"nue_exposure_support": "none", "stld_exposure_support": "none",
                   "nue_offsetting_channel": "no_documented_offset",
                   "stld_offsetting_channel": "no_documented_offset"}
        request, response = fixture(choices=choices)
        original_request, original_response = copy.deepcopy(request), copy.deepcopy(response)
        result = guard_panel(request, response)
        self.assertEqual(set(choices), {c["question_id"] for c in result["contradictions"]})
        self.assertTrue(all(c["code"] == "unsupported_exposure_interpretation" for c in result["contradictions"]))
        self.assertEqual(original_request, request)
        self.assertEqual(original_response, response)
        self.assertEqual(response, result["raw_model_response"])
        self.assertFalse(result["raw_probabilities_modified"])
        result["raw_model_response"]["answers"]["nue_exposure_support"]["probabilities"]["none"] = 0
        self.assertEqual(1, response["answers"]["nue_exposure_support"]["probabilities"]["none"])

    def test_missing_exposure_section_also_flags_unsupported_producer_claim(self):
        request, response = fixture(choices={"nue_producer_channel": "favorable"})
        request["state"]["evidence"].pop("dated_candidate_issuer_exposures")
        result = guard_panel(request, response)
        self.assertEqual(["nue_producer_channel"], [c["question_id"] for c in result["contradictions"]])
        self.assertFalse(result["trade_eligible"])

    def test_unknown_prior_blocks_comparable_and_positive_novelty(self):
        for comparison in ("matched", "partial"):
            for novelty in ("new_or_changed", "correction", "mixed", "confirmation"):
                with self.subTest(comparison=comparison, novelty=novelty):
                    request, response = fixture(choices={"comparison_validity": comparison,
                                                        "novelty_in_supplied_history": novelty})
                    result = guard_panel(request, response)
                    self.assertEqual({"comparison_validity", "novelty_in_supplied_history"},
                                     {c["question_id"] for c in result["contradictions"]})
                    self.assertEqual(novelty, result["raw_model_response"]["answers"]["novelty_in_supplied_history"]["choice"])

    def test_unknown_prior_does_not_convert_mismatched_into_matched(self):
        request, response = fixture(choices={"comparison_validity": "mismatched"})
        result = guard_panel(request, response)
        self.assertEqual([], result["contradictions"])
        self.assertEqual("needs_review", result["status"])

    def test_nonempty_map_does_not_prove_review_or_cover_other_issuers(self):
        evidence = state()
        evidence["dated_candidate_issuer_exposures"] = [{"issuer_id": "NUE", "as_of": "2026-01-01"}]
        request, response = fixture(evidence, {"nue_exposure_support": "direct", "stld_exposure_support": "none"})
        result = guard_panel(request, response)
        self.assertEqual("needs_review", result["status"])
        self.assertFalse(result["trade_eligible"])
        self.assertEqual([], result["reviewed_issuer_ids"])
        self.assertTrue(any(b["code"] == "issuer_exposure_unreviewed" for b in result["blockers"]))
        self.assertEqual(["stld_exposure_support"], [c["question_id"] for c in result["contradictions"]])

    def test_future_or_undated_records_cannot_establish_historical_coverage(self):
        for change in ({"as_of": "2026-06-01"}, {"available_at": "2026-05-16T01:00:00Z"},
                       {"available_at": "2026-05-15"}, {"available_at": None}):
            with self.subTest(change=change):
                evidence = state()
                record = exposure("NUE")
                record.update(change)
                evidence["dated_candidate_issuer_exposures"] = [record]
                request, response = fixture(evidence)
                result = guard_panel(request, response)
                self.assertTrue(any(b["code"] == "issuer_exposure_time_unresolved" and b["issuer_id"] == "NUE"
                                    for b in result["blockers"]))
                self.assertNotIn("NUE", result["reviewed_issuer_ids"])
                self.assertFalse(result["historical_hindsight_eliminated"])

    def test_fully_reviewed_attestations_still_never_authorize_trade(self):
        evidence = state()
        evidence["episode_manifest"]["review"] = review()
        evidence["episode_manifest"]["content_availability"]["content_version_matches_timestamp"] = True
        evidence["prior_public_case_state_with_evidence"] = {
            "completeness": "complete", "documents": [{"source": "synthetic prior"}], "review": review()}
        evidence["dated_candidate_issuer_exposures"] = [exposure("NUE"), exposure("STLD")]
        evidence["validated_rates_and_dates"]["review"] = review()
        request, response = fixture(evidence, {"nue_exposure_support": "direct", "comparison_validity": "matched"})
        result = guard_panel(request, response)
        self.assertEqual("consistent_with_reviewed_inputs", result["status"])
        self.assertEqual([], result["blockers"])
        self.assertEqual([], result["contradictions"])
        self.assertEqual(["NUE", "STLD"], result["reviewed_issuer_ids"])
        self.assertFalse(result["trade_eligible"])
        self.assertFalse(result["alpha_proven"])
        self.assertFalse(result["market_surprise_established"])
        self.assertEqual("interpretation_consistency_only", result["scope"])

    def test_a_review_label_without_attribution_is_not_a_review(self):
        evidence = state()
        evidence["episode_manifest"]["review"] = {"status": "complete"}
        evidence["prior_public_case_state_with_evidence"] = {"completeness": "complete", "documents": [{}]}
        request, response = fixture(evidence, {"comparison_validity": "matched"})
        result = guard_panel(request, response)
        codes = {b["code"] for b in result["blockers"]}
        self.assertIn("source_review_incomplete", codes)
        self.assertIn("prior_state_not_reviewed_complete", codes)
        self.assertEqual("needs_review", result["status"])

    def test_unwrapped_state_is_supported_and_malformed_response_fails(self):
        request, response = fixture()
        wrapped = guard_panel(request, response)
        request["state"] = request["state"]["evidence"]
        self.assertEqual(wrapped, guard_panel(request, response))
        response["answers"].pop("nue_exposure_support")
        with self.assertRaises(JevValidationError):
            guard_panel(request, response)


if __name__ == "__main__":
    unittest.main()
