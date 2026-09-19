"""Synthetic boundary cases; no network, prices or inference calls."""

from copy import deepcopy
import unittest

from jev_alpha.audit import dossier_template, evaluate_audit


def candidate(document_id="2025-12345", episode_id="core:korea:2025-final"):
    row = dossier_template({"document_number": document_id, "title": "Synthetic fixture"})
    row["episode_id"] = episode_id
    row["review"] = {
        "status": "complete", "reviewer": "Synthetic test reviewer",
        "reviewed_at": "2026-09-18T12:00:00Z",
    }
    for name in ("material_change", "issuer_exposure", "economic_magnitude", "earliest_availability"):
        row[name].update({
            "status": "supported", "rationale": "Synthetic adjudicated finding.",
            "evidence": [{"source_url": "https://example.test/source", "excerpt": "Synthetic evidence."}],
        })
    row["issuer_exposure"].update({"issuer_ids": ["NUE", "STLD"], "as_of": "2025-01-01"})
    row["earliest_availability"].update({
        "timestamp": "2025-07-10T08:45:00-04:00", "earlier_sources_checked": True,
        "content_version_matches_timestamp": True,
    })
    return row


class AuditTests(unittest.TestCase):
    def test_discovery_and_model_claims_never_prepopulate_adjudication(self):
        row = dossier_template({
            "document_number": "2026-14026", "publication_date": "2026-07-13",
            "material_change": {"status": "supported"}, "episode_id": "claimed",
            "earliest_availability": "2026-07-13T00:00:00Z",
        })
        self.assertEqual(row["material_change"]["status"], "unknown")
        self.assertIsNone(row["episode_id"])
        self.assertIsNone(row["earliest_availability"]["timestamp"])
        self.assertIs(row["earliest_availability"]["content_version_matches_timestamp"], False)
        self.assertEqual(evaluate_audit([row])["readiness"], "needs_review")

    def test_ready_means_only_price_feasibility(self):
        rows = [candidate(f"2025-{n}", f"episode-{n}") for n in range(3)]
        result = evaluate_audit(rows)
        self.assertEqual(result["readiness"], "ready_for_price_pilot")
        self.assertFalse(result["alpha_proven"])
        self.assertFalse(result["market_surprise_claim_permitted"])
        self.assertEqual(result["counts"]["expectations_unknown_dossiers"], 3)

    def test_documents_countries_and_issuers_are_not_independent_episodes(self):
        rows = [candidate(f"2025-{n}", "same-announcement") for n in range(3)]
        result = evaluate_audit(rows)
        self.assertEqual(result["counts"]["ready_dossiers"], 3)
        self.assertEqual(result["counts"]["ready_episodes"], 1)
        self.assertEqual(result["readiness"], "needs_review")

    def test_same_document_cannot_be_counted_under_multiple_episodes(self):
        rows = [candidate("same-document", f"episode-{n}") for n in range(3)]
        result = evaluate_audit(rows)
        self.assertEqual(result["counts"]["ready_episodes"], 0)
        self.assertEqual(result["counts"]["unique_documents"], 1)
        self.assertEqual(result["counts"]["malformed_dossiers"], 3)

    def test_unknown_magnitude_blocks_even_with_verified_product_exposure(self):
        row = candidate()
        row["economic_magnitude"]["status"] = "unknown"
        result = evaluate_audit([row], min_candidate_episodes=1)
        self.assertEqual(result["readiness"], "needs_review")
        self.assertIn("economic_magnitude", [item["field"] for item in result["blockers"]])

    def test_public_inspection_is_insufficient_without_earlier_channel_check(self):
        for checked in (False, None, "true", 1):
            with self.subTest(checked=checked):
                row = candidate()
                row["earliest_availability"]["earlier_sources_checked"] = checked
                self.assertEqual(evaluate_audit([row], min_candidate_episodes=1)["readiness"], "needs_review")

    def test_published_text_cannot_inherit_pi_time_without_version_evidence(self):
        # The metadata's PI timestamp may be valid while the supplied text is a
        # later published revision. Earlier-channel review alone cannot resolve
        # that mismatch, and truthy strings/integers are not manual verification.
        for verified in (False, None, "true", 1):
            with self.subTest(verified=verified):
                row = candidate()
                row["earliest_availability"]["content_version_matches_timestamp"] = verified
                result = evaluate_audit([row], min_candidate_episodes=1)
                self.assertEqual(result["readiness"], "needs_review")
                self.assertIn("earliest_availability.content_version_matches_timestamp",
                              [item["field"] for item in result["blockers"]])
        row = candidate()
        del row["earliest_availability"]["content_version_matches_timestamp"]
        self.assertEqual(evaluate_audit([row], min_candidate_episodes=1)["readiness"], "needs_review")

    def test_naive_date_only_and_invalid_timestamps_block(self):
        for timestamp in ("2025-07-10", "2025-07-10T08:45:00", "2025-02-30T08:45:00Z", 123):
            with self.subTest(timestamp=timestamp):
                row = candidate()
                row["earliest_availability"]["timestamp"] = timestamp
                self.assertEqual(evaluate_audit([row], min_candidate_episodes=1)["readiness"], "needs_review")

    def test_future_exposure_mapping_blocks(self):
        row = candidate()
        row["issuer_exposure"]["as_of"] = "2025-07-11"
        self.assertEqual(evaluate_audit([row], min_candidate_episodes=1)["readiness"], "needs_review")

    def test_supported_assertion_needs_real_citation_shape(self):
        for evidence in ([], [{"source_url": "not a URL", "excerpt": "x"}],
                         [{"source_url": "https://example.test", "excerpt": ""}],
                         [{"source_url": "https://[broken", "excerpt": "x"}], None):
            with self.subTest(evidence=evidence):
                row = candidate()
                row["material_change"]["evidence"] = evidence
                self.assertEqual(evaluate_audit([row], min_candidate_episodes=1)["readiness"], "needs_review")

    def test_exclusion_requires_evidence_but_not_unnecessary_downstream_work(self):
        row = dossier_template({"document_number": "clerical"})
        row["episode_id"] = "clerical-episode"
        row["review"] = candidate()["review"]
        row["material_change"] = {
            "status": "unsupported", "rationale": "Clerical correction only.",
            "evidence": [{"source_url": "https://example.test/correction", "excerpt": "Corrects spelling only."}],
        }
        self.assertEqual(evaluate_audit([row])["readiness"], "reject")
        row["material_change"]["evidence"] = []
        self.assertEqual(evaluate_audit([row])["readiness"], "needs_review")

    def test_unresolved_selected_dossier_blocks_cohort(self):
        rows = [candidate(f"doc-{n}", f"episode-{n}") for n in range(3)]
        unresolved = candidate("doc-4", "episode-4")
        unresolved["review"]["status"] = "pending"
        self.assertEqual(evaluate_audit(rows + [unresolved])["readiness"], "needs_review")

    def test_partial_episode_not_in_eligible_episode_ids(self):
        first = candidate("doc-1", "same")
        second = candidate("doc-2", "same")
        second["economic_magnitude"]["status"] = "unknown"
        result = evaluate_audit([first, second], min_candidate_episodes=1)
        self.assertEqual(result["eligible_episode_ids"], [])

    def test_malformed_root_rows_and_nested_values_do_not_crash(self):
        malformed = [None, "bad", 2, {}, {"document_id": []},
                     {"schema_version": "profit-audit-v1", "document_id": "d", "episode_id": "e", "material_change": {"status": []}}]
        for value in (None, {}, "bad", malformed):
            with self.subTest(value=value):
                result = evaluate_audit(value)
                self.assertEqual(result["readiness"], "needs_review")
                self.assertTrue(result["blockers"])

    def test_invalid_threshold_cannot_relax_gate(self):
        for value in (0, -1, True, 1.5, "3"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                evaluate_audit([], min_candidate_episodes=value)

    def test_input_is_not_mutated(self):
        rows = [candidate()]
        before = deepcopy(rows)
        evaluate_audit(rows)
        self.assertEqual(rows, before)


if __name__ == "__main__":
    unittest.main()
