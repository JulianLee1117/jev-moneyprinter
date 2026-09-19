import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from jev_alpha.experiment import digest
from jev_alpha.jev import JevRequestError
from jev_alpha.transitions import numeric_comparisons, prepare_transition, run_transition, guard_transition


def fixture():
    packets = {}
    docs = []
    dates = {"current": "2026-05-18", "preliminary": "2026-01-08", "earlier_operative_deposit": "2025-05-07"}
    observations = {}
    for role, published in dates.items():
        doc = role
        packets[doc] = {"episode_manifest": {"document_id": doc, "source_sha256": role,
                        "source_url": "https://example.org/" + role,
                        "content_availability": {"candidate_publication_date": published,
                          "candidate_timestamp": "2026-05-15T08:45:00-04:00" if role == "current" else published + "T08:45:00-04:00",
                          "content_version_matches_timestamp": False,
                          "caveat": "Exact archived version not historically established"}},
                        "current_source_passages_with_ids": [{"passage_id": role, "text": "rate evidence"}]}
        docs.append({"document_id": doc, "role": role, "source_sha256": role, "passage_ids": [role]})
        observations[role] = {"percent": "0.00" if role == "preliminary" else "0.99", "case_id": "A-583-856",
                              "entity_id": "great_grandeul_samoa", "rate_type": "AD_margin",
                              "review_period": "2023-2024", "document_id": doc, "passage_ids": [role]}
        if role == "earlier_operative_deposit":
            observations[role]["legal_effect"] = "final_deposit"
    spec = {"schema_version": "transition-spec-v1", "episode_id": "example", "event_channel_date": "2026-05-15",
            "documents": docs, "rate_candidates": [{"candidate_id": "great_grandeul", **observations}],
            "timing": {"earliest_public_verified": False}, "development_expected": {"delta": "review"}}
    pack = {"model": "typesafe/jev-1.13", "questions": {"delta": {"type": "choice", "instructions": "Compare evidence",
                                     "criteria": {"review": "Conflicting source", "unchanged": "Same"}}}}
    return spec, packets, pack


class TransitionsTests(unittest.TestCase):
    def test_scoped_passage_ids_are_losslessly_compressed(self):
        spec, packets, pack = fixture()
        sha, original_id = 'a' * 64, 'a' * 16 + ':p0001'
        packets['current']['episode_manifest']['source_sha256'] = sha
        packets['current']['current_source_passages_with_ids'][0]['passage_id'] = original_id
        spec['documents'][0].update(source_sha256=sha, passage_ids=[original_id])
        spec['rate_candidates'][0]['current']['passage_ids'] = [original_id]
        manifest = prepare_transition(spec, packets, pack)
        evidence = manifest['request']['state']
        document = evidence['documents'][0]
        self.assertEqual(document['passage_id_prefix'] + document['passages'][0]['passage_id'], original_id)
        self.assertEqual(evidence['rate_candidates'][0]['current']['passage_ids'], ['p0001'])
        self.assertEqual(spec['rate_candidates'][0]['current']['passage_ids'], [original_id])

    def test_naive_candidate_timestamp_rejected(self):
        spec, packets, pack = fixture()
        packets['current']['episode_manifest']['content_availability']['candidate_timestamp'] = '2026-05-15T08:45:00'
        with self.assertRaisesRegex(ValueError, 'timezone'):
            prepare_transition(spec, packets, pack)

    def test_guard_uses_arithmetic_not_mutable_expected_labels(self):
        spec, packets, pack = fixture()
        options = {name: name for name in ['increase', 'decrease', 'mixed', 'unchanged', 'unknown']}
        pack['questions'] = {'preliminary_margin_direction': {'type': 'choice', 'instructions': 'Compare', 'criteria': options}}
        spec['development_expected'] = {'preliminary_margin_direction': 'increase'}
        second = copy.deepcopy(spec['rate_candidates'][0])
        second['candidate_id'] = 'second'
        second['current']['percent'] = '0.00'
        second['preliminary']['percent'] = '0.99'
        spec['rate_candidates'].append(second)
        manifest = prepare_transition(spec, packets, pack)
        response = {'model': 'typesafe/jev-1.13', 'usage': {'input_tokens': 1, 'output_tokens': 1},
                    'answers': {'preliminary_margin_direction': {'type': 'choice', 'choice': 'increase',
                        'confidence': 0.9, 'probabilities': {name: 1.0 if name == 'increase' else 0.0 for name in options}}}}
        result = guard_transition(manifest, response)
        self.assertEqual(result['deterministic_preliminary_direction'], 'mixed')
        self.assertEqual(result['interpretation_findings'][0]['code'], 'model_numeric_direction_disagreement')
        self.assertEqual(result['raw_model_response']['answers'], response['answers'])
        self.assertFalse(result['expected_labels_used'])
        self.assertFalse(result['trade_eligible'])

    def test_guard_aggregate_missing_reference_remains_unknown(self):
        spec, packets, pack = fixture()
        spec['rate_candidates'][0].pop('preliminary')
        manifest = prepare_transition(spec, packets, pack)
        response = {'model': 'typesafe/jev-1.13', 'usage': {'input_tokens': 1, 'output_tokens': 1},
                    'answers': {'delta': {'type': 'choice', 'choice': 'review', 'confidence': 0.9,
                                         'probabilities': {'review': 0.9, 'unchanged': 0.1}}}}
        result = guard_transition(manifest, response)
        self.assertEqual(result['deterministic_preliminary_direction'], 'unknown')
        self.assertIn('earliest_public_time_unverified', result['economic_timing_blockers'])

    def test_preliminary_change_not_new_operative_rate(self):
        spec, _, _ = fixture()
        result = numeric_comparisons(spec["rate_candidates"])[0]
        self.assertEqual(result["comparisons"]["preliminary"]["delta_percentage_points"], "0.99")
        self.assertEqual(result["comparisons"]["earlier_operative_deposit"]["direction"], "unchanged")
        self.assertFalse(result["operative_history_complete"])

    def test_exact_decimal_and_missing_reference(self):
        spec, _, _ = fixture()
        row = spec["rate_candidates"][0]
        row["current"]["percent"] = "2.04"
        row["preliminary"]["percent"] = "0.01"
        row.pop("earlier_operative_deposit")
        result = numeric_comparisons([row])[0]["comparisons"]
        self.assertEqual(result["preliminary"]["delta_percentage_points"], "2.03")
        self.assertEqual(result["earlier_operative_deposit"]["status"], "unknown")

    def test_cross_case_entity_type_period_rejected(self):
        for field in ["case_id", "entity_id", "rate_type", "review_period"]:
            with self.subTest(field=field):
                spec, _, _ = fixture()
                spec["rate_candidates"][0]["preliminary"][field] = "different"
                with self.assertRaises(ValueError): numeric_comparisons(spec["rate_candidates"])

    def test_invalid_rate_rejected(self):
        for bad in ["NaN", "Infinity", "-1", 0.99, "1%"]:
            spec, _, _ = fixture()
            spec["rate_candidates"][0]["current"]["percent"] = bad
            with self.assertRaises(ValueError): numeric_comparisons(spec["rate_candidates"])

    def test_labels_not_sent_to_provider_and_inputs_unchanged(self):
        spec, packets, pack = fixture()
        before = copy.deepcopy(spec)
        result = prepare_transition(spec, packets, pack)
        self.assertEqual(spec, before)
        self.assertNotIn("expected", result["request"]["state"])
        self.assertFalse(result["trade_eligible"])

    def test_future_prior_rejected(self):
        spec, packets, pack = fixture()
        packets["preliminary"]["episode_manifest"]["content_availability"]["candidate_publication_date"] = "2026-05-15"
        with self.assertRaises(ValueError): prepare_transition(spec, packets, pack)

    def test_changed_source_rejected(self):
        spec, packets, pack = fixture()
        spec["documents"][0]["source_sha256"] = "changed"
        with self.assertRaises(ValueError): prepare_transition(spec, packets, pack)

    def test_omitted_rate_anchor_rejected(self):
        spec, packets, pack = fixture()
        spec["rate_candidates"][0]["current"]["passage_ids"] = ["missing"]
        with self.assertRaises(ValueError): prepare_transition(spec, packets, pack)

    def test_exposure_cannot_leak_same_day_or_future_source(self):
        spec, packets, pack = fixture()
        spec["issuer_exposures"] = [{"source_available_date": "2026-05-15", "source_url": "https://example.org"}]
        with self.assertRaises(ValueError): prepare_transition(spec, packets, pack)

    def test_public_inspection_can_precede_publication_with_caveats_preserved(self):
        spec, packets, pack = fixture()
        result = prepare_transition(spec, packets, pack)
        current = result["request"]["state"]["documents"][0]
        self.assertEqual(current["publication_date"], "2026-05-18")
        self.assertFalse(current["content_availability"]["content_version_matches_timestamp"])
        self.assertFalse(current["content_availability"]["earliest_public_verified"])
        self.assertIn("historically", current["content_availability"]["caveat"])

    def test_later_current_publication_requires_nonfuture_candidate_channel(self):
        for candidate in [None, "2026-05-16T08:45:00-04:00"]:
            spec, packets, pack = fixture()
            packets["current"]["episode_manifest"]["content_availability"]["candidate_timestamp"] = candidate
            with self.subTest(candidate=candidate), self.assertRaises(ValueError):
                prepare_transition(spec, packets, pack)

    def test_prior_candidate_channel_cannot_be_future_even_with_earlier_publication(self):
        spec, packets, pack = fixture()
        packets["preliminary"]["episode_manifest"]["content_availability"]["candidate_timestamp"] = "2026-05-16T08:45:00-04:00"
        with self.assertRaisesRegex(ValueError, "candidate channel"):
            prepare_transition(spec, packets, pack)

    def test_one_source_can_be_preliminary_and_provisional_operative_deposit(self):
        spec, packets, pack = fixture()
        spec["documents"] = spec["documents"][:2]
        spec["documents"][1].pop("role")
        spec["documents"][1]["roles"] = ["preliminary", "earlier_operative_deposit"]
        prior = copy.deepcopy(spec["rate_candidates"][0]["preliminary"])
        prior["legal_effect"] = "provisional_deposit"
        spec["rate_candidates"][0]["earlier_operative_deposit"] = prior
        result = prepare_transition(spec, packets, pack)
        comparison = result["deterministic_rate_comparisons"][0]["comparisons"]
        self.assertEqual(comparison["earlier_operative_deposit"]["delta_percentage_points"], "0.99")
        self.assertEqual(result["request"]["state"]["documents"][1]["roles"], ["preliminary", "earlier_operative_deposit"])
        self.assertFalse(result["request"]["state"]["operative_history_complete"])

    def test_operative_prior_requires_explicit_deposit_effect(self):
        spec, packets, pack = fixture()
        spec["rate_candidates"][0]["earlier_operative_deposit"].pop("legal_effect")
        with self.assertRaisesRegex(ValueError, "deposit effect"):
            prepare_transition(spec, packets, pack)

    def test_manifest_mutations_rejected_before_any_paid_call(self):
        for field, value in [("expected", {"delta": "unchanged"}),
                             ("expected", {}), ("model", "other-model"),
                             ("deterministic_rate_comparisons", [{"invented": 999}])]:
            spec, packets, pack = fixture()
            manifest = prepare_transition(spec, packets, pack)
            manifest[field] = value
            with TemporaryDirectory() as tmp, patch("jev_alpha.transitions.run_one") as paid:
                with self.subTest(field=field), self.assertRaises(ValueError):
                    run_transition(None, manifest, Path(tmp)/"result.json", live=True)
                paid.assert_not_called()

    def test_arithmetic_is_recomputed_even_when_comparison_hash_is_rewritten(self):
        spec, packets, pack = fixture()
        manifest = prepare_transition(spec, packets, pack)
        manifest["deterministic_rate_comparisons"][0]["comparisons"]["preliminary"]["delta_percentage_points"] = "999"
        manifest["deterministic_rate_comparisons_sha256"] = digest(manifest["deterministic_rate_comparisons"])
        with TemporaryDirectory() as tmp, self.assertRaisesRegex(ValueError, "comparisons"):
            run_transition(None, manifest, Path(tmp)/"result.json")

    def test_failed_and_skipped_arms_preserve_all_label_denominators(self):
        spec, packets, pack = fixture()
        manifest = prepare_transition(spec, packets, pack)
        failure = JevRequestError("Synthetic provider error")
        failure.response_blob_sha256 = "diagnostic"
        with TemporaryDirectory() as tmp, patch("jev_alpha.transitions.run_one", side_effect=failure) as paid:
            result = run_transition(None, manifest, Path(tmp)/"result.json", live=True, baseline=True)
            paid.assert_called_once()
            self.assertEqual(result["status"], "incomplete")
            self.assertEqual(result["requested_label_count"], 2)
            self.assertEqual(result["answered_label_count"], 0)
            self.assertEqual(result["label_accuracy_including_missing"], 0)
            self.assertEqual([r["status"] for r in result["runs"]], ["failed", "skipped_after_failure"])
            self.assertEqual(result["runs"][0]["diagnostic_sha256"], "diagnostic")
            self.assertEqual([r["total"] for r in result["runs"]], [1, 1])

    def test_completed_first_arm_survives_second_arm_failure(self):
        spec, packets, pack = fixture()
        manifest = prepare_transition(spec, packets, pack)
        success = {"response": {"model": "typesafe/jev-1.13", "usage": {"input_tokens": 1, "output_tokens": 1},
                   "answers": {"delta": {"type": "choice", "choice": "review", "confidence": .9,
                                         "probabilities": {"review": .9, "unchanged": .1}}}}}
        with TemporaryDirectory() as tmp, patch("jev_alpha.transitions.run_one", side_effect=[success, OSError("Synthetic failure")]):
            result = run_transition(None, manifest, Path(tmp)/"result.json", live=True, baseline=True)
            self.assertEqual(result["label_accuracy_including_missing"], .5)
            self.assertEqual(result["answered_label_count"], 1)
            self.assertEqual(result["runs"][0]["run"], success)

    def test_unexpected_scoring_failure_is_preserved_then_raised(self):
        spec, packets, pack = fixture()
        manifest = prepare_transition(spec, packets, pack)
        malformed = {"response": {"answers": {}}}
        with TemporaryDirectory() as tmp, patch("jev_alpha.transitions.run_one", return_value=malformed), patch("jev_alpha.transitions.guard_transition", side_effect=RuntimeError("Synthetic programming failure")):
            path = Path(tmp)/"result.json"
            with self.assertRaises(RuntimeError):
                run_transition(None, manifest, path, live=True, baseline=True)
            saved = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(saved["runs"][0]["run"], malformed)
            self.assertEqual(saved["runs"][1]["status"], "skipped_after_failure")

    def test_wrapped_research_state_recomputes_arithmetic_and_dry_run_denominator(self):
        spec, packets, pack = fixture()
        pack["state_policy"] = "Read supplied evidence only."
        manifest = prepare_transition(spec, packets, pack)
        with TemporaryDirectory() as tmp, patch("jev_alpha.transitions.run_one") as paid:
            result = run_transition(None, manifest, Path(tmp)/"result.json", baseline=True)
            paid.assert_not_called()
            self.assertEqual(result["requested_label_count"], 2)
            self.assertIsNone(result["label_accuracy_including_missing"])


if __name__ == "__main__": unittest.main()
