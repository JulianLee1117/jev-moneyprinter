"""Offline schema and billing-boundary tests. All model outputs are synthetic.

Fixture shapes follow the official OpenRouter Decisions OpenAPI and TypeSafe's
Score documentation, checked 2026-09-18. No fixture is an actual inference result.
"""

import json
import os
from pathlib import Path
import unittest
from unittest.mock import Mock, patch
import urllib.error

from jev_alpha.jev import (ENDPOINT, MODEL, JevClient, JevRequestError,
                           JevValidationError, build_request, estimate_request,
                           validate_response)


def pack():
    return {"model": MODEL, "question_count": 3,
            "state_policy": "Use supplied evidence only.",
            "required_state_sections": ["passages"],
            "questions": {
                "change": {"type": "choice", "instructions": "What changed?",
                           "criteria": {"new": "Changed", "old": "Previously known", "unknown": None}},
                "ambiguous": {"type": "noul", "instructions": "Is applicability ambiguous?"},
                "strength": {"type": "score", "instructions": "How strong is the evidence?",
                             "criteria": ["Absent", "Indirect", "Direct"]}}}


def response():
    return {"id": "synthetic-test-id", "model": "typesafe/jev-1.13-20260917",
            "provider": "TypeSafe", "usage": {"input_tokens": 1200, "output_tokens": 37,
                                               "cost": 0.0000504},
            "answers": {
                "change": {"type": "choice", "choice": "new", "confidence": 0.6,
                           "probabilities": {"new": 0.7, "old": 0.2, "unknown": 0.1}},
                "ambiguous": {"type": "noul", "noul": 0.35},
                "strength": {"type": "score", "score": 1.2, "confidence": 0.2,
                             "probabilities": {"0": 0.1, "1": 0.6, "2": 0.3},
                             "legend": {"0": "Absent", "1": "Indirect", "2": "Direct"}}}}


class RequestTests(unittest.TestCase):
    def test_real_pack_builds_all_28_questions_without_io(self):
        path = Path(__file__).resolve().parents[1] / "research/jev-core-question-pack.v1.json"
        question_pack = json.loads(path.read_text(encoding="utf-8"))
        state = {name: [] for name in question_pack["required_state_sections"]}
        with patch("jev_alpha.jev.urllib.request.build_opener") as network:
            request = build_request(state, question_pack)
        network.assert_not_called()
        self.assertEqual(28, len(request["questions"]))
        self.assertEqual(state, request["state"]["evidence"])
        self.assertLess(estimate_request(request)["conservative_input_tokens"], 28_000)

    def test_subset_policy_and_copies(self):
        original_pack, state = pack(), {"passages": ["Evidence"]}
        request = build_request(state, original_pack, ["ambiguous"])
        self.assertEqual(["ambiguous"], list(request["questions"]))
        self.assertIn("research_policy", request["questions"]["ambiguous"]["instructions"])
        request["state"]["evidence"]["passages"].append("Other")
        self.assertEqual(["Evidence"], state["passages"])
        self.assertEqual(pack(), original_pack)

    def test_missing_sections_bad_count_and_bad_selections(self):
        with self.assertRaises(JevValidationError):
            build_request({}, pack())
        changed = pack()
        changed["question_count"] = 4
        with self.assertRaises(JevValidationError):
            build_request({"passages": []}, changed)
        for selection in ([], ["missing"], ["change", "change"], "change"):
            with self.subTest(selection=selection), self.assertRaises(JevValidationError):
                build_request({"passages": []}, pack(), selection)

    def test_malformed_question_and_nonfinite_inputs(self):
        bad_questions = [
            {"type": "noul", "instructions": "?", "criteria": {"true": "Yes"}},
            {"type": "choice", "instructions": "?", "criteria": {"yes": "One option"}},
            {"type": "score", "instructions": "?", "criteria": ["Only one"]},
            {"type": "score", "instructions": "?", "criteria": ["a", "b"], "temperature": 0},
            {"type": "text", "instructions": "?"},
        ]
        for question in bad_questions:
            with self.subTest(question=question), self.assertRaises(JevValidationError):
                build_request({}, {"model": MODEL, "questions": {"q": question}})
        for value in (float("nan"), float("inf"), {1: "non-string key"}):
            with self.subTest(value=value), self.assertRaises(JevValidationError):
                build_request({"passages": value}, pack())

    def test_estimate_counts_utf8_and_does_not_claim_exact_billing(self):
        request = build_request({"passages": ["鋼板"]}, pack())
        estimate = estimate_request(request)
        self.assertFalse(estimate["is_exact"])
        self.assertGreater(estimate["conservative_input_tokens"], estimate["request_bytes"])
        self.assertGreaterEqual(estimate["estimated_billable_input_tokens"], estimate["conservative_input_tokens"])
        self.assertGreater(estimate["estimated_cost_usd"], 0)


class ResponseTests(unittest.TestCase):
    def setUp(self):
        self.request = build_request({"passages": []}, pack())

    def test_preserves_full_distributions_usage_and_serving_model(self):
        fixture = response()
        validated = validate_response(fixture, self.request)
        self.assertEqual(fixture, validated)
        validated["answers"]["change"]["probabilities"]["new"] = 0
        self.assertEqual(0.7, fixture["answers"]["change"]["probabilities"]["new"])

    def test_observed_two_decimal_choice_mass_is_preserved_with_delta(self):
        probabilities = {"historical_or_context": 0.23, "operative_change": 0.28,
                         "procedural": 0.33, "scope_or_exception": 0.01, "unclear": 0.14}
        self.request["questions"]["change"]["criteria"] = {key: key for key in probabilities}
        fixture = response()
        fixture["answers"]["change"]["probabilities"] = probabilities
        fixture["answers"]["change"]["choice"] = "procedural"
        validated = validate_response(fixture, self.request)
        answer = validated["answers"]["change"]
        self.assertEqual(probabilities, answer["probabilities"])
        self.assertAlmostEqual(answer["probability_mass_delta"], -0.01)
        self.assertNotIn("probability_mass_delta", fixture["answers"]["change"])
        self.assertEqual(validated, validate_response(validated, self.request))

    def test_choice_rounding_tolerance_depends_on_option_count_and_is_capped(self):
        cases = [({"a": 0.5, "b": 0.49}, True),
                 ({"a": 0.5, "b": 0.48}, False),
                 ({"a": 0.5, "b": 0.4}, False),
                 ({"a": 0.5, "b": 0.48, "c": 0}, False),
                 ({"a": 1, "b": 0.02, "c": 0, "d": 0}, True),
                 ({"a": 0.97, "b": 0, "c": 0, "d": 0, "e": 0, "f": 0}, False),
                 ({"a": 0.501, "b": 0.49}, False),
                 ({"a": 0, "b": 0}, False)]
        for probabilities, accepted in cases:
            with self.subTest(probabilities=probabilities):
                self.request["questions"]["change"]["criteria"] = {key: key for key in probabilities}
                fixture = response()
                fixture["answers"]["change"]["probabilities"] = probabilities
                fixture["answers"]["change"]["choice"] = "a"
                if accepted:
                    validated = validate_response(fixture, self.request)
                    self.assertEqual(probabilities, validated["answers"]["change"]["probabilities"])
                else:
                    with self.assertRaises(JevValidationError):
                        validate_response(fixture, self.request)

    def test_score_distribution_still_requires_unit_mass(self):
        fixture = response()
        fixture["answers"]["strength"]["probabilities"]["2"] = 0.29
        with self.assertRaises(JevValidationError):
            validate_response(fixture, self.request)

    def test_rejects_missing_or_extra_answers_and_distributions(self):
        for mutation in (lambda r: r["answers"].pop("change"),
                         lambda r: r["answers"].update(extra={"type": "noul", "noul": 0.5}),
                         lambda r: r["answers"]["change"].pop("probabilities"),
                         lambda r: r["answers"]["change"]["probabilities"].pop("unknown")):
            fixture = response()
            mutation(fixture)
            with self.assertRaises(JevValidationError):
                validate_response(fixture, self.request)

    def test_rejects_invalid_probability_values_without_normalizing(self):
        for value in (-0.1, 1.1, True, "0.7", float("nan"), float("inf"), 0.8):
            fixture = response()
            fixture["answers"]["change"]["probabilities"]["new"] = value
            with self.subTest(value=value), self.assertRaises(JevValidationError):
                validate_response(fixture, self.request)

    def test_choice_winner_noul_and_score_semantics(self):
        for question, field, value in (("change", "choice", "old"),
                                      ("change", "choice", "invented"),
                                      ("ambiguous", "noul", True),
                                      ("ambiguous", "noul", 1.1),
                                      ("strength", "score", 0.2),
                                      ("strength", "score", 3),
                                      ("strength", "legend", {"0": "Wrong"})):
            fixture = response()
            fixture["answers"][question][field] = value
            with self.subTest(question=question, field=field), self.assertRaises(JevValidationError):
                validate_response(fixture, self.request)

    def test_requires_finite_snake_case_usage(self):
        for usage in ({"inputTokens": 100, "outputTokens": 10},
                      {"input_tokens": True, "output_tokens": 0},
                      {"input_tokens": 100, "output_tokens": -1},
                      {"input_tokens": 100, "output_tokens": 0, "cost": 10 ** 400},
                      {"input_tokens": 100, "output_tokens": 0, "cost": float("nan")}):
            fixture = response()
            fixture["usage"] = usage
            with self.subTest(usage=usage), self.assertRaises(JevValidationError):
                validate_response(fixture, self.request)


class ClientTests(unittest.TestCase):
    def setUp(self):
        self.request = build_request({"passages": []}, pack())

    @patch("jev_alpha.jev.urllib.request.build_opener")
    def test_explicit_submission_has_key_only_in_header(self, opener):
        remote = opener.return_value.open.return_value.__enter__.return_value
        remote.read.return_value = json.dumps(response()).encode()
        client = JevClient(api_key="synthetic-test-key")
        result = client.submit(self.request)
        call = opener.return_value.open.call_args
        sent = call.args[0]
        self.assertEqual(ENDPOINT, sent.full_url)
        self.assertEqual("POST", sent.method)
        self.assertEqual("Bearer synthetic-test-key", sent.get_header("Authorization"))
        self.assertNotIn(b"synthetic-test-key", sent.data)
        self.assertEqual(30, call.kwargs["timeout"])
        self.assertEqual(response(), result)
        opener.return_value.open.assert_called_once()

    @patch("jev_alpha.jev.urllib.request.build_opener")
    def test_injected_transport_receives_separate_key_and_uses_shared_validation(self, opener):
        transport = Mock()
        transport.post.return_value = json.dumps(response()).encode()
        client = JevClient(api_key="synthetic-test-key", timeout=12, transport=transport)
        self.assertEqual(response(), client.submit(self.request))
        call = transport.post.call_args
        self.assertEqual(ENDPOINT, call.args[0])
        self.assertEqual(self.request, json.loads(call.args[1]))
        self.assertNotIn(b"synthetic-test-key", call.args[1])
        self.assertEqual("synthetic-test-key", call.args[2])
        self.assertEqual({"timeout": 12}, call.kwargs)
        opener.assert_not_called()
        transport.post.assert_called_once()

    def test_injected_transport_does_not_bypass_request_budget_or_key_checks(self):
        transport = Mock()
        for client in (JevClient(transport=transport, max_estimated_cost_usd=0),
                       JevClient(transport=transport, max_input_tokens=1)):
            with self.assertRaises(JevValidationError):
                client.submit(self.request)
        with patch.dict(os.environ, {}, clear=True), self.assertRaises(JevRequestError):
            JevClient(transport=transport).submit(self.request)
        with self.assertRaises(JevValidationError):
            JevClient(api_key="synthetic-test-key", transport=transport).submit({})
        with self.assertRaises(JevRequestError):
            JevClient(api_key="invalid\nkey", transport=transport).submit(self.request)
        transport.post.assert_not_called()

    def test_injected_transport_response_still_requires_valid_probabilities(self):
        transport = Mock()
        fixture = response()
        fixture["answers"]["change"]["probabilities"]["new"] = 0.8
        transport.post.return_value = json.dumps(fixture).encode()
        with self.assertRaises(JevRequestError) as failure:
            JevClient(api_key="synthetic-test-key", transport=transport).submit(self.request)
        self.assertEqual(fixture, failure.exception.response_data)
        self.assertEqual("Answer probabilities must be finite, bounded and sum to one.",
                         failure.exception.validation_reason)
        transport.post.assert_called_once()

    def test_injected_transport_must_return_bytes_and_respect_response_limit(self):
        for raw in ("synthetic-test-key", b"x" * 2_000_001):
            transport = Mock()
            transport.post.return_value = raw
            with self.assertRaises(JevRequestError) as failure:
                JevClient(api_key="synthetic-test-key", transport=transport).submit(self.request)
            self.assertNotIn("synthetic-test-key", str(failure.exception))
            self.assertIsNone(failure.exception.response_data)
            transport.post.assert_called_once()

    def test_injected_transport_error_preserves_status_and_redacts_remote_details(self):
        from jev_alpha.transport import TransportError

        for status, uncertain in ((429, True), (None, False)):
            transport = Mock()
            transport.post.side_effect = TransportError(
                "synthetic-test-key private remote body", status=status,
                outcome_uncertain=uncertain)
            with self.assertRaises(JevRequestError) as failure:
                JevClient(api_key="synthetic-test-key", transport=transport).submit(self.request)
            self.assertEqual(status, failure.exception.status)
            self.assertEqual(uncertain, failure.exception.outcome_uncertain)
            self.assertNotIn("synthetic-test-key", str(failure.exception))
            self.assertNotIn("private remote body", str(failure.exception))
            self.assertIsNone(failure.exception.response_data)
            self.assertIsNone(failure.exception.validation_reason)
            self.assertIsNone(failure.exception.__cause__)
            transport.post.assert_called_once()

    @patch("jev_alpha.jev.urllib.request.build_opener")
    def test_budget_context_and_missing_key_fail_before_network(self, opener):
        for client in (JevClient(max_estimated_cost_usd=0), JevClient(max_input_tokens=1)):
            with self.assertRaises(JevValidationError):
                client.submit(self.request)
        with patch.dict(os.environ, {}, clear=True), self.assertRaises(JevRequestError):
            JevClient().submit(self.request)
        opener.assert_not_called()

    @patch("jev_alpha.jev.urllib.request.build_opener")
    def test_timeout_is_not_retried_and_error_is_redacted(self, opener):
        opener.return_value.open.side_effect = urllib.error.URLError("synthetic-test-key")
        with self.assertRaises(JevRequestError) as failure:
            JevClient(api_key="synthetic-test-key").submit(self.request)
        self.assertTrue(failure.exception.outcome_uncertain)
        self.assertNotIn("synthetic-test-key", str(failure.exception))
        self.assertIsNone(failure.exception.__cause__)
        opener.return_value.open.assert_called_once()

    @patch("jev_alpha.jev.urllib.request.build_opener")
    def test_http_429_is_reported_without_retry_or_provider_body(self, opener):
        opener.return_value.open.side_effect = urllib.error.HTTPError(
            ENDPOINT, 429, "sensitive remote body", {}, None)
        with self.assertRaises(JevRequestError) as failure:
            JevClient(api_key="synthetic-test-key").submit(self.request)
        self.assertEqual(429, failure.exception.status)
        self.assertNotIn("sensitive", str(failure.exception))
        self.assertIsNone(failure.exception.response_data)
        self.assertIsNone(failure.exception.validation_reason)
        opener.return_value.open.assert_called_once()

    @patch("jev_alpha.jev.urllib.request.build_opener")
    def test_invalid_paid_response_is_not_retried(self, opener):
        remote = opener.return_value.open.return_value.__enter__.return_value
        remote.read.return_value = b'{"error":"synthetic-test-key"}'
        with self.assertRaises(JevRequestError) as failure:
            JevClient(api_key="synthetic-test-key").submit(self.request)
        self.assertTrue(failure.exception.outcome_uncertain)
        self.assertNotIn("synthetic-test-key", str(failure.exception))
        self.assertIsNone(failure.exception.response_data)
        self.assertEqual("Provider returned an error response.", failure.exception.validation_reason)
        opener.return_value.open.assert_called_once()

    @patch("jev_alpha.jev.urllib.request.build_opener")
    def test_invalid_distribution_retains_finite_response_and_safe_reason(self, opener):
        remote = opener.return_value.open.return_value.__enter__.return_value
        fixture = response()
        fixture["answers"]["change"]["probabilities"]["new"] = 0.8
        remote.read.return_value = json.dumps(fixture).encode()
        with self.assertRaises(JevRequestError) as failure:
            JevClient(api_key="synthetic-test-key").submit(self.request)
        error = failure.exception
        self.assertEqual(fixture, error.response_data)
        self.assertEqual("Answer probabilities must be finite, bounded and sum to one.", error.validation_reason)
        self.assertIn(error.validation_reason, str(error))
        self.assertNotIn("synthetic-test-key", json.dumps(error.response_data))
        self.assertNotIn("synthetic-test-key", str(error))
        self.assertTrue(error.outcome_uncertain)
        self.assertIsNone(error.__cause__)
        opener.return_value.open.assert_called_once()

    @patch("jev_alpha.jev.urllib.request.build_opener")
    def test_malformed_json_has_no_preserved_response_or_remote_parse_message(self, opener):
        remote = opener.return_value.open.return_value.__enter__.return_value
        remote.read.return_value = b'{"private_field": "synthetic-test-key"'
        with self.assertRaises(JevRequestError) as failure:
            JevClient(api_key="synthetic-test-key").submit(self.request)
        self.assertIsNone(failure.exception.response_data)
        self.assertIsNone(failure.exception.validation_reason)
        self.assertNotIn("private_field", str(failure.exception))
        self.assertNotIn("synthetic-test-key", str(failure.exception))
        opener.return_value.open.assert_called_once()

    @patch("jev_alpha.jev.urllib.request.build_opener")
    def test_nonfinite_json_is_not_preserved(self, opener):
        remote = opener.return_value.open.return_value.__enter__.return_value
        fixture = response()
        fixture["answers"]["ambiguous"]["noul"] = float("nan")
        remote.read.return_value = json.dumps(fixture).encode()
        with self.assertRaises(JevRequestError) as failure:
            JevClient(api_key="synthetic-test-key").submit(self.request)
        self.assertIsNone(failure.exception.response_data)
        self.assertEqual("Data must be finite, acyclic UTF-8 JSON.", failure.exception.validation_reason)
        opener.return_value.open.assert_called_once()

    @patch("jev_alpha.jev.urllib.request.build_opener")
    def test_credential_echo_in_parsed_invalid_body_is_not_preserved(self, opener):
        remote = opener.return_value.open.return_value.__enter__.return_value
        fixture = response()
        key = 'synthetic-"test\\key'
        fixture["unexpected_metadata"] = {"nested": "Bearer " + key}
        fixture["answers"]["change"].pop("probabilities")
        remote.read.return_value = json.dumps(fixture).encode()
        with self.assertRaises(JevRequestError) as failure:
            JevClient(api_key=key).submit(self.request)
        self.assertIsNone(failure.exception.response_data)
        self.assertEqual("Answer must preserve the complete probability distribution.",
                         failure.exception.validation_reason)
        self.assertNotIn(key, str(failure.exception))
        opener.return_value.open.assert_called_once()

    @patch("jev_alpha.jev.urllib.request.build_opener")
    def test_duplicate_keys_in_paid_response_are_rejected(self, opener):
        remote = opener.return_value.open.return_value.__enter__.return_value
        body = json.dumps(response())
        remote.read.return_value = body.replace('"noul": 0.35', '"noul": 0.1, "noul": 0.35').encode()
        with self.assertRaises(JevRequestError) as failure:
            JevClient(api_key="synthetic-test-key").submit(self.request)
        self.assertIsNone(failure.exception.response_data)
        self.assertIsNone(failure.exception.validation_reason)
        opener.return_value.open.assert_called_once()


if __name__ == "__main__":
    unittest.main()
