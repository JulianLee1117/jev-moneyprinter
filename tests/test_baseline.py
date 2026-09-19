"""Offline tests only. All completion payloads are synthetic fixtures."""

import io
import json
import os
import unittest
from unittest.mock import patch
import urllib.error

from jev_alpha.baseline import (BaselineClient, ENDPOINT, MODEL,
                                build_baseline_request, estimate_baseline)
from jev_alpha.jev import JevRequestError, JevValidationError, build_request


def request():
    return build_request({"passages": ["Synthetic evidence only."]}, {
        "model": "typesafe/jev-1.13", "questions": {
            "finality": {"type": "choice", "instructions": "Is this final?",
                         "criteria": {"final": "Operative", "unknown": "Insufficient evidence"}},
            "exception": {"type": "noul", "instructions": "Does an exception apply?"},
        }})


def response():
    return {
        "id": "synthetic-baseline-id", "model": "openai/gpt-5.4-mini-serving-build",
        "provider": "SyntheticProvider", "usage": {
            "prompt_tokens": 300, "completion_tokens": 150, "cost": 0.0009},
        "choices": [{"finish_reason": "stop", "message": {
            "role": "assistant", "content": json.dumps({"answers": {
                "finality": {"type": "choice", "choice": "unknown",
                             "probabilities": {"final": 0.2, "unknown": 0.8}},
                "exception": {"type": "noul", "noul": 0.35},
            }})}}],
    }


def network_response(mock_network, fixture):
    mock_network.return_value.open.return_value.__enter__.return_value.read.return_value = (
        json.dumps(fixture).encode("utf-8")
    )


class BaselineRequestTests(unittest.TestCase):
    def test_exact_evidence_questions_schema_and_no_network(self):
        original = request()
        with patch("jev_alpha.baseline.urllib.request.build_opener") as network:
            body = build_baseline_request(original)
        network.assert_not_called()
        self.assertEqual(MODEL, body["model"])
        user = json.loads(body["messages"][1]["content"])
        self.assertEqual(original["state"], user["state"])
        self.assertEqual(original["questions"], user["questions"])
        self.assertTrue(body["provider"]["require_parameters"])
        schema = body["response_format"]["json_schema"]
        self.assertTrue(schema["strict"])
        answers = schema["schema"]["properties"]["answers"]
        self.assertEqual(set(original["questions"]), set(answers["required"]))
        self.assertFalse(answers["additionalProperties"])
        self.assertEqual({"type", "choice", "probabilities"},
                         set(answers["properties"]["finality"]["required"]))
        user["state"]["passages"].append("Mutated")
        self.assertEqual(1, len(original["state"]["passages"]))

    def test_score_rejected_before_network(self):
        body = request()
        body["questions"]["score"] = {
            "type": "score", "instructions": "Strength?", "criteria": ["low", "high"]}
        with patch("jev_alpha.baseline.urllib.request.build_opener") as network:
            with self.assertRaises(JevValidationError):
                BaselineClient("synthetic-key").submit(body)
        network.assert_not_called()

    def test_estimate_prices_bytes_and_reserves_output(self):
        estimate = estimate_baseline(request())
        self.assertEqual(2048, estimate["reserved_output_tokens"])
        self.assertEqual(estimate["request_bytes"] + 1024,
                         estimate["conservative_input_tokens"])
        self.assertAlmostEqual((estimate["conservative_input_tokens"] * 0.75
                                + 2048 * 4.5) / 1_000_000,
                               estimate["estimated_cost_usd"])
        self.assertFalse(estimate["is_exact"])


class BaselineTransportTests(unittest.TestCase):
    def test_normalization_preserves_raw_and_actual_model(self):
        fixture = response()
        with patch("jev_alpha.baseline.urllib.request.build_opener") as network:
            network_response(network, fixture)
            result = BaselineClient("synthetic-key").submit(request())
            submitted = network.return_value.open.call_args.args[0]
        self.assertEqual(ENDPOINT, submitted.full_url)
        self.assertEqual("POST", submitted.get_method())
        self.assertEqual(fixture["model"], result["model"])
        self.assertEqual(fixture, result["raw_response"])
        self.assertEqual({"input_tokens": 300, "output_tokens": 150, "cost": 0.0009},
                         result["usage"])
        self.assertEqual("self_reported_chat_output", result["probability_origin"])

    def test_budget_and_missing_key_prevent_network(self):
        with patch("jev_alpha.baseline.urllib.request.build_opener") as network:
            with self.assertRaises(JevValidationError):
                BaselineClient("synthetic-key", max_estimated_cost_usd=0).submit(request())
            with patch.dict(os.environ, {}, clear=True), self.assertRaises(JevRequestError):
                BaselineClient().submit(request())
            with self.assertRaises(JevRequestError):
                BaselineClient("bad\nkey").submit(request())
        network.assert_not_called()

    def test_missing_usage_not_fabricated(self):
        fixture = response()
        fixture.pop("usage")
        with patch("jev_alpha.baseline.urllib.request.build_opener") as network:
            network_response(network, fixture)
            with self.assertRaises(JevRequestError) as caught:
                BaselineClient("synthetic-key").submit(request())
        self.assertTrue(caught.exception.outcome_uncertain)
        network.return_value.open.assert_called_once()

    def test_truncated_or_refused_response_is_rejected(self):
        for kind in ("length", "refusal"):
            fixture = response()
            if kind == "length":
                fixture["choices"][0]["finish_reason"] = "length"
            else:
                fixture["choices"][0]["message"]["refusal"] = "synthetic refusal"
            with self.subTest(kind=kind), patch("jev_alpha.baseline.urllib.request.build_opener") as network:
                network_response(network, fixture)
                with self.assertRaises(JevRequestError):
                    BaselineClient("synthetic-key").submit(request())
                network.return_value.open.assert_called_once()

    def test_probability_and_schema_failures_are_not_repaired(self):
        for mutation in (
            lambda a: a["finality"]["probabilities"].update(final=0.5),
            lambda a: a["finality"].update(choice="final"),
            lambda a: a["exception"].update(noul=True),
            lambda a: a["exception"].update(explanation="extra"),
            lambda a: a.pop("exception"),
        ):
            fixture = response()
            content = json.loads(fixture["choices"][0]["message"]["content"])
            mutation(content["answers"])
            fixture["choices"][0]["message"]["content"] = json.dumps(content)
            with patch("jev_alpha.baseline.urllib.request.build_opener") as network:
                network_response(network, fixture)
                with self.assertRaises(JevRequestError):
                    BaselineClient("synthetic-key").submit(request())

    def test_duplicate_json_keys_are_rejected(self):
        fixture = response()
        fixture["choices"][0]["message"]["content"] = '{"answers":{},"answers":{}}'
        with patch("jev_alpha.baseline.urllib.request.build_opener") as network:
            network_response(network, fixture)
            with self.assertRaises(JevRequestError):
                BaselineClient("synthetic-key").submit(request())

    def test_http_body_redacted_no_retry_and_redirects_disabled(self):
        error = urllib.error.HTTPError(ENDPOINT, 429, "secret marker", {},
                                       io.BytesIO(b"secret marker body"))
        with patch("jev_alpha.baseline.urllib.request.build_opener") as network:
            network.return_value.open.side_effect = error
            with self.assertRaises(JevRequestError) as caught:
                BaselineClient("synthetic-key").submit(request())
            handler = network.call_args.args[0]
            self.assertIsNone(handler.redirect_request(None, None, 302, "", {}, "https://evil.invalid"))
            network.return_value.open.assert_called_once()
        self.assertEqual(429, caught.exception.status)
        self.assertTrue(caught.exception.outcome_uncertain)
        self.assertNotIn("secret marker", str(caught.exception))
        self.assertIsNone(caught.exception.response_data)
        self.assertIsNone(caught.exception.validation_reason)

    def test_validation_failure_preserves_raw_cost_and_fixed_reason(self):
        fixture = response()
        fixture["choices"][0]["finish_reason"] = "length"
        with patch("jev_alpha.baseline.urllib.request.build_opener") as network:
            network_response(network, fixture)
            with self.assertRaises(JevRequestError) as caught:
                BaselineClient("synthetic-key").submit(request())
            network.return_value.open.assert_called_once()
        error = caught.exception
        self.assertTrue(error.outcome_uncertain)
        self.assertEqual(fixture, error.response_data)
        self.assertEqual(0.0009, error.response_data["usage"]["cost"])
        self.assertEqual("Baseline completion was not complete.", error.validation_reason)
        self.assertNotIn("synthetic-key", str(error))

    def test_invalid_inner_json_preserves_envelope_without_decoder_text(self):
        fixture = response()
        fixture["choices"][0]["message"]["content"] = '{"answers": "synthetic remote marker'
        with patch("jev_alpha.baseline.urllib.request.build_opener") as network:
            network_response(network, fixture)
            with self.assertRaises(JevRequestError) as caught:
                BaselineClient("synthetic-key").submit(request())
        self.assertEqual(fixture, caught.exception.response_data)
        self.assertEqual("Baseline completion could not be normalized.",
                         caught.exception.validation_reason)
        self.assertNotIn("synthetic remote marker", str(caught.exception))

    def test_diagnostics_exclude_provider_errors_nonfinite_and_credential_echoes(self):
        key = 'synthetic-"secret\\key'
        for kind in ("error", "outer_nan", "inner_nan", "echo", "inner_echo",
                     "inner_array_echo", "truncated_echo"):
            fixture = response()
            fixture["choices"][0]["finish_reason"] = "length"
            if kind == "error":
                fixture["error"] = {"message": "remote error body"}
            elif kind == "outer_nan":
                fixture["usage"]["cost"] = float("nan")
            elif kind == "inner_nan":
                fixture["choices"][0]["message"]["content"] = '{"answers":{"bad":NaN}}'
            elif kind == "echo":
                fixture["echo"] = key
            elif kind == "inner_array_echo":
                fixture["choices"][0]["message"]["content"] = json.dumps([key])
            elif kind == "truncated_echo":
                fixture["choices"][0]["message"]["content"] = json.dumps({"echo": key})[:-1]
            else:
                fixture["choices"][0]["message"]["content"] = json.dumps({"echo": key})
            with self.subTest(kind=kind), patch("jev_alpha.baseline.urllib.request.build_opener") as network:
                network_response(network, fixture)
                with self.assertRaises(JevRequestError) as caught:
                    BaselineClient(key).submit(request())
                self.assertIsNone(caught.exception.response_data)
                self.assertNotIn(key, str(caught.exception))
                self.assertNotIn("remote error body", str(caught.exception))
                network.return_value.open.assert_called_once()

    def test_malformed_outer_json_has_no_diagnostic(self):
        with patch("jev_alpha.baseline.urllib.request.build_opener") as network:
            network.return_value.open.return_value.__enter__.return_value.read.return_value = b'{"usage":'
            with self.assertRaises(JevRequestError) as caught:
                BaselineClient("synthetic-key").submit(request())
        self.assertIsNone(caught.exception.response_data)
        self.assertIsNone(caught.exception.validation_reason)


if __name__ == "__main__":
    unittest.main()
