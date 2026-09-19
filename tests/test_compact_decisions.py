import copy
import json
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

from jev_alpha.compact_decisions import (MODEL, build_compact_request, estimate_compact,
                                         run_compact, validate_compact_response)
from jev_alpha.jev import JevRequestError, JevValidationError, build_request
from jev_alpha.store import Store, canonical_json
from jev_alpha.transport import MAX_RESPONSE_BYTES, TransportError


KEY = "synthetic-private-test-key"


def request():
    return build_request({"passages": [{"id": "p1", "text": "Synthetic source. 😀"}],
                          "as_of": "2026-09-18"}, {
        "model": "typesafe/jev-1.13", "state_policy": "Use the dated source only.",
        "questions": {
            "direction": {"type": "choice", "instructions": "Apply the reported comparison.",
                          "criteria": {"increase": "Explicit increase", "decrease": "Explicit decrease",
                                       "unknown": "Insufficient comparison"}},
            "scope": {"type": "choice", "instructions": {"compare": ["whole", "part"]},
                      "criteria": {"company": "Entire company", "segment": "Specified segment",
                                   "unspecified": None}},
        }})


def raw_response(answers=None):
    return {"model": MODEL, "id": "synthetic-completion", "provider": "synthetic-provider",
            "choices": [{"finish_reason": "stop", "message": {
                "content": json.dumps(answers or {"direction": "unknown", "scope": "segment"})}}],
            "usage": {"prompt_tokens": 200, "completion_tokens": 30, "cost": .002}}


def transport(raw=None):
    return Mock(post=Mock(return_value=json.dumps(raw if raw is not None else raw_response()).encode()))


class CompactRequestTests(unittest.TestCase):
    def test_same_evidence_question_instructions_and_every_criterion(self):
        original = request()
        before = copy.deepcopy(original)
        body = build_compact_request(original)
        supplied = json.loads(body["messages"][1]["content"])
        self.assertEqual(supplied, {"state": before["state"], "questions": before["questions"]})
        self.assertEqual(original, before)
        self.assertEqual(body["model"], MODEL)
        self.assertEqual(body["max_tokens"], 2048)
        self.assertTrue(body["provider"]["require_parameters"])
        schema = body["response_format"]["json_schema"]["schema"]
        self.assertEqual(set(schema["required"]), set(before["questions"]))
        self.assertFalse(schema["additionalProperties"])
        for name, question in before["questions"].items():
            self.assertEqual(schema["properties"][name],
                             {"type": "string", "enum": list(question["criteria"])})
        supplied["state"]["evidence"]["passages"][0]["text"] = "Changed"
        self.assertEqual(original, before)
        self.assertNotIn("operating", body["messages"][0]["content"].lower())

    def test_non_choice_questions_rejected_before_credentials_or_attempt(self):
        with TemporaryDirectory() as tmp, Store(tmp) as store, patch("jev_alpha.compact_decisions.read_key") as key:
            for kind, criteria in (("noul", {"true": "yes", "false": "no"}),
                                   ("score", ["low", "high"])):
                req = request()
                req["questions"]["scope"] = {"type": kind, "instructions": "Synthetic", "criteria": criteria}
                client = transport()
                with self.subTest(kind=kind), self.assertRaises(JevValidationError):
                    run_compact(store, req, transport=client)
                client.post.assert_not_called()
            key.assert_not_called()
            self.assertEqual(store.db.execute("SELECT COUNT(*) FROM model_attempts").fetchone()[0], 0)

    def test_estimate_counts_full_schema_and_output_reservation(self):
        body = build_compact_request(request())
        estimate = estimate_compact(request())
        self.assertEqual(estimate["request_bytes"], len(canonical_json(body)))
        self.assertEqual(estimate["reserved_output_tokens"], 2048)
        self.assertAlmostEqual(estimate["estimated_cost_usd"],
                               ((estimate["request_bytes"] + 1024) * .75 + 2048 * 4.5) / 1_000_000)
        self.assertFalse(estimate["is_exact"])


class CompactRunTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.store = Store(self.temp.name)
        self.addCleanup(self.store.db.close)
        key = patch("jev_alpha.compact_decisions.read_key", return_value=KEY)
        self.key = key.start()
        self.addCleanup(key.stop)

    def test_exact_request_cache_preserves_raw_and_has_no_probabilities(self):
        raw = raw_response()
        raw["model"] = MODEL + "-2026-03-17"
        client = transport(raw)
        req = request()
        first = run_compact(self.store, req, transport=client)
        self.key.return_value = None  # Reading a successful cache requires no key.
        second = run_compact(self.store, req, max_cost_usd=0, transport=client)
        self.assertFalse(first["cached"])
        self.assertTrue(second["cached"])
        self.assertEqual(first["request_hash"], second["request_hash"])
        self.assertEqual(second["incremental_cost_usd"], 0)
        self.assertEqual(first["response"]["raw_response"], raw)
        self.assertEqual(first["response"]["serving_model"], raw["model"])
        self.assertEqual(first["response"]["answers"], {"direction": "unknown", "scope": "segment"})
        self.assertEqual(first["response"]["probability_origin"], "none_categorical_chat")
        self.assertEqual(first["response"]["usage"]["cost"], .002)
        self.assertEqual(client.post.call_count, 1)
        self.assertEqual(json.loads(client.post.call_args.args[1]), build_compact_request(req))

    def test_cache_identity_changes_with_criteria_guidance_and_evidence(self):
        client = transport()
        req = request()
        hashes = [run_compact(self.store, req, transport=client)["request_hash"]]
        req["questions"]["direction"]["criteria"]["unknown"] = "Different decision definition"
        hashes.append(run_compact(self.store, req, transport=client)["request_hash"])
        req["state"]["evidence"]["passages"][0]["text"] += " More evidence."
        hashes.append(run_compact(self.store, req, transport=client)["request_hash"])
        self.assertEqual(len(set(hashes)), 3)
        self.assertEqual(client.post.call_count, 3)

    def test_all_labels_required_and_only_exact_allowed_categories(self):
        invalid = [{"direction": "unknown"},
                   {"direction": "unknown", "scope": "segment", "extra": "unknown"},
                   {"direction": "Unknown", "scope": "segment"},
                   {"direction": True, "scope": "segment"},
                   {"direction": {"choice": "unknown", "probability": .5}, "scope": "segment"}]
        for index, answers in enumerate(invalid):
            req = request()
            req["state"]["fixture"] = index
            with self.subTest(answers=answers), self.assertRaises(JevRequestError):
                run_compact(self.store, req, transport=transport(raw_response(answers)))

    def test_wrong_model_refusal_truncation_duplicates_and_usage_rejected(self):
        fixtures = []
        for model in ("openai/other-model", MODEL + "-unexpected", MODEL + "-2026-99-99", None):
            raw = raw_response()
            raw["model"] = model
            fixtures.append(raw)
        for kind in ("length", "refusal", "tool", "missing_usage", "bool_tokens", "nan_cost", "duplicate"):
            raw = raw_response()
            if kind == "length":
                raw["choices"][0]["finish_reason"] = "length"
            elif kind == "refusal":
                raw["choices"][0]["message"]["refusal"] = "Synthetic refusal"
            elif kind == "tool":
                raw["choices"][0]["message"]["tool_calls"] = [{}]
            elif kind == "missing_usage":
                raw.pop("usage")
            elif kind == "bool_tokens":
                raw["usage"]["prompt_tokens"] = True
            elif kind == "nan_cost":
                raw["usage"]["cost"] = float("nan")
            else:
                raw["choices"][0]["message"]["content"] = '{"direction":"unknown","scope":"segment","scope":"company"}'
            fixtures.append(raw)
        for index, raw in enumerate(fixtures):
            req = request()
            req["state"]["fixture"] = index
            client = transport(raw)
            with self.subTest(index=index), self.assertRaises(JevRequestError):
                run_compact(self.store, req, transport=client)
            client.post.assert_called_once()

    def test_invalid_response_retains_actual_charge_and_cannot_retry(self):
        raw = raw_response({"direction": "unknown"})
        raw["usage"]["cost"] = .20  # Actual cost may exceed the local estimate.
        client = transport(raw)
        with self.assertRaises(JevRequestError) as caught:
            run_compact(self.store, request(), transport=client)
        error = caught.exception
        self.assertIsNotNone(error.response_blob_sha256)
        self.assertEqual(json.loads((self.store.blobs / error.response_blob_sha256).read_bytes()), raw)
        row = self.store.db.execute("SELECT * FROM model_attempts").fetchone()
        self.assertEqual(row["status"], "invalid_response")
        self.assertEqual(row["accounted_usd"], .20)
        before = tuple(row)
        with self.assertRaises(ValueError):
            run_compact(self.store, request(), transport=client)
        self.assertEqual(tuple(self.store.db.execute("SELECT * FROM model_attempts").fetchone()), before)
        self.assertEqual(client.post.call_count, 1)

    def test_transport_failure_redacted_and_full_reservation_retained(self):
        client = Mock(post=Mock(side_effect=TransportError("remote echoed " + KEY, status=503)))
        with self.assertRaises(JevRequestError) as caught:
            run_compact(self.store, request(), transport=client)
        self.assertNotIn(KEY, str(caught.exception))
        self.assertEqual(caught.exception.status, 503)
        self.assertIsNone(caught.exception.response_data)
        row = self.store.db.execute("SELECT * FROM model_attempts").fetchone()
        self.assertEqual(row["status"], "outcome_uncertain")
        self.assertGreater(row["accounted_usd"], 0)
        self.assertEqual(row["accounted_usd"], row["reserved_usd"])
        with self.assertRaises(ValueError):
            run_compact(self.store, request(), transport=client)
        client.post.assert_called_once()

    def test_missing_cost_is_not_zero_or_fabricated(self):
        raw = raw_response()
        raw["usage"].pop("cost")
        result = run_compact(self.store, request(), transport=transport(raw))
        self.assertNotIn("cost", result["response"]["usage"])
        self.assertIsNone(result["incremental_cost_usd"])
        row = self.store.db.execute("SELECT * FROM model_attempts").fetchone()
        self.assertEqual(row["status"], "completed")
        self.assertEqual(row["accounted_usd"], row["reserved_usd"])

    def test_credentials_never_archived_even_nested_escaped_echo(self):
        for index, raw in enumerate((
                {"error": {"message": KEY}},
                raw_response({"direction": KEY, "scope": "segment"}),
                raw_response({"direction": "unknown", "scope": "segment"}))):
            if index == 2:
                raw["choices"][0]["message"]["content"] = '{"direction":"' + ''.join('\\u%04x' % ord(c) for c in KEY) + '","scope":"segment"}'
            req = request()
            req["state"]["fixture"] = index
            with self.subTest(index=index), self.assertRaises(JevRequestError) as caught:
                run_compact(self.store, req, transport=transport(raw))
            self.assertIsNone(caught.exception.response_blob_sha256)
            self.assertIsNone(caught.exception.response_data)
            self.assertNotIn(KEY, str(caught.exception))
        for blob in self.store.blobs.iterdir():
            self.assertNotIn(KEY.encode(), blob.read_bytes())

    def test_budget_guards_and_cumulative_reservations_prevent_submission(self):
        client = transport()
        for kwargs in ({"phase_budget_usd": True}, {"phase_budget_usd": float("nan")},
                       {"phase_budget_usd": 0}, {"max_cost_usd": 0},
                       {"max_cost_usd": float("inf")}):
            with self.subTest(kwargs=kwargs), self.assertRaises(JevValidationError):
                run_compact(self.store, request(), transport=client, **kwargs)
        with self.assertRaises(ValueError):
            run_compact(self.store, request(), phase_budget_usd=.000001, transport=client)
        client.post.assert_not_called()
        first = request()
        run_compact(self.store, first, transport=client)
        second = request()
        second["state"]["new"] = "different request"
        budget = estimate_compact(second)["estimated_cost_usd"] + .001
        with self.assertRaises(ValueError):
            run_compact(self.store, second, phase_budget_usd=budget, transport=client)
        client.post.assert_called_once()

    def test_corrupted_cache_rejected_without_resubmitting(self):
        req = request()
        result = run_compact(self.store, req, transport=transport())
        bad = copy.deepcopy(result["response"])
        bad["answers"]["direction"] = "increase"  # Valid category, contradictory raw evidence.
        with self.assertRaises(JevValidationError):
            validate_compact_response(bad, req)
        self.store.db.execute("UPDATE model_runs SET response_sha256=?",
                              (self.store.put_blob(canonical_json(bad)),))
        self.store.db.commit()
        client = transport()
        with self.assertRaises(JevValidationError):
            run_compact(self.store, req, transport=client)
        client.post.assert_not_called()

    def test_invalid_or_oversize_transport_bytes_are_not_archived(self):
        for index, payload in enumerate(("not bytes", b"{", b"x" * (MAX_RESPONSE_BYTES + 1))):
            req = request()
            req["state"]["fixture"] = index
            client = Mock(post=Mock(return_value=payload))
            with self.subTest(index=index), self.assertRaises(JevRequestError) as caught:
                run_compact(self.store, req, transport=client)
            self.assertIsNone(caught.exception.response_blob_sha256)
            client.post.assert_called_once()

    def test_invalid_key_or_key_in_source_prevents_attempt(self):
        client = transport()
        for value in (None, "", "invalid\nkey"):
            self.key.return_value = value
            with self.assertRaises(JevRequestError):
                run_compact(self.store, request(), transport=client)
        self.key.return_value = KEY
        req = request()
        req["state"]["accidental_secret"] = KEY
        with self.assertRaises(JevRequestError):
            run_compact(self.store, req, transport=client)
        client.post.assert_not_called()
        self.assertEqual(self.store.db.execute("SELECT COUNT(*) FROM model_attempts").fetchone()[0], 0)
        self.assertEqual(list(self.store.blobs.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
