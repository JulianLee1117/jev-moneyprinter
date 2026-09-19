import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import threading
import time
import unittest
from unittest.mock import Mock, patch

from jev_alpha import procurement_models as pm
from jev_alpha.store import Store, canonical_json
from jev_alpha.transport import TransportError


KEY = "synthetic-procurement-private-key"
PROTOCOL = {"budget_usd": 50, "max_cost_usd": .25}


def state(index=0):
    return {"text": f"Synthetic procurement unit {index}: recommend Alpha for $100; project budget $200.",
            "issuer_candidates": [{"candidate_id": "issuer-a", "symbol": "AAA", "alias": "Alpha"}],
            "amount_candidates": [{"candidate_id": "amount-a", "value_usd": 100, "evidence": "$100"},
                                  {"candidate_id": "amount-b", "value_usd": 200, "evidence": "$200"}],
            "prior_records": [], "target_amount_id": "amount-a"}


def entries(count=1):
    return [{"packet_id": f"packet-{i}", "request": pm.build_panel(state(i)), "split": "development"}
            for i in range(count)]


def manifest():
    return {"all_models_verified": True, "models": {
        arm: {"model": model, "status": "verified", "context_length": 32000,
              "prices_per_token": {"prompt": .000000042 if arm == "jev" else .0000002,
                                   "completion": 0 if arm == "jev" else .00000125, "request": 0}}
        for arm, model in pm.MODELS.items()}}


def response(body, cost=.001):
    if body["model"] == pm.MODELS["jev"]:
        return {"model": body["model"] + "-20260917", "answers": {
            name: {"type": "choice", "choice": "unknown",
                   "probabilities": {c: int(c == "unknown") for c in q["criteria"]}}
            for name, q in body["questions"].items()},
            "usage": {"input_tokens": 100, "output_tokens": 0, "cost": cost}}
    questions = json.loads(body["messages"][1]["content"])["questions"]
    return {"model": body["model"] + "-2026-03-17",
            "choices": [{"finish_reason": "stop", "message": {
                "content": json.dumps({name: "unknown" for name in questions})}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 30, "cost": cost}}


def post_response(url, body, key, **kwargs):
    return canonical_json(response(json.loads(body)))


class PanelTests(unittest.TestCase):
    def test_full_evidence_and_anchor_in_every_independent_question(self):
        original = state()
        before = copy.deepcopy(original)
        req = pm.build_panel(original)
        self.assertEqual(req["state"]["evidence"], before)
        self.assertEqual(original, before)
        self.assertEqual(set(req["questions"]), set(pm.QUESTION_IDS))
        self.assertEqual(set(req["questions"]["amount"]["criteria"]), {"amount-a", "none", "unknown"})
        for question in req["questions"].values():
            self.assertIn("target_amount_id='amount-a'", question["instructions"])
            self.assertIn("Other amounts", question["instructions"])
        self.assertIn("independently", req["state"]["research_policy"])
        self.assertIn("absence does not establish", req["state"]["research_policy"])

    def test_unanchored_and_no_amount_packets_remain_explicit(self):
        s = state()
        del s["target_amount_id"]
        self.assertEqual(len(pm.build_panel(s)["questions"]["amount"]["criteria"]), 4)
        s["amount_candidates"] = []
        self.assertEqual(set(pm.build_panel(s)["questions"]["amount"]["criteria"]), {"none", "unknown"})

    def test_invalid_candidate_anchor_and_missing_evidence_rejected(self):
        invalid = []
        for field, value in (("target_amount_id", "missing"), ("prior_records", None),
                             ("issuer_candidates", [{"candidate_id": "unknown"}]),
                             ("amount_candidates", [{"candidate_id": "x"}, {"candidate_id": "x"}])):
            s = state()
            s[field] = value
            invalid.append(s)
        s = state()
        del s["text"]
        invalid.append(s)
        for s in invalid:
            with self.subTest(state=s), self.assertRaises(ValueError):
                pm.build_panel(s)

    def test_chat_receives_identical_state_questions_and_categorical_schema(self):
        req = pm.build_panel(state())
        for arm in ("nano", "mini"):
            body = pm.build_chat_request(req, pm.MODELS[arm])
            self.assertEqual(json.loads(body["messages"][1]["content"]),
                             {"state": req["state"], "questions": req["questions"]})
            schema = body["response_format"]["json_schema"]["schema"]
            self.assertFalse(schema["additionalProperties"])
            self.assertNotIn("tools", body)
            for name, question in req["questions"].items():
                self.assertEqual(schema["properties"][name], {"type": "string", "enum": list(question["criteria"])})

    def test_choice_guidance_does_not_duplicate_full_candidate_provenance(self):
        s = state()
        s["issuer_candidates"][0]["verified_aliases"] = [{"source_passage": "provenance" * 1000}]
        req = pm.build_panel(s)
        self.assertEqual(req["state"]["evidence"], s)
        self.assertNotIn("provenance", req["questions"]["recipient"]["criteria"]["issuer-a"])
        self.assertNotIn("evidence", json.loads(req["questions"]["amount"]["criteria"]["amount-a"]))

    def test_nonce_requires_all_development_authorizations(self):
        s = state()
        s.update(benchmark_execution_nonce="explicit-repeat-1", counterfactual_benchmark=True)
        e = [{"packet_id": "repeat", "request": pm.build_panel(s), "split": "development"}]
        with self.assertRaises(ValueError):
            pm._prepare(e, "jev", PROTOCOL)
        protocol = dict(PROTOCOL, development_only=True, allow_paid_repeats=True)
        self.assertEqual(pm._prepare(e, "jev", protocol)[0]["benchmark_nonce"], "explicit-repeat-1")
        e[0]["split"] = "holdout"
        with self.assertRaises(ValueError):
            pm._prepare(e, "jev", protocol)

    def test_chat_uses_advertised_native_context_for_same_evidence(self):
        s = state()
        s["text"] = "x" * 27000
        req = pm.build_panel(s)
        info = manifest()["models"]["mini"]
        info["context_length"] = 400000
        body = pm.build_chat_request(req, pm.MODELS["mini"])
        result = pm._estimate(body, req, info, "mini")
        self.assertGreater(result["conservative_input_tokens"], 32000)
        self.assertEqual(result["context_limit_tokens"], 400000)
        with self.assertRaises(ValueError):
            pm._estimate(req, req, info, "jev")

    def test_external_request_is_verified_and_exempt_bodies_are_not_retained(self):
        from jev_alpha.procurement import archive_packet_request
        from jev_alpha.procurement_audit import _bound_row
        original = entries()[0]
        original.update(model_required=False, skip_inference_reason="no_verified_alias_in_supplied_evidence",
                        deterministic_signal_status="unknown")
        with TemporaryDirectory() as tmp:
            external = copy.deepcopy(original)
            archive_packet_request(Path(tmp), external)
            self.assertNotIn("request", external)
            for arm in pm.MODELS:
                expected = pm._prepare([original], arm, PROTOCOL)[0]
                actual = pm._prepare([external], arm, PROTOCOL)[0]
                self.assertEqual(actual, expected)
                self.assertNotIn("request", actual)
                self.assertNotIn("body", actual)
                _bound_row(pm._not_required(actual, arm), external, arm)
                required = pm._prepare([{**external, "model_required": True}], arm, PROTOCOL)[0]
                self.assertEqual(required["request"], original["request"])
                self.assertIn("body", required)
            changed = copy.deepcopy(original["request"])
            changed["state"]["evidence"]["text"] += " changed"
            Path(external["request_path"]).write_bytes(canonical_json(changed))
            with self.assertRaises(ValueError):
                pm._prepare([external], "jev", PROTOCOL)
            with self.assertRaises(ValueError):
                _bound_row(pm._not_required(expected, "jev"), external, "jev")


class MetadataTests(unittest.TestCase):
    def public(self, url):
        if url == pm.CATALOG_URL:
            return {"data": [{"id": model, "supported_parameters": ["reasoning"],
                              "reasoning": {"mandatory": False, "default_enabled": False,
                                            "supported_efforts": ["high", "low", "none"]}}
                             for model in pm.MODELS.values()]}
        model = next(m for m in pm.MODELS.values() if url == f"{pm.CATALOG_URL}/{m}/endpoints")
        endpoint = {"status": 0, "context_length": 32000, "name": model,
                    "supported_parameters": ["max_tokens", "response_format", "reasoning"],
                    "pricing": {"prompt": ".0000001", "completion": ".000001", "web_search": ".01"}}
        expensive = copy.deepcopy(endpoint)
        expensive["pricing"]["prompt"] = ".0000002"
        return {"data": {"id": model, "architecture": {"output_modalities": ["decisions"]},
                         "endpoints": [endpoint, expensive]}}

    def test_current_public_metadata_archived_and_maximum_price_reserved(self):
        with TemporaryDirectory() as tmp, patch.object(pm, "_public_json", side_effect=self.public) as get, \
                patch.object(pm, "read_key") as key, patch.object(pm, "CurlJSONTransport") as transport:
            result = pm.verify_models(Path(tmp))
            self.assertTrue(result["all_models_verified"])
            self.assertEqual(get.call_count, 4)
            self.assertTrue(Path(result["manifest_path"]).exists())
            for row in result["models"].values():
                self.assertEqual(row["prices_per_token"]["prompt"], .0000002)
                self.assertIn("web_search", row["excluded_feature_charges"])
            key.assert_not_called()
            transport.assert_not_called()

    def test_bad_or_unpriced_endpoint_does_not_fallback(self):
        def unknown_charge(url):
            value = self.public(url)
            if url != pm.CATALOG_URL:
                value["data"]["endpoints"][0]["pricing"]["unclassified_fee"] = ".5"
            return value
        with TemporaryDirectory() as tmp, patch.object(pm, "_public_json", side_effect=unknown_charge):
            result = pm.verify_models(Path(tmp))
            self.assertFalse(result["all_models_verified"])
            self.assertTrue(all(r["status"] == "unavailable" for r in result["models"].values()))

    def test_explicit_none_reasoning_requires_catalog_and_endpoint_support(self):
        request = pm.build_panel(state())
        for arm in ("nano", "mini"):
            body = pm.build_chat_request(request, pm.MODELS[arm])
            self.assertEqual(body["reasoning"], {"effort": "none"})
            self.assertEqual(body["max_tokens"], 1024)
            self.assertTrue(body["provider"]["require_parameters"])
            self.assertFalse(body["provider"]["allow_fallbacks"])
        for fault in (None, "mandatory", "missing_none", "catalog_parameter", "endpoint_parameter"):
            def public(url):
                value = self.public(url)
                if url == pm.CATALOG_URL:
                    for row in value["data"]:
                        if fault == "mandatory":
                            row["reasoning"]["mandatory"] = True
                        elif fault == "missing_none":
                            row["reasoning"]["supported_efforts"] = ["low", "minimal"]
                        elif fault == "catalog_parameter":
                            row["supported_parameters"] = []
                elif fault == "endpoint_parameter":
                    for row in value["data"]["endpoints"]:
                        row["supported_parameters"].remove("reasoning")
                return value
            with self.subTest(fault=fault), TemporaryDirectory() as tmp, \
                    patch.object(pm, "_public_json", side_effect=public), \
                    patch.object(pm, "read_key") as key, patch.object(pm, "CurlJSONTransport") as transport:
                result = pm.verify_models(Path(tmp))
                self.assertEqual(result["all_models_verified"], fault is None)
                self.assertEqual(result["models"]["jev"]["status"], "verified")
                for arm in ("nano", "mini"):
                    self.assertEqual(result["models"][arm]["status"], "verified" if fault is None else "unavailable")
                key.assert_not_called()
                transport.assert_not_called()


class RunTests(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.counter = 0
        for name, kwargs in (("verify_models", {"return_value": manifest()}),
                             ("read_key", {"return_value": KEY}),
                             ("CurlJSONTransport", {"return_value": Mock(post=Mock(side_effect=post_response))})):
            p = patch.object(pm, name, **kwargs)
            setattr(self, name, p.start())
            self.addCleanup(p.stop)
        self.post = self.CurlJSONTransport.return_value.post

    def run_arm(self, e=None, arm="jev", protocol=None, workers=1, live=True):
        self.counter += 1
        return pm.run_requests(self.root, e or entries(), protocol or PROTOCOL,
                               arm=arm, workers=workers, live=live, out=self.root / f"run-{self.counter}")

    def test_dry_run_never_reads_key_metadata_or_creates_model_ledger(self):
        report = self.run_arm(live=False)
        self.assertEqual(report["records"][0]["status"], "dry_run")
        self.verify_models.assert_not_called()
        self.read_key.assert_not_called()
        self.post.assert_not_called()
        self.assertFalse((self.root / "models").exists())

    def test_empty_live_cohort_is_explicit_and_does_not_call_metadata_or_models(self):
        report = pm.run_requests(self.root, [], PROTOCOL, arm="jev", live=True, out=self.root / "empty")
        self.assertTrue(report["empty_cohort"])
        self.assertTrue(report["all_completed"])
        self.assertEqual(report["records"], [])
        self.verify_models.assert_not_called()
        self.read_key.assert_not_called()
        self.post.assert_not_called()

    def test_not_required_is_explicit_no_inference_and_preserved_all_arms(self):
        e = entries()
        e[0]["model_required"] = False
        for arm in pm.MODELS:
            record = self.run_arm(e, arm)["records"][0]
            self.assertEqual(record["status"], "not_required")
            self.assertEqual(record["answers"], {})
            self.assertEqual(record["cost_usd"], 0)
            self.assertIsNone(record["ready_at"])
        self.verify_models.assert_not_called()
        self.read_key.assert_not_called()
        self.post.assert_not_called()
        self.assertFalse((self.root / "models").exists())

    def test_exempt_unknown_retains_reason_without_fabricated_answers_or_latency(self):
        from jev_alpha.procurement_audit import _bound_row
        e = entries()
        e[0].update(model_required=False, skip_inference_reason="no_verified_alias_in_shared_evidence",
                    deterministic_signal_status="unknown")
        for arm in pm.MODELS:
            record = self.run_arm(e, arm)["records"][0]
            self.assertEqual(record["status"], "not_required")
            self.assertEqual(record["reason"], e[0]["skip_inference_reason"])
            self.assertEqual(record["deterministic_signal_status"], "unknown")
            self.assertEqual(record["answers"], {})
            self.assertEqual(record["cost_usd"], 0)
            self.assertIsNone(record["ready_at"])
            self.assertIsNone(record["timing_ms"])
            self.assertFalse(record["latency_eligible"])
            _bound_row(record, e[0], arm)
            with self.assertRaises(ValueError):
                _bound_row(dict(record, deterministic_signal_status="no_signal"), e[0], arm)
        self.verify_models.assert_not_called()
        self.read_key.assert_not_called()
        self.post.assert_not_called()

    def test_jev_probabilities_preserved_chat_categories_and_shared_root(self):
        for arm in pm.MODELS:
            report = self.run_arm(arm=arm)
            record = report["records"][0]
            self.assertEqual(record["status"], "completed")
            self.assertFalse(record["cached"])
            self.assertIsNotNone(record["ready_at"])
            self.assertTrue(record["latency_eligible"])
            self.assertGreaterEqual(record["timing_ms"]["end_to_end"], record["timing_ms"]["inference"])
            if arm == "jev":
                self.assertEqual(record["answers"]["amount"]["probabilities"], {"amount-a": 0, "none": 0, "unknown": 1})
            else:
                self.assertEqual(record["answers"]["amount"], "unknown")
        with Store(self.root / "models") as store:
            self.assertEqual(store.db.execute("SELECT COUNT(*) FROM model_attempts").fetchone()[0], 3)
            self.assertAlmostEqual(pm._spent(store), .003)

    def test_cache_has_original_cost_no_new_key_or_fresh_latency(self):
        first = self.run_arm()
        self.read_key.reset_mock()
        second = self.run_arm()
        a, b = first["records"][0], second["records"][0]
        self.assertEqual(a["answers"], b["answers"])
        self.assertEqual(a["ready_at"], b["ready_at"])
        self.assertEqual(b["cost_usd"], .001)
        self.assertEqual(b["incremental_cost_usd"], 0)
        self.assertIsNone(b["timing_ms"])
        self.assertFalse(b["latency_eligible"])
        self.assertEqual(second["fresh_latency_observations"], 0)
        self.assertIsNone(second["latency_ms_p50"])
        self.read_key.assert_not_called()
        self.assertEqual(self.post.call_count, 1)

    def test_every_packet_context_checked_before_first_submission(self):
        e = entries(2)
        e[1]["request"]["state"]["evidence"]["text"] = "x" * 40_000
        with self.assertRaises(ValueError):
            self.run_arm(e)
        self.read_key.assert_not_called()
        self.post.assert_not_called()
        self.assertFalse((self.root / "models" / "research.sqlite3").exists())

    def test_unavailable_preflight_is_explicit_and_unpaid(self):
        meta = manifest()
        meta["models"]["jev"]["status"] = "unavailable"
        self.verify_models.return_value = meta
        result = self.run_arm()
        self.assertEqual(result["records"][0]["status"], "preflight_unavailable")
        self.read_key.assert_not_called()
        self.post.assert_not_called()

    def test_per_call_guard_before_credential_and_billing(self):
        with self.assertRaises(ValueError):
            self.run_arm(protocol=dict(PROTOCOL, max_cost_usd=0))
        self.read_key.assert_not_called()
        self.post.assert_not_called()

    def test_invalid_response_charged_no_retry_and_remaining_group_halted(self):
        def invalid(url, body, key, **kwargs):
            raw = response(json.loads(body), cost=.2)
            raw["model"] = "typesafe/other"
            return canonical_json(raw)
        self.post.side_effect = invalid
        first = self.run_arm(entries(2))
        self.assertEqual([r["status"] for r in first["records"]], ["invalid_response", "unattempted"])
        self.assertAlmostEqual(first["ledger_accounted_after_usd"], .2)
        second = self.run_arm(entries(1))
        self.assertEqual(second["records"][0]["status"], "prior_attempt_blocked")
        self.assertEqual(self.post.call_count, 1)

    def test_unknown_transport_failure_retains_reservation_redacts_and_blocks(self):
        self.post.side_effect = TransportError("Remote error " + KEY, status=503)
        result = self.run_arm(entries(2))
        record = result["records"][0]
        self.assertEqual(record["status"], "outcome_uncertain")
        self.assertIsNone(record["cost_usd"])
        self.assertGreater(record["accounted_usd"], 0)
        self.assertNotIn(KEY, json.dumps(result))
        self.assertEqual(result["records"][1]["status"], "unattempted")
        with Store(self.root / "models") as store:
            row = store.db.execute("SELECT * FROM model_attempts").fetchone()
            self.assertEqual(row["accounted_usd"], row["reserved_usd"])

    def test_resumed_429_attempt_retains_unknown_billing_and_failure_summaries(self):
        self.post.side_effect = TransportError("Rate limited", status=429)
        first = self.run_arm(arm="mini")
        original_report_bytes = (self.root / "run-1" / "report.json").read_bytes()
        with Store(self.root / "models") as store:
            original_attempts = [tuple(r) for r in store.db.execute("SELECT * FROM model_attempts")]
        self.read_key.reset_mock()

        resumed = self.run_arm(arm="mini")

        self.assertEqual(first["records"][0]["http_status"], 429)
        self.assertEqual(resumed["records"][0]["status"], "prior_attempt_blocked")
        self.assertIsNone(resumed["records"][0]["cost_usd"])
        self.assertEqual(resumed["unknown_cost_records"], 1)
        self.assertEqual(resumed["failed_records"], 1)
        self.assertFalse(resumed["all_completed"])
        self.assertEqual(resumed["reported_incremental_cost_usd"], 0)
        self.assertEqual(resumed["fresh_latency_observations"], 0)
        self.assertIsNone(resumed["records"][0]["timing_ms"])
        self.assertEqual(self.post.call_count, 1)
        self.read_key.assert_not_called()
        self.assertEqual((self.root / "run-1" / "report.json").read_bytes(), original_report_bytes)
        with Store(self.root / "models") as store:
            self.assertEqual([tuple(r) for r in store.db.execute("SELECT * FROM model_attempts")], original_attempts)

    def test_credential_echo_is_not_archived(self):
        def echo(url, body, key, **kwargs):
            raw = response(json.loads(body))
            raw["echo"] = key
            return canonical_json(raw)
        self.post.side_effect = echo
        result = self.run_arm()
        self.assertEqual(result["records"][0]["status"], "outcome_uncertain")
        for path in (self.root / "models" / "blobs").iterdir():
            self.assertNotIn(KEY.encode(), path.read_bytes())

    def test_nested_chat_input_credential_echo_rejected_before_request_archive(self):
        synthetic_key = 'synthetic-"quoted"-key'
        self.read_key.return_value = synthetic_key
        e = entries()
        e[0]["request"]["state"]["evidence"]["text"] = synthetic_key
        with self.assertRaises(ValueError):
            self.run_arm(e, arm="mini")
        self.post.assert_not_called()
        self.assertEqual(list((self.root / "models" / "blobs").iterdir()), [])

    def test_budget_reserves_one_group_and_releases_estimates_for_whole_corpus(self):
        with patch.object(pm, "_estimate", return_value={"estimated_cost_usd": .02}):
            result = self.run_arm(entries(10), protocol=dict(PROTOCOL, budget_usd=.04))
        self.assertGreater(result["whole_corpus_conservative_cost_usd"], .04)
        self.assertTrue(result["all_completed"])
        self.assertEqual(self.post.call_count, 10)
        self.assertAlmostEqual(result["ledger_accounted_after_usd"], .01)

    def test_insufficient_group_budget_is_atomic_and_never_submits_partial_group(self):
        with patch.object(pm, "_estimate", return_value={"estimated_cost_usd": .02}):
            result = self.run_arm(entries(4), protocol=dict(PROTOCOL, budget_usd=.05), workers=4)
        self.assertTrue(all(r["status"] == "unattempted" for r in result["records"]))
        self.post.assert_not_called()
        with Store(self.root / "models") as store:
            self.assertEqual(store.db.execute("SELECT COUNT(*) FROM model_attempts").fetchone()[0], 0)

    def test_existing_other_arm_spend_consumes_same_budget(self):
        self.run_arm(arm="mini")
        with patch.object(pm, "_estimate", return_value={"estimated_cost_usd": .02}):
            result = self.run_arm(arm="jev", protocol=dict(PROTOCOL, budget_usd=.0205))
        self.assertEqual(result["records"][0]["status"], "unattempted")
        self.assertEqual(self.post.call_count, 1)

    def test_request_hash_and_shape_rejected_before_any_calls(self):
        e = entries()
        e[0]["request_hash"] = "incorrect"
        with self.assertRaises(ValueError):
            self.run_arm(e)
        self.verify_models.assert_not_called()
        self.read_key.assert_not_called()

    def test_parallel_worker_caps_actual_fresh_requests(self):
        for workers in (1, 4, 8):
            active = peak = 0
            lock = threading.Lock()
            def slow(url, body, key, **kwargs):
                nonlocal active, peak
                with lock:
                    active += 1
                    peak = max(peak, active)
                time.sleep(.03)
                result = post_response(url, body, key, **kwargs)
                with lock:
                    active -= 1
                return result
            self.post.side_effect = slow
            e = entries(8)
            for entry in e:
                entry["request"]["state"]["evidence"]["batch"] = workers
            report = self.run_arm(e, workers=workers)
            self.assertEqual(peak, workers)
            self.assertEqual(report["fresh_latency_observations"], 8)
            self.assertTrue(report["all_completed"])
            self.assertGreater(report["fresh_execution_wall_ms"], 0)

    def test_corrupt_cache_blocks_resubmission(self):
        e = entries()
        self.run_arm(e)
        with Store(self.root / "models") as store:
            bad = store.put_blob(canonical_json({"wrong": "cache"}))
            with store.db:
                store.db.execute("UPDATE model_runs SET response_sha256=?", (bad,))
        result = self.run_arm(e)
        self.assertEqual(result["records"][0]["status"], "invalid_cache")
        self.assertEqual(self.post.call_count, 1)

    def test_cache_bytes_cannot_be_rewritten_under_original_digest(self):
        e = entries()
        self.run_arm(e)
        with Store(self.root / "models") as store:
            pointer = store.db.execute("SELECT response_sha256 FROM model_runs").fetchone()[0]
            path = store.blobs / pointer
            cached = json.loads(path.read_bytes())
            cached["ready_at"] = "1900-01-01T00:00:00Z"
            path.write_bytes(canonical_json(cached))
        result = self.run_arm(e)
        self.assertEqual(result["records"][0]["status"], "invalid_cache")
        self.assertEqual(self.post.call_count, 1)

    def test_invalid_serving_models_and_chat_schema_fail_closed(self):
        req = pm.build_panel(state())
        body = pm.build_chat_request(req, pm.MODELS["nano"])
        raw = response(body)
        for model in (None, "openai/other", pm.MODELS["nano"] + "-2026-99-99", pm.MODELS["nano"] + "-suffix"):
            bad = copy.deepcopy(raw)
            bad["model"] = model
            with self.assertRaises(ValueError):
                pm._normalize(bad, req, "nano")
        for content in ('{}', '{"amount":"unknown","amount":"unknown"}'):
            bad = copy.deepcopy(raw)
            bad["choices"][0]["message"]["content"] = content
            with self.assertRaises(ValueError):
                pm._normalize(bad, req, "nano")

    def test_dev_benchmark_disjoint_and_no_cached_latency_claims(self):
        protocol = dict(PROTOCOL, development_only=True)
        result = pm.benchmark_development(self.root, entries(9), protocol, arm="jev", out=self.root / "bench")
        groups = result["plan"]["groups"]
        packet_ids = [x["packet_id"] for g in groups.values() for x in g]
        self.assertEqual(len(set(packet_ids)), 9)
        self.assertEqual(set(groups), {"1", "4", "8"})
        self.assertIsNone(result["selected_workers"])
        self.post.assert_not_called()
        e = entries(3)
        e[0]["split"] = "holdout"
        with self.assertRaises(ValueError):
            pm.benchmark_development(self.root, e, protocol, arm="jev", out=self.root / "bad")


if __name__ == "__main__":
    unittest.main()
