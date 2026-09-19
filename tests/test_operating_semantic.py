import copy
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from jev_alpha.experiment import digest
from jev_alpha.operating_semantic import (DEFAULT_MODEL, inspection_windows, prepare_semantic_control,
    run_semantic_control, validate_rerank)
from jev_alpha.operating_sources import RERANK_URL, ScryClient
from jev_alpha.store import Store


def protocol():
    return {"use_case": "technical_evaluation_only", "source_mode": "scry",
            "vendors": [{"vendor_id": "mdb", "aliases": ["mongodb"]},
                        {"vendor_id": "net", "aliases": ["cloudflare"]}],
            "window": {"start": "2026-06-20T00:00:00Z", "end_exclusive": "2026-09-19T00:00:00Z"},
            "max_records": 1000, "max_per_vendor": 125,
            "source_quotas": {"hackernews": 63, "reddit": 62}, "selection_seed": "fixture",
            "subreddits": ["devops"], "budget": {"scry_usd": 5}}


def record(index, text=None, vendors=None):
    text = f"MongoDB ordinary question {index}." if text is None else text
    return {"record_id": f"r{index:04d}", "source": "hackernews", "native_id": str(index),
            "url": "https://example.com/" + str(index), "text": text,
            "content_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "published_at": "2026-08-01T00:00:00Z", "captured_at": "2026-09-18T00:00:00Z",
            "state_observed_at": None, "vendor_ids": vendors or ["mdb"]}


def cohort(rows=None, split="development"):
    rows = [record(i) for i in range(6)] if rows is None else rows
    return {"protocol_sha256": digest(protocol()), "records": rows,
            "splits": {r["record_id"]: split for r in rows}}


def envelope(group, model=DEFAULT_MODEL, scores=None):
    docs = group["documents"]
    scores = list(range(len(docs), 0, -1)) if scores is None else scores
    return {"reranked": True, "tier_applied": "fast", "model_applied": model,
            "results": [{"id": d["id"], "rank": i, "score": scores[i],
                         "document_chars": len(d["text"]), "best_window": [0, min(3500, len(d["text"]))]}
                        for i, d in enumerate(docs)],
            "usage": {"local_window_chars": 3500, "windows_scored": group["minimum_windows"]}}


class SemanticTests(unittest.TestCase):
    def test_window_boundaries_and_complete_unicode_coverage(self):
        for length, count in [(1, 1), (3500, 1), (3501, 2), (6500, 2), (6501, 3)]:
            windows = inspection_windows("😀" * length)
            self.assertEqual(len(windows), count)
            covered = set()
            for w in windows:
                covered.update(range(w["start"], w["end"]))
            self.assertEqual(covered, set(range(length)))

    def test_full_text_same_cohort_no_action_filter_and_no_eval_leak(self):
        rows = [record(1, "ordinary neutral question" * 300), record(2), record(3, "EVALUATION SENTINEL")]
        c = cohort(rows)
        c["splits"][rows[2]["record_id"]] = "evaluation"
        p = prepare_semantic_control(protocol(), c)
        group = p["groups"][0]
        self.assertEqual(group["payload"]["documents"], [{"id": r["record_id"], "text": r["text"]} for r in rows[:2]])
        self.assertNotIn("EVALUATION SENTINEL", json.dumps(p))
        self.assertEqual(group["payload"]["top_n"], 2)
        self.assertEqual(group["selection_count"], 1)

    def test_overlarge_vendor_group_and_singleton_unknown_without_requests(self):
        huge = "x" * (3000 * 1024 + 1)
        c = cohort([record(1, huge), record(2), record(3, vendors=["net"])])
        with TemporaryDirectory() as tmp, patch("jev_alpha.operating_semantic.ScryClient") as client:
            result = run_semantic_control(object(), protocol(), c, Path(tmp)/"out", live=True)
            self.assertEqual(result["summary"]["unknown_pairs"], 3)
            client.assert_not_called()
            p = json.loads((Path(tmp)/"out"/"plan.json").read_text())
            self.assertEqual(p["records"][0]["text"], huge)

    def test_dryrun_and_no_overwrite_never_construct_paid_client(self):
        with TemporaryDirectory() as tmp, patch("jev_alpha.operating_semantic.ScryClient") as client:
            out = Path(tmp)/"out"
            result = run_semantic_control(object(), protocol(), cohort(), out)
            self.assertEqual(result["summary"]["status"], "dry_run")
            with self.assertRaises(FileExistsError):
                run_semantic_control(object(), protocol(), cohort(), out, live=True)
            client.assert_not_called()

    def test_default_model_is_pinned_for_both_splits(self):
        for split in ("development", "evaluation"):
            p = prepare_semantic_control(protocol(), cohort(split=split), split=split)
            self.assertEqual(p["model_pin"], DEFAULT_MODEL)
            self.assertEqual(p["groups"][0]["payload"]["model"], DEFAULT_MODEL)

    def test_unavailable_does_not_select_input_order_and_halts_no_retry(self):
        rows = [record(1, vendors=["mdb", "net"]), record(2, vendors=["mdb", "net"])]
        with TemporaryDirectory() as tmp, patch("jev_alpha.operating_semantic.ScryClient") as cls:
            client = cls.return_value
            client.request.return_value = {"envelope": {"reranked": False, "degraded_reason": "fixture"}}
            client.spending.return_value = {"accounted_usd": .1}
            result = run_semantic_control(object(), protocol(), cohort(rows), Path(tmp)/"out", live=True)
            self.assertEqual(result["summary"]["selected_pairs"], 0)
            self.assertEqual(result["summary"]["unknown_pairs"], 4)
            self.assertEqual(client.request.call_count, 1)
            client.close.assert_called_once()

    def test_complete_scores_select_exact_ceiling_and_source_ledger_client(self):
        p = prepare_semantic_control(protocol(), cohort())
        with TemporaryDirectory() as tmp, patch("jev_alpha.operating_semantic.ScryClient") as cls:
            store = object()
            client = cls.return_value
            client.request.return_value = {"envelope": envelope(p["groups"][0]), "accounted_usd": .001, "cache_hit": False}
            client.spending.return_value = {"limit_usd": 5, "accounted_usd": 1.23}
            result = run_semantic_control(store, protocol(), cohort(), Path(tmp)/"out", live=True)
            cls.assert_called_once_with(store, protocol(), max_scry_usd=5)
            args, kwargs = client.request.call_args
            self.assertEqual(args[:2], ("POST", RERANK_URL))
            self.assertEqual(kwargs, {"exposure_usd": .1})
            self.assertEqual(result["summary"]["selected_pairs"], 2)
            selection = json.loads((Path(tmp)/"out"/"selection.json").read_text())
            self.assertEqual(selection["selected_pairs"], [{"record_id": "r0000", "vendor_id": "mdb"}, {"record_id": "r0001", "vendor_id": "mdb"}])
            self.assertEqual(result["summary"]["models_applied"], [DEFAULT_MODEL])

    def test_schema_duplicate_unknown_nonfinite_and_missing_coverage_rejected(self):
        group = prepare_semantic_control(protocol(), cohort())["groups"][0]
        mutations = [
            lambda e: e["results"][0].update(id="not-in-cohort"),
            lambda e: e["results"][1].update(id=e["results"][0]["id"]),
            lambda e: e["results"][0].update(score=float("nan")),
            lambda e: e["results"][0].update(score=True),
            lambda e: e["results"].pop(),
            lambda e: e["results"][0].update(document_chars=1),
            lambda e: e["results"][0].update(best_window=[0, 999999]),
            lambda e: e["usage"].update(windows_scored=1),
            lambda e: e.update(tier_applied="hosted"),
        ]
        for mutation in mutations:
            value = envelope(group)
            mutation(value)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                validate_rerank(value, group)

    def test_exact_model_pin_rejects_substitution(self):
        group = prepare_semantic_control(protocol(), cohort())["groups"][0]
        with self.assertRaises(ValueError):
            validate_rerank(envelope(group, "different-model"), group, model="registered-model")
        p = prepare_semantic_control(protocol(), cohort(split="evaluation"), split="evaluation", model="registered-model")
        self.assertEqual(p["groups"][0]["payload"]["model"], "registered-model")

    def test_pinned_tier_wire_requires_matching_pin_and_complete_unicode_coverage(self):
        # Provider's pre-cohort Unicode wire probe: 34 code points, 38 bytes.
        text = "We cut our Datadog bill by 50%: é😀"
        self.assertEqual((len(text), len(text.encode("utf-8"))), (34, 38))
        group = prepare_semantic_control(protocol(), cohort([record(1, text), record(2, "Datadog is software.")]))["groups"][0]
        value = envelope(group)
        value["tier_applied"] = "pinned"
        self.assertEqual(len(validate_rerank(value, group, model=DEFAULT_MODEL)), 2)
        for model in (None, "different-model"):
            with self.subTest(model=model), self.assertRaises(ValueError):
                validate_rerank(value, group, model=model)
        bad_coverage = copy.deepcopy(value)
        bad_coverage["results"][0]["document_chars"] = 38
        with self.assertRaises(ValueError):
            validate_rerank(bad_coverage, group, model=DEFAULT_MODEL)
        bad_windows = copy.deepcopy(value)
        bad_windows["usage"]["windows_scored"] = 1
        with self.assertRaises(ValueError):
            validate_rerank(bad_windows, group, model=DEFAULT_MODEL)

    def test_score_ties_use_frozen_id_order(self):
        group = prepare_semantic_control(protocol(), cohort())["groups"][0]
        value = envelope(group, scores=[1]*6)
        value["results"].reverse()
        ranking = validate_rerank(value, group)
        self.assertEqual([r["id"] for r in ranking], sorted(group["record_ids"]))

    def test_existing_source_spend_blocks_rerank_without_transport(self):
        with TemporaryDirectory() as tmp, Store(tmp) as store:
            with Store(store.root/"scry-billing") as ledger:
                ledger.reserve_attempt({"existing_source_query": True}, 4.95, 5)
            with patch("jev_alpha.operating_sources.read_key", return_value="fixture-key"), patch("jev_alpha.operating_sources.shutil.which", return_value="curl.exe"), patch("jev_alpha.operating_sources.subprocess.run") as transport:
                result = run_semantic_control(store, protocol(), cohort(), Path(tmp)/"out", live=True)
                self.assertEqual(result["summary"]["unknown_pairs"], 6)
                transport.assert_not_called()
            report = json.loads((Path(tmp)/"out"/"report.json").read_text())
            self.assertEqual(report["shared_scry_spending"]["accounted_usd"], 4.95)

    def test_tampered_cohort_hash_fails_before_client(self):
        c = cohort()
        c["records"][0]["text"] += "tampered"
        with TemporaryDirectory() as tmp, patch("jev_alpha.operating_semantic.ScryClient") as client:
            with self.assertRaises(ValueError):
                run_semantic_control(object(), protocol(), c, Path(tmp)/"out", live=True)
            client.assert_not_called()


if __name__ == "__main__":
    unittest.main()
