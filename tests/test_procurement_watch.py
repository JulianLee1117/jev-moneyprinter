import copy
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from jev_alpha import procurement_watch as watch
from jev_alpha.store import write_new_json


PROTOCOL = {"orders_enabled": False, "sources": [{"id": "agency", "kind": "legistar", "client": "agency"}],
            "prospective_max_events": 30, "prospective_max_days": 90, "prospective_poll_seconds": 300}
DECISION = {"decisions": {"jev": {"decision": "promising_association_only"}}, "jev_contribution_verified": True}


class RecorderTests(unittest.TestCase):
    def setUp(self):
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.now = "2026-09-18T12:00:00+00:00"
        self.payload = b"startup existing bytes"
        self.complete = True
        self.substantive = True
        self.url = "https://agency.example/document"
        self.protocol = copy.deepcopy(PROTOCOL)
        p = patch.object(watch, "utc_now", side_effect=lambda: self.now)
        p.start()
        self.addCleanup(p.stop)
        p = patch.object(watch, "collect", side_effect=self.collect)
        self.collect_mock = p.start()
        self.addCleanup(p.stop)

    def collect(self, snapshot, protocol, *, live):
        self.assertTrue(live)
        self.assertEqual(protocol["sources"], PROTOCOL["sources"])
        path = snapshot / "bytes" / "document"
        path.parent.mkdir(parents=True)
        path.write_bytes(self.payload)
        text_path = snapshot / "bytes" / "text"
        text_path.write_text((self.payload.decode() + " ") * 10 if self.substantive else "", encoding="utf-8")
        return {"manifest_path": "sources/census.json", "documents": [{"document_id": "doc-1", "source_id": "agency",
            "url": self.url, "sha256": hashlib.sha256(self.payload).hexdigest(),
            "blob_path": "bytes/document", "text_path": "bytes/text", "captured_at": self.now,
            "extraction_status": "extracted_plain_text" if self.substantive else "capture_failed"}],
            "source_results": [{"source_id": "agency", "complete": self.complete, "errors": []}]}

    def capture(self, *, live=True, report=None):
        return watch.capture_once(self.root, self.protocol, DECISION if report is None else report, live=live)

    def test_dry_run_has_no_collector_or_filesystem_mutation(self):
        result = self.capture(live=False)
        self.assertEqual(result["status"], "dry_run")
        self.assertTrue(result["recorder_only"])
        self.collect_mock.assert_not_called()
        self.assertFalse((self.root / "prospective").exists())

    def test_both_promising_jev_and_explicit_contribution_are_required(self):
        for report in ({}, {"decisions": DECISION["decisions"]},
                       {"decisions": {"mini": {"decision": "promising_association_only"}}, "jev_contribution_verified": True}):
            with self.assertRaises(ValueError):
                self.capture(report=report)
        self.collect_mock.assert_not_called()

    def test_baseline_old_bytes_excluded_new_bytes_first_seen_and_reversion_not_new(self):
        baseline = self.capture()
        self.assertEqual(baseline["candidate_count"], 0)
        self.assertEqual(baseline["baseline_sources"], ["agency"])
        self.now = "2026-09-18T12:10:00+00:00"
        self.payload = b"newly observed changed bytes"
        changed = self.capture()
        self.assertEqual(changed["candidate_count"], 1)
        self.assertEqual(changed["candidates"][0]["first_seen_at"], self.now)
        self.assertIsNone(changed["candidates"][0]["eligible_trading_event"])
        self.assertIsNone(changed["candidates"][0]["ready_at"])
        self.assertEqual(changed["unobserved_delay_seconds"], 300)
        self.assertEqual(changed["previous_sha256"], watch._hash(baseline))
        self.now = "2026-09-18T12:15:00+00:00"
        self.payload = b"startup existing bytes"
        reverted = self.capture()
        self.assertEqual(reverted["candidate_count"], 1)
        self.assertEqual(reverted["candidates"], [])
        self.assertEqual(len(list((self.root / "prospective" / "checkpoints").glob("*.json"))), 3)

    def test_failed_startup_source_is_baselined_on_recovery_not_promoted(self):
        self.complete = False
        self.substantive = False
        first = self.capture()
        self.assertEqual(first["baseline_sources"], [])
        self.assertEqual(first["candidate_count"], 0)
        self.assertTrue(first["failures"])
        self.now = "2026-09-18T12:05:00+00:00"
        self.payload = b"old bytes newly recovered from startup gap"
        self.complete = True
        self.substantive = True
        recovered = self.capture()
        self.assertEqual(recovered["baseline_sources"], ["agency"])
        self.assertEqual(recovered["candidate_count"], 0)

    def test_incomplete_html_source_baselines_observed_set_and_marks_uncertainty(self):
        self.complete = False
        baseline = self.capture()
        self.assertEqual(baseline["baseline_sources"], ["agency"])
        self.assertEqual(baseline["candidate_count"], 0)
        self.now = "2026-09-18T12:05:00+00:00"
        self.payload = b"changed substantive content in incomplete inventory"
        changed = self.capture()
        self.assertEqual(changed["candidate_count"], 1)
        candidate = changed["candidates"][0]
        self.assertFalse(candidate["inventory_coverage_complete"])
        self.assertTrue(candidate["may_preexist_observation"])
        self.assertIsNone(candidate["eligible_trading_event"])
        self.assertTrue(changed["failures"])

    def test_poll_interval_and_frozen_config_are_enforced(self):
        self.capture()
        self.now = "2026-09-18T12:04:59+00:00"
        self.assertEqual(self.capture()["status"], "not_due")
        self.assertEqual(self.collect_mock.call_count, 1)
        self.protocol["sources"][0]["client"] = "changed"
        self.now = "2026-09-18T12:05:00+00:00"
        with self.assertRaises(ValueError):
            self.capture()
        self.assertFalse((self.root / "prospective" / "capture.lock").exists())

    def test_reviewed_eligible_event_cap_stops_before_another_network_call(self):
        self.protocol["prospective_max_events"] = 1
        self.capture()
        self.now = "2026-09-18T12:05:00+00:00"
        self.payload = b"new bytes"
        record = self.capture()
        self.assertEqual(record["status"], "captured")
        self.assertEqual(record["eligible_event_count"], 0)
        write_new_json(self.root / "prospective" / "eligible-events" / "review-1.json", {
            "schema_version": "procurement-prospective-eligibility-v1", "reviewed": True,
            "event_id": "project-issuer-1", "candidate_id": record["candidates"][0]["candidate_id"], "status": "eligible"})
        self.now = "2026-09-18T12:10:00+00:00"
        self.assertEqual(self.capture()["status"], "bounded_stop")
        self.assertEqual(self.collect_mock.call_count, 2)

    def test_unclassified_candidates_do_not_consume_eligible_event_limit(self):
        self.protocol["prospective_max_events"] = 1
        self.capture()
        for minute in (5, 10):
            self.now = f"2026-09-18T12:{minute:02d}:00+00:00"
            self.payload = f"new bytes at {minute}".encode()
            record = self.capture()
            self.assertEqual(record["status"], "captured")
            self.assertEqual(record["eligible_event_count"], 0)
        self.assertEqual(record["candidate_count"], 2)

    def test_day_cap_stops_even_with_no_new_candidates(self):
        self.protocol["prospective_max_days"] = 1
        self.capture()
        self.now = "2026-09-19T12:00:00+00:00"
        self.assertEqual(self.capture()["status"], "bounded_stop")
        self.assertEqual(self.collect_mock.call_count, 1)

    def test_collector_exception_becomes_durable_failure_without_auto_retry(self):
        self.collect_mock.side_effect = RuntimeError("synthetic collector failure")
        result = self.capture()
        self.assertEqual(result["failures"][0]["reason"], "collector_failed")
        self.assertEqual(result["baseline_sources"], [])
        self.assertTrue((self.root / "prospective" / "checkpoints" / "000000.json").exists())
        self.assertEqual(self.collect_mock.call_count, 1)


if __name__ == "__main__":
    unittest.main()
