"""Synthetic regression tests for financial chronology and evidence integrity."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from jev_alpha.experiment import digest
from jev_alpha.financial_sources import compile_packets


class FinancialSourcesTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.doc = self.write("source.txt", "Entire invented source, including qualifying context.")
        self.pages = self.write("pages.json", [{"page_number": 1, "text": "Invented."}])
        self.census = {"event_count": 3, "events": [
            {"event_id": "tsmc-2024-q4", "issuer": "TSMC", "release_date": "2025-01-16"},
            {"event_id": "micron-fy2025-q2", "issuer": "Micron", "release_date": "2025-03-20"},
            {"event_id": "tsmc-2025-q1", "issuer": "TSMC", "release_date": "2025-04-17"},
        ]}
        self.census_ref = self.write("census.json", self.census)
        self.mu = {"census_sha256": self.census_ref["sha256"], "sources": [
            self.source("micron-fy2025-q1", "2024-12-18", True),
            self.source("micron-fy2025-q2", "2025-03-20", True),
        ]}
        self.tsmc = {"frozen_event_census_sha256": self.census_ref["sha256"], "documents": [
            self.source("tsmc-2024-q3", "2024-10-17"),
            self.source("tsmc-2024-q4", "2025-01-16"),
            self.source("tsmc-2025-q1", "2025-04-17"),
        ]}
        self.exposure = {
            "census_canonical_sha256": digest(self.census),
            "annual_exposure_versions": [{"exposure_version_id": "annual", "source_id": "supplier",
                                         "usable_from_date_conservative": "2024-08-30"}],
            "relationship_supplements": [],
            "sources": [{"source_id": "supplier", "requested_url": "https://example.invalid/supplier",
                         "body_path": self.doc["path"], "body_sha256": self.doc["sha256"],
                         "text_path": self.doc["path"], "text_sha256": self.doc["sha256"]}],
            "event_mapping": [{**event, "annual_exposure_version_id": "annual",
                               "dated_relationship_source_ids": ["supplier"],
                               "qualitative_historical_supplier_relationship_supported": True,
                               "customer_specific_revenue_weight_pct": None}
                              for event in self.census["events"]],
        }
        self.intervals = {"intervals": [
            {"event_id": "micron-fy2025-q2", "interval_after_date": "2024-12-18",
             "interval_before_date": "2025-03-20", "complete_public_prior_state_reconciliation": False,
             "news_bodies_complete_for_current_feed_interior": True, "keyword_hits": 0,
             "body_path": self.doc["path"], "body_sha256": self.doc["sha256"]},
        ]}
        self.supplement = {"slides": []}

    def write(self, name, value):
        data = value if isinstance(value, str) else json.dumps(value)
        raw = data.encode("utf-8")
        (self.root / name).write_bytes(raw)
        return {"path": name, "sha256": hashlib.sha256(raw).hexdigest()}

    def source(self, eid, day, micron=False):
        common = {"event_id": eid, "pages_path": self.pages["path"], "pages_sha256": self.pages["sha256"],
                  "page_count": 1}
        if micron:
            return {**common, "publication_date_displayed": day, "requested_url": "https://example.invalid/micron",
                    "body_path": self.doc["path"], "body_sha256": self.doc["sha256"],
                    "full_text_path": self.doc["path"], "full_text_sha256": self.doc["sha256"],
                    "capture_finished_at": "2026-09-19T04:00:00Z"}
        return {**common, "source_release_date": day, "source_url": "https://example.invalid/tsmc",
                "raw_path": self.doc["path"], "raw_sha256": self.doc["sha256"],
                "text_path": self.doc["path"], "text_sha256": self.doc["sha256"],
                "first_captured_at": "2026-09-19T04:00:00Z",
                "explicit_after_call_amendments": [{"page_number": 1, "context": "Original text."}]}

    def compile(self):
        for name, value in (("mu", self.mu), ("tsmc", self.tsmc), ("exposure", self.exposure),
                            ("intervals", self.intervals), ("supplement", self.supplement)):
            self.write(name + ".json", value)
        return compile_packets(root=self.root, census_path="census.json", micron_path="mu.json",
                               tsmc_path="tsmc.json", exposure_path="exposure.json", interval_paths=["intervals.json"],
                               supplemental_paths=["supplement.json"])

    def add_slides(self):
        self.supplement["slides"].append({
            "source_id": "mu-current-slides", "event_id": "micron-fy2025-q2",
            "displayed_release_date": "2025-03-20", "url": "https://example.invalid/slides",
            "body_path": self.doc["path"], "body_sha256": self.doc["sha256"],
            "pages_path": self.pages["path"], "pages_sha256": self.pages["sha256"], "page_count": 1})

    def test_full_census_missingness_and_amendments_survive(self):
        result = self.compile()
        self.assertEqual(result["event_count"], 3)
        self.assertEqual(result["documents"][2]["post_call_amendment_pages"], [1])
        self.assertNotIn("Original text.", json.dumps(result))
        packet = result["packets"][1]
        self.assertIn("intervening_guidance_reconciliation_incomplete", packet["gaps"])
        self.assertIn("prepared_remarks_do_not_include_earnings_qa", packet["gaps"])
        self.assertFalse(packet["quantitative_customer_revenue_weight_available"])
        self.assertFalse(packet["ready_for_price_collection"])
        self.assertEqual(result["verified_interval_file_count"], 1)

    def test_missing_predecessor_never_falls_back_to_older_quarter(self):
        self.tsmc["documents"] = [r for r in self.tsmc["documents"] if r["event_id"] != "tsmc-2024-q4"]
        result = self.compile()
        self.assertEqual(result["event_count"], 3)
        self.assertIsNone(result["packets"][2]["prior_document_id"])
        self.assertEqual(result["packets"][2]["required_prior_document_id"], "tsmc-2024-q4")
        self.assertIn("current_document_missing", result["packets"][0]["gaps"])

    def test_future_supplier_filing_is_rejected(self):
        self.exposure["annual_exposure_versions"][0]["usable_from_date_conservative"] = "2025-08-12"
        with self.assertRaisesRegex(ValueError, "Future supplier"):
            self.compile()

    def test_superseded_supplier_filing_is_rejected(self):
        self.exposure["annual_exposure_versions"][0]["superseded_from_date_in_this_annual_pair"] = "2025-01-16"
        with self.assertRaisesRegex(ValueError, "superseded"):
            self.compile()

    def test_valid_slides_survive_a_missing_primary_document(self):
        self.add_slides()
        self.mu["sources"].pop()
        result = self.compile()
        packet = result["packets"][1]
        self.assertIn("current_document_missing", packet["gaps"])
        self.assertEqual(packet["current_supplemental_document_ids"], ["mu-current-slides"])
        self.assertIn("exact_historical_supplemental_availability_unverified", packet["gaps"])
        self.assertEqual(result["packets"][0]["current_supplemental_document_ids"], [])

    def test_future_slide_date_is_rejected(self):
        self.add_slides()
        self.supplement["slides"][0]["displayed_release_date"] = "2026-03-20"
        with self.assertRaisesRegex(ValueError, "Supplemental source identity/date"):
            self.compile()

    def test_changed_document_bytes_are_rejected(self):
        (self.root / "source.txt").write_text("Changed context")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            self.compile()

    def test_wrong_interval_boundary_is_rejected(self):
        self.intervals["intervals"][0]["interval_before_date"] = "2025-03-21"
        with self.assertRaisesRegex(ValueError, "interval does not match"):
            self.compile()

    def test_foreign_or_changed_census_is_rejected(self):
        self.mu["census_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "different event census"):
            self.compile()

    def test_duplicate_interval_is_rejected(self):
        self.intervals["intervals"].append(copy.deepcopy(self.intervals["intervals"][0]))
        with self.assertRaisesRegex(ValueError, "duplicate event"):
            self.compile()

    def test_tampered_intervening_body_is_rejected(self):
        ref = self.write("news.txt", "An intervening revision")
        self.intervals["intervals"][0].update(body_path=ref["path"], body_sha256=ref["sha256"])
        (self.root / "news.txt").write_text("Revision omitted")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            self.compile()


if __name__ == "__main__":
    unittest.main()
