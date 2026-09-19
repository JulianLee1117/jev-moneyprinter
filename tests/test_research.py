import json
import tempfile
import unittest
from pathlib import Path

from jev_alpha.cli import import_discovery, generate_dossiers, main
from jev_alpha.research import census, select_pilot, source_observations, evidence_packet, text_passages
from jev_alpha.store import Store, write_new_json, canonical_json


class ResearchTests(unittest.TestCase):
    def test_synthetic_discovery_replays_and_keeps_capture_separate(self):
        with tempfile.TemporaryDirectory() as directory, Store(directory) as store:
            # Invented metadata keeps this import/version regression independent
            # of the private research archive and covers both title spellings.
            records = [
                {"document_number": "2024-10001", "publication_date": "2024-03-01",
                 "title": "Synthetic Corrosion-Resistant Steel: Preliminary Results"},
                {"document_number": "2025-10002", "publication_date": "2025-06-02",
                 "title": "Synthetic Corrosion Resistant Steel: Final Results"},
                {"document_number": "2025-10003", "publication_date": "2025-06-03",
                 "title": "Synthetic Unrelated Administrative Notice"},
            ]
            captured_at = "2000-01-01T00:00:00+00:00"
            source_url = "https://example.invalid/synthetic/federal-register-discovery"
            source = Path(directory) / "synthetic-discovery.json"
            write_new_json(source, {"records": records, "captured_at_utc": captured_at,
                                    "request_url": source_url})
            first = import_discovery(store, source)
            replay = import_discovery(store, source)
            self.assertEqual(first, replay)
            self.assertEqual(first["imported_records"], 3)
            self.assertEqual(first["archive_documents"], 3)
            self.assertEqual(store.documents(), records)
            summary = census(store.documents())
            self.assertEqual(summary["documents"], 3)
            self.assertEqual(summary["core_title_matches"], 2)
            self.assertEqual(summary["title_matches_by_year"], {"2024": 1, "2025": 1})
            self.assertEqual(summary["title_stage_hints"], {"preliminary": 1, "final": 1})
            self.assertIsNone(summary["independent_event_count"])
            observations = store.db.execute(
                "SELECT observed_at,original_captured_at,source_url,sha256 FROM observations ORDER BY id"
            ).fetchall()
            self.assertEqual(len(observations), 2)
            for row in observations:
                self.assertNotEqual(row["observed_at"], captured_at)
                self.assertEqual(row["original_captured_at"], captured_at)
                self.assertEqual(row["source_url"], source_url)
            self.assertEqual(observations[0]["sha256"], observations[1]["sha256"])
            versions = store.db.execute(
                "SELECT document_id,COUNT(*),COUNT(DISTINCT observation_id) "
                "FROM document_versions GROUP BY document_id ORDER BY document_id"
            ).fetchall()
            self.assertEqual([tuple(row) for row in versions],
                             [(record["document_number"], 2, 2) for record in records])

    def test_selection_reproducible_independent_of_input_order(self):
        docs = [{"document_number": f"2025-{i:05d}", "publication_date": "2025-03-01",
                 "title": f"Corrosion-Resistant Steel: {'Final' if i % 2 else 'Preliminary'} Results"} for i in range(40)]
        a, b = select_pilot(docs), select_pilot(list(reversed(docs)))
        self.assertEqual(a["selected"], b["selected"])
        self.assertEqual(a["universe_sha256"], b["universe_sha256"])
        self.assertEqual(len({x["document_number"] for x in a["selected"]}), 20)
        self.assertFalse(a["price_data_used"])

    def test_frozen_file_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frozen.json"
            write_new_json(path, {"a": 1})
            with self.assertRaises(FileExistsError):
                write_new_json(path, {"a": 2})
            self.assertEqual(json.loads(path.read_text())["a"], 1)

    def test_small_sample_covers_recent_and_older_cycles(self):
        docs = [{"document_number": f"{year}-12345", "publication_date": f"{year}-03-01", "title": "Corrosion-Resistant Steel Final Results"} for year in range(2015, 2027)]
        selected = select_pilot(docs, count=2)["selected"]
        self.assertEqual({d["publication_date"][:4] for d in selected}, {"2015", "2026"})

    def test_uncertain_model_attempt_retains_budget_and_blocks_retries(self):
        with tempfile.TemporaryDirectory() as directory, Store(directory) as store:
            request = {"test": 1}
            digest = store.reserve_attempt(request, 0.01, 0.02)
            store.finish_attempt(digest, None)
            with self.assertRaisesRegex(ValueError, "recorded attempt"):
                store.reserve_attempt(request, 0.01, 0.02)
            with self.assertRaisesRegex(ValueError, "cumulative"):
                store.reserve_attempt({"test": 2}, 0.011, 0.02)
            row = store.db.execute("SELECT accounted_usd,status FROM model_attempts").fetchone()
            self.assertEqual(tuple(row), (0.01, "outcome_uncertain"))

    def test_deliberate_retry_retains_original_uncertain_reservation(self):
        with tempfile.TemporaryDirectory() as directory, Store(directory) as store:
            request = {"test": 1}
            first = store.reserve_attempt(request, 0.01, 0.03)
            with self.assertRaisesRegex(ValueError, "finished failed"):
                store.reserve_attempt(request, 0.01, 0.03, retry_reason="inspect failure")
            store.finish_attempt(first, None)
            second = store.reserve_attempt(request, 0.01, 0.03, retry_reason="One diagnostic repeat after reviewing validation failure; preserve initial uncertain cost")
            self.assertNotEqual(first, second)
            store.finish_attempt(second, {"usage": {"cost": 0.002}})
            self.assertAlmostEqual(store.db.execute("SELECT SUM(accounted_usd) FROM model_attempts").fetchone()[0], 0.012)
            envelope = json.loads((store.blobs/second).read_bytes())
            self.assertEqual(envelope["retry_of"], first)

    def test_same_bytes_dedup_but_observations_retained(self):
        with tempfile.TemporaryDirectory() as directory, Store(directory) as store:
            store.observe("https://example.com/1", b"same", kind="test")
            store.observe("https://example.com/1", b"same", kind="test")
            self.assertEqual(len(list(store.blobs.iterdir())), 1)
            self.assertEqual(store.db.execute("SELECT COUNT(*) FROM observations").fetchone()[0], 2)

    def test_pi_date_never_becomes_verified_earliest_date(self):
        with tempfile.TemporaryDirectory() as directory, Store(directory) as store:
            store.observe("https://www.federalregister.gov/api/v1/public-inspection-documents/2026-14026.json",
                          canonical_json({"document_number": "2026-14026", "filed_at": "2026-07-10T05:45:00-07:00"}), kind="public_inspection", document_id="2026-14026")
            result = source_observations(store, "2026-14026")
            self.assertFalse(result["earliest_disclosure_verified"])
            self.assertEqual(result["public_inspection_candidate"]["filed_at"], "2026-07-10T05:45:00-07:00")

    def test_rejected_pi_identity_cannot_reenter_packet(self):
        with tempfile.TemporaryDirectory() as directory, Store(directory) as store:
            store.observe("https://example.com", canonical_json({"document_number": "2026-14027", "filed_at": "2026-07-10T05:45:00-07:00"}), kind="public_inspection", document_id="2026-14026")
            with self.assertRaisesRegex(ValueError, "mismatched"):
                source_observations(store, "2026-14026")

    def test_published_text_does_not_inherit_pi_timestamp(self):
        with tempfile.TemporaryDirectory() as directory, Store(directory) as store:
            doc_id = "2026-14026"
            obs = store.observe("https://example.com", b"x", kind="test")
            store.put_document({"document_number": doc_id, "title": "CORE", "publication_date": "2026-07-13"}, obs)
            store.observe("https://example.com/pi", canonical_json({"document_number": doc_id, "filed_at": "2026-07-10T05:45:00-07:00"}), kind="public_inspection", document_id=doc_id)
            store.observe("https://example.com/text", b"Later published source text. " * 10, kind="published_text", document_id=doc_id)
            content = evidence_packet(store, doc_id)["episode_manifest"]["content_availability"]
            self.assertIsNone(content["candidate_timestamp"])
            self.assertEqual(content["candidate_publication_date"], "2026-07-13")
            self.assertFalse(content["content_version_matches_timestamp"])

    def test_future_prior_rejected_before_packet(self):
        with tempfile.TemporaryDirectory() as directory, Store(directory) as store:
            for doc_id, day in (("2025-10001", "2025-03-01"), ("2025-10002", "2025-03-02")):
                obs = store.observe("https://example.com", b"x", kind="test")
                store.put_document({"document_number": doc_id, "publication_date": day, "title": "CORE"}, obs)
                store.observe("https://example.com/text", b"Actual source text. " * 20, kind="published_text", document_id=doc_id)
            with self.assertRaisesRegex(ValueError, "strictly before"):
                evidence_packet(store, "2025-10001", prior_ids=["2025-10002"])

    def test_parser_preserves_table_separation(self):
        raw = ("<html><body><p>Operative rates apply as follows.</p><table><tr><td>Producer A</td><td>12.3</td></tr></table>"
               "<p>Context and exclusions remain important to interpreting the new rates correctly.</p><script>ignore_me()</script></body></html>").encode()
        passages = text_passages(raw, "abcd" * 16)
        text = "\n".join(p["text"] for p in passages)
        self.assertIn("Producer A | 12.3", text)
        self.assertNotIn("ignore_me", text)

    def test_dossier_regeneration_preserves_annotations(self):
        with tempfile.TemporaryDirectory() as directory, Store(Path(directory)/"archive") as store:
            manifest = {"selected": [{"document_number": "2025-12345", "title": "CORE", "publication_date": "2025-01-01"}]}
            out = Path(directory) / "dossiers"
            generate_dossiers(store, manifest, out)
            with self.assertRaisesRegex(ValueError, "not be overwritten"):
                generate_dossiers(store, manifest, out)


if __name__ == "__main__":
    unittest.main()
