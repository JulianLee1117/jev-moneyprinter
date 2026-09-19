import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from jev_alpha.experiment import compact_state, digest, run_one, run_comparison, support_diagnostics
from jev_alpha.jev import JevRequestError, build_request
from jev_alpha.store import Store, write_new_json


def request_entry():
    pack = {"model": "typesafe/jev-1.13", "state_policy": "Use source evidence only.", "questions": {
        "decision_finality": {"type": "choice", "instructions": "Which stage?", "criteria": {"final": "Final", "preliminary": "Preliminary"}}}}
    state = {"current_source_passages_with_ids": [{"passage_id": "p1", "text": "The final determination is affirmative."}],
             "episode_manifest": {"title": "Final results", "document_id": "2025-12345"}}
    request = build_request(state, pack)
    return {"document_id": "2025-12345", "request": request, "request_hash": digest(request), "input_hash": digest(request["state"])}


def comparison_manifest():
    labels = {"schema_version": "benchmark_labels_v1", "records": [{"document_id": "2025-12345", "question_id": "decision_finality", "expected": "final", "split": "development", "label_origin": "synthetic", "evidence": []}]}
    return {"schema_version": "comparison-v1", "labels": labels, "labels_sha256": digest(labels), "requests": [request_entry()]}


class ExperimentTests(unittest.TestCase):
    def test_retrieval_is_deterministic_and_omissions_explicit(self):
        state = {"current_source_passages_with_ids": [{"passage_id": f"p{i}", "text": "Cash deposit exception. " * 50} for i in range(40)], "episode_manifest": {}}
        original = copy.deepcopy(state)
        a, b = compact_state(state, byte_budget=5000), compact_state(state, byte_budget=5000)
        self.assertEqual(a, b)
        self.assertEqual(original, state)
        selected = {p["passage_id"] for p in a["current_source_passages_with_ids"]}
        omitted = set(a["episode_manifest"]["omitted_passage_ids"])
        self.assertFalse(selected & omitted)
        self.assertEqual(len(selected | omitted), 40)

    def test_bad_label_fails_before_paid_calls(self):
        with tempfile.TemporaryDirectory() as folder, Store(Path(folder)/"archive") as store:
            manifest = comparison_manifest()
            manifest["labels"]["records"][0]["expected"] = "not-an-option"
            manifest["labels_sha256"] = digest(manifest["labels"])
            path = Path(folder)/"manifest.json"
            write_new_json(path, manifest)
            with patch("jev_alpha.experiment.run_one") as mock:
                with self.assertRaises(ValueError):
                    run_comparison(store, path, Path(folder)/"out", live=True)
                mock.assert_not_called()

    def test_holdout_cannot_be_used_in_development_runner(self):
        with tempfile.TemporaryDirectory() as folder, Store(Path(folder)/"archive") as store:
            manifest = comparison_manifest()
            manifest["labels"]["records"][0]["split"] = "holdout"
            manifest["labels_sha256"] = digest(manifest["labels"])
            path = Path(folder)/"manifest.json"
            write_new_json(path, manifest)
            with self.assertRaisesRegex(ValueError, "development-only"):
                run_comparison(store, path, Path(folder)/"out")

    def test_failure_halts_new_paid_calls_but_keeps_coverage_denominator(self):
        with tempfile.TemporaryDirectory() as folder, Store(Path(folder)/"archive") as store:
            path = Path(folder)/"manifest.json"
            write_new_json(path, comparison_manifest())
            with patch("jev_alpha.experiment.openrouter_key", return_value="synthetic"), patch("jev_alpha.experiment.run_one", side_effect=JevRequestError("HTTP 429", status=429)) as mock:
                result = run_comparison(store, path, Path(folder)/"out", live=True)
            self.assertEqual(mock.call_count, 1)
            self.assertEqual(result["failures"], 2)
            groups = result["metrics"]["groups"]
            for group in groups:
                if group["model"] != "keyword-rules-v1":
                    self.assertEqual(group["coverage"], 0)

    def test_cache_failure_keeps_known_billed_cost(self):
        with tempfile.TemporaryDirectory() as folder, Store(folder) as store:
            response = {"usage": {"cost": 0.001}, "model": "typesafe/jev-1.13"}
            with patch("jev_alpha.experiment.openrouter_key", return_value="synthetic"), patch("jev_alpha.experiment.JevClient") as client, patch.object(store, "cache_response", side_effect=OSError("disk unavailable")):
                client.return_value.submit.return_value = response
                with self.assertRaises(OSError):
                    run_one(store, request_entry()["request"])
            row = store.db.execute("SELECT accounted_usd,status FROM model_attempts").fetchone()
            self.assertEqual(row[0], 0.001)
            self.assertEqual(row[1], "completed")

    def test_support_diagnostics_do_not_rewrite_inputs(self):
        manifest = comparison_manifest()
        manifest["labels"]["records"][0]["evidence"] = [{"passage_ids": ["p1", "missing"]}]
        original = copy.deepcopy(manifest)
        result = support_diagnostics(manifest)
        self.assertEqual(result[0]["cited_passages_retained"], 1)
        self.assertFalse(result[0]["all_cited_passages_retained"])
        self.assertEqual(manifest, original)


if __name__ == "__main__":
    unittest.main()
