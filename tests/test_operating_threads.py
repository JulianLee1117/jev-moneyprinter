"""Offline bounded original-API parent resolution tests."""
import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from jev_alpha.operating_threads import KIND, _fetch_hn_item, _url, resolve_operating_threads
from jev_alpha.store import Store

STAMP = "2026-09-19T03:00:00+00:00"


def record(native, parent=None, root=None):
    return {"record_id": f"hackernews:{native}", "source": "hackernews", "native_id": str(native),
            "text": f"Original full content {native}", "content_sha256": hashlib.sha256(f"Original full content {native}".encode()).hexdigest(),
            "vendor_ids": ["mdb"], "url": f"https://news.ycombinator.com/item?id={native}",
            "captured_at": STAMP, "thread_id": f"hackernews:{root}" if root else None,
            "parent_id": f"hackernews:{parent}" if parent else None,
            "thread_resolution": "root" if root else "parent_only"}


def fetched(item, status=200):
    return {"status": status, "body": json.dumps(item).encode(), "captured_at": STAMP}


class ThreadResolutionTests(unittest.TestCase):
    @patch("jev_alpha.operating_threads.urllib.request.build_opener")
    def test_original_api_transport_uses_fixed_url_no_auth_and_one_timeout(self, build_opener):
        response = build_opener.return_value.open.return_value.__enter__.return_value
        response.status = 200
        response.read.return_value = b'{"id":123,"type":"story"}'
        result = _fetch_hn_item("123")
        request = build_opener.return_value.open.call_args.args[0]
        self.assertEqual("https://hacker-news.firebaseio.com/v0/item/123.json", request.full_url)
        self.assertIsNone(request.get_header("Authorization"))
        self.assertEqual(8, build_opener.return_value.open.call_args.kwargs["timeout"])
        self.assertEqual(200, result["status"])
        build_opener.return_value.open.assert_called_once()

    @patch("jev_alpha.operating_threads._fetch_hn_item")
    def test_existing_capture_parent_edges_resolve_without_network_or_text_edits(self, fetch):
        capture = {"records": [record(3, 2), record(2, 1, 1)], "protocol_sha256": "frozen"}
        original = copy.deepcopy(capture)
        with tempfile.TemporaryDirectory() as directory, Store(Path(directory) / "store") as store:
            result = resolve_operating_threads(store, capture, Path(directory) / "result.json")
        self.assertEqual(capture, original)
        self.assertEqual("hackernews:1", result["records"][0]["thread_id"])
        for before, after in zip(capture["records"], result["records"]):
            for key in ("text", "content_sha256", "record_id", "vendor_ids", "url", "captured_at"):
                self.assertEqual(before[key], after[key])
        self.assertEqual(0, result["thread_resolution_run"]["requests"])
        fetch.assert_not_called()

    @patch("jev_alpha.operating_threads._fetch_hn_item")
    def test_each_merged_reference_gets_its_own_root_and_cache_avoids_second_call(self, fetch):
        first = record(5, 4)
        first["source_references"] = [dict(first), record(8, 7)]
        capture = {"records": [first]}
        def reply(native):
            return fetched({"id": int(native), "type": "story"})
        fetch.side_effect = reply
        with tempfile.TemporaryDirectory() as directory, Store(Path(directory) / "store") as store:
            result = resolve_operating_threads(store, capture, Path(directory) / "result.json", live=True)
            cached = resolve_operating_threads(store, capture, Path(directory) / "cached.json", live=False)
        self.assertEqual(2, fetch.call_count)
        self.assertEqual("hackernews:4", result["records"][0]["source_references"][0]["thread_id"])
        self.assertEqual("hackernews:7", result["records"][0]["source_references"][1]["thread_id"])
        self.assertEqual(2, cached["thread_resolution_run"]["cache_hits"])
        self.assertEqual(0, cached["thread_resolution_run"]["unresolved_native_roots"])

    @patch("jev_alpha.operating_threads._fetch_hn_item")
    def test_request_cap_and_parent_chain_validation(self, fetch):
        fetch.side_effect = lambda native: fetched({"id": int(native), "type": "comment", "parent": int(native) - 1})
        with tempfile.TemporaryDirectory() as directory, Store(Path(directory) / "store") as store:
            result = resolve_operating_threads(store, {"records": [record(10, 9)]}, Path(directory) / "result.json", live=True, max_requests=2)
        self.assertEqual(2, fetch.call_count)
        self.assertIsNone(result["records"][0]["thread_id"])
        self.assertEqual("request_limit", result["thread_resolution_run"]["status"])

    @patch("jev_alpha.operating_threads._fetch_hn_item")
    def test_wrong_id_missing_type_and_cycles_do_not_create_roots(self, fetch):
        capture = {"records": [record(5, 4), record(8, 7), record(10, 11), record(11, 10)]}
        fetch.side_effect = lambda native: fetched({"id": 999, "type": "story"} if native == "4" else {"id": 7, "deleted": True})
        with tempfile.TemporaryDirectory() as directory, Store(Path(directory) / "store") as store:
            result = resolve_operating_threads(store, capture, Path(directory) / "result.json", live=True)
        self.assertTrue(all(r["thread_id"] is None for r in result["records"]))
        reasons = {r["reason"] for r in result["thread_resolution_run"]["unresolved"]}
        self.assertEqual({"native_id_mismatch", "invalid_item_type_or_parent", "cycle"}, reasons)

    @patch("jev_alpha.operating_threads._fetch_hn_item")
    def test_hop_limit_and_refusal_do_not_retry(self, fetch):
        fetch.side_effect = lambda native: fetched({"id": int(native), "type": "comment", "parent": int(native) - 1})
        with tempfile.TemporaryDirectory() as directory, Store(Path(directory) / "store") as store:
            result = resolve_operating_threads(store, {"records": [record(100, 99)]}, Path(directory) / "hops.json", live=True)
            self.assertEqual("hop_limit", result["thread_resolution_run"]["unresolved"][0]["reason"])
            self.assertEqual(19, fetch.call_count)
            fetch.reset_mock()
            fetch.side_effect = lambda native: fetched({"error": "denied"}, status=403)
            result = resolve_operating_threads(store, {"records": [record(500, 499), record(400, 399)]}, Path(directory) / "refusal.json", live=True, workers=1)
            self.assertEqual(1, fetch.call_count)
            self.assertEqual("source_refusal", result["thread_resolution_run"]["status"])

    @patch("jev_alpha.operating_threads._fetch_hn_item")
    def test_invalid_bounds_and_existing_output_fail_before_network(self, fetch):
        with tempfile.TemporaryDirectory() as directory, Store(Path(directory) / "store") as store:
            out = Path(directory) / "result.json"
            for arguments in ({"max_requests": 501}, {"workers": 5}, {"max_requests": True}):
                with self.assertRaises(ValueError):
                    resolve_operating_threads(store, {"records": []}, out, live=True, **arguments)
            out.write_text("preserved")
            with self.assertRaises(FileExistsError):
                resolve_operating_threads(store, {"records": []}, out, live=True)
        fetch.assert_not_called()


if __name__ == "__main__":
    unittest.main()
