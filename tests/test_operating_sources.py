"""Offline source, provenance, sampling and spending tests; no real credentials."""
import copy
import hashlib
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from jev_alpha.operating_sources import (
    QUERY_URL, SCHEMA_URL, STATUS_MARKER, OperatingSourceError, ScryClient,
    build_source_query, collect_operating_sources, merge_operating_records,
    normalize_operating_record, selection_hash, validate_protocol,
)
from jev_alpha.store import Store

CAPTURE = "2026-09-18T12:34:56+00:00"


def schema():
    hn = ["hn_id", "uri", "title", "payload", "original_author", "original_timestamp",
          "parent_hn_id", "story_hn_id", "first_observed_on", "observed_on", "search_text_lc",
          "is_deleted", "dead"]
    reddit = ["id", "subreddit", "author", "created_utc", "title", "selftext", "url",
              "retrieved_on", "observed_on", "state_observed_at", "search_text_lc"]
    return {"surfaces": [
        {"relation": "hackernews.items", "description": "The view reads FINAL, newest wins.", "columns": [{"name": n, "type": "DateTime" if n == "original_timestamp" else "String"} for n in hn],
         "freshness": {"known_holes": ["story roots incomplete"]}},
        {"relation": "reddit.posts", "columns": [{"name": n, "type": "UInt64" if n == "created_utc" else "String"} for n in reddit]},
    ]}


def protocol():
    return {"experiment_id": "offline-sources-test", "use_case": "technical_evaluation_only",
            "source_mode": "scry", "window": {"start": "2026-06-20T00:00:00Z", "end_exclusive": "2026-09-19T00:00:00Z"},
            "vendors": [{"vendor_id": "mdb", "aliases": ["mongodb", "mongo"]},
                        {"vendor_id": "net", "aliases": ["cloudflare"]}],
            "subreddits": ["devops"], "max_per_vendor": 4, "max_records": 8,
            "source_quotas": {"hackernews": 2, "reddit": 2}, "selection_seed": "frozen-test",
            "budget": {"scry_usd": 5, "initial_query_ceiling_usd": 0.10}}


def hn(native=1, text="MongoDB ordinary product question", **extra):
    return {"hn_id": native, "payload": text, "original_timestamp": "2026-09-01 10:00:00",
            "observed_on": "2026-09-17", "parent_hn_id": 999, "story_hn_id": None,
            "original_author": "fixture_author", **extra}


def reddit(native="abc", text="MongoDB ordinary product question", **extra):
    return {"id": native, "subreddit": "devops", "selftext": text,
            "created_utc": "2026-09-01 10:00:00", "state_observed_at": "2026-09-17 22:00:00", **extra}


def completed(payload, status=200, returncode=0):
    return subprocess.CompletedProcess(["curl"], returncode,
        json.dumps(payload).encode() + STATUS_MARKER + str(status).encode())


class NormalizationTests(unittest.TestCase):
    def test_full_text_hash_and_conservative_date_bound(self):
        raw = hn(text="MongoDB\n" + "long body\n" * 10000, title="A title")
        result = normalize_operating_record("hackernews", raw, captured_at=CAPTURE, vendor_ids=["mdb"])
        expected = raw["title"] + "\n\n" + raw["payload"]
        self.assertEqual(expected, result["text"])
        self.assertEqual(hashlib.sha256(expected.encode()).hexdigest(), result["content_sha256"])
        self.assertEqual("2026-09-18T00:00:00+00:00", result["state_observed_at"])
        self.assertEqual("day_upper_bound", result["state_observed_at_precision"])
        self.assertEqual("2026-09-01T10:00:00+00:00", result["published_at"])
        self.assertEqual("parent_only", result["thread_resolution"])
        self.assertIsNone(result["thread_id"])
        self.assertEqual("hackernews:999", result["parent_id"])
        self.assertEqual("https://news.ycombinator.com/item?id=1", result["url"])
        self.assertNotIn("fixture_author", json.dumps(result))

    def test_unknown_observation_is_not_replaced_with_publication_or_capture(self):
        result = normalize_operating_record("hackernews", hn(observed_on=None), captured_at=CAPTURE)
        self.assertIsNone(result["state_observed_at"])
        self.assertEqual("unknown", result["state_observed_at_precision"])
        self.assertEqual(CAPTURE, result["captured_at"])

    def test_reddit_fullname_and_original_reference(self):
        result = normalize_operating_record("reddit", reddit(url="https://example.com/article"), captured_at=CAPTURE)
        self.assertEqual("reddit:t3_abc", result["record_id"])
        self.assertEqual("reddit:t3_abc", result["thread_id"])
        self.assertEqual("hour_upper_bound", result["state_observed_at_precision"])
        self.assertEqual("https://www.reddit.com/r/devops/comments/abc/", result["url"])
        self.assertEqual("https://example.com/article", result["linked_url"])

    def test_exact_capture_required_and_invalid_ids_rejected(self):
        for capture in (None, "2026-09-18", "2026-09-18T10:00:00"):
            with self.subTest(capture=capture), self.assertRaises(ValueError):
                normalize_operating_record("hackernews", hn(), captured_at=capture)
        with self.assertRaises(ValueError):
            normalize_operating_record("hackernews", hn(native="1;DROP"), captured_at=CAPTURE)
        with self.assertRaises(ValueError):
            normalize_operating_record("reddit", reddit(native="abc/def"), captured_at=CAPTURE)

    def test_body_merge_retains_all_vendor_and_thread_provenance(self):
        first = normalize_operating_record("hackernews", hn(text="MongoDB  cloudflare"), captured_at=CAPTURE, vendor_ids=["mdb"])
        second = normalize_operating_record("reddit", reddit(text="MongoDB\ncloudflare"), captured_at=CAPTURE, vendor_ids=["net"])
        merged = merge_operating_records([second, first, dict(first, vendor_ids=["net"])])
        self.assertEqual(1, len(merged))
        self.assertEqual(["mdb", "net"], merged[0]["vendor_ids"])
        self.assertEqual(["reddit:t3_abc"], merged[0]["duplicate_record_ids"])
        self.assertEqual(2, len(merged[0]["source_references"]))


class QueryTests(unittest.TestCase):
    def test_schema_gate_and_bounded_hash_selection_no_action_filter(self):
        p = protocol()
        with self.assertRaisesRegex(ValueError, "Schema"):
            build_source_query(p, p["vendors"][0], "hackernews", {})
        hn_sql = build_source_query(p, p["vendors"][0], "hackernews", schema())
        reddit_sql = build_source_query(p, p["vendors"][0], "reddit", schema())
        self.assertNotIn("LIMIT 1 BY hn_id", hn_sql)
        self.assertNotIn("parseDateTimeBestEffortOrNull(toString(original_timestamp)", hn_sql)
        self.assertIn("SHA256(concat('frozen-test|hackernews:', toString(hn_id)))", hn_sql)
        self.assertTrue(hn_sql.endswith("LIMIT 2"))
        self.assertIn("hasAnyTokens(search_text_lc, ['mongodb', 'mongo'])", hn_sql)
        self.assertNotIn("migrat", hn_sql)
        self.assertNotIn("score", hn_sql)
        self.assertIn("toDateTime(created_utc, 'UTC')", reddit_sql)
        self.assertIn("lowerUTF8(subreddit) IN ('devops')", reddit_sql)
        self.assertIn("concat('t3_', toString(id))", reddit_sql)
        self.assertIn("id IN (SELECT DISTINCT id FROM reddit.posts", reddit_sql)
        self.assertIn("LIMIT 1 BY id) WHERE hasAnyTokens", reddit_sql)
        self.assertEqual(2, reddit_sql.count("hasAnyTokens"))

    def test_token_matching_does_not_turn_mongo_into_mongolia(self):
        p = protocol()
        p["fixtures"] = {"schema": schema(), "captured_at": CAPTURE, "hackernews": [
            hn(1, text="Mongolia geography"), hn(2, text="MongoDB ordinary product question")], "reddit": []}
        with tempfile.TemporaryDirectory() as directory, Store(Path(directory) / "archive") as store:
            result = collect_operating_sources(store, p, Path(directory) / "result.json")
        self.assertEqual(["hackernews:2"], [r["record_id"] for r in result["records"]])
        self.assertIn("whole alphanumeric tokens", result["selection_semantics"]["alias_match"])

    def test_source_utc_datetime_columns_remain_indexable(self):
        p, current = protocol(), schema()
        for surface in current["surfaces"]:
            for column in surface["columns"]:
                if column["name"] in {"original_timestamp", "created_utc"}:
                    column["type"] = "Nullable(DateTime64(6, 'UTC'))"
        for source in ("hackernews", "reddit"):
            sql = build_source_query(p, p["vendors"][0], source, current)
            self.assertNotIn("toString(original_timestamp)", sql)
            self.assertNotIn("toString(created_utc)", sql)
            self.assertNotIn("multiSearchAny", sql)

    def test_protocol_requires_explicit_terms_scope_and_bounded_quotas(self):
        for mutate in (lambda p: p.pop("use_case"), lambda p: p.update(use_case="commercial"),
                       lambda p: p.update(max_records=1001),
                       lambda p: p.update(source_quotas={"hackernews": 3, "reddit": 2}),
                       lambda p: p["window"].update(start="2026-06-20")):
            p = protocol()
            mutate(p)
            with self.assertRaises(ValueError):
                validate_protocol(p)

    @patch("jev_alpha.operating_sources.subprocess.run")
    @patch("jev_alpha.operating_sources.urllib.request.urlopen")
    def test_fixture_hash_sample_dedup_and_missing_source_not_crossfilled(self, urlopen, run):
        p = protocol()
        rows = [hn(i, text=f"MongoDB neutral {i}") for i in range(1, 9)]
        p["fixtures"] = {"schema": schema(), "captured_at": CAPTURE,
                         "hackernews": rows, "reddit": [reddit(text="Cloudflare neutral opening post")]}
        with tempfile.TemporaryDirectory() as directory, Store(Path(directory) / "archive") as store:
            result = collect_operating_sources(store, p, Path(directory) / "result.json")
            p["fixtures"]["hackernews"].reverse()
            again = collect_operating_sources(store, p, Path(directory) / "again.json")
        self.assertEqual(result["records"], again["records"])
        expected = sorted([f"hackernews:{i}" for i in range(1, 9)], key=lambda r: selection_hash(p["selection_seed"], r))[:2]
        self.assertEqual(sorted(expected), sorted(r["record_id"] for r in result["records"] if r["source"] == "hackernews"))
        self.assertEqual(3, result["record_count"])
        missing = next(c for c in result["coverage"] if c["vendor_id"] == "mdb" and c["source"] == "reddit")
        self.assertEqual(2, missing["unfilled"])
        self.assertTrue(missing["no_crossfill"])
        run.assert_not_called()
        urlopen.assert_not_called()

    def test_latest_state_removing_mention_does_not_resurrect_old_text(self):
        p = protocol()
        p["fixtures"] = {"schema": schema(), "captured_at": CAPTURE, "hackernews": [
            hn(1, text="MongoDB old mention", observed_on="2026-09-16"),
            hn(1, text="Edited text no longer includes the product", observed_on="2026-09-17")], "reddit": []}
        with tempfile.TemporaryDirectory() as directory, Store(Path(directory) / "archive") as store:
            result = collect_operating_sources(store, p, Path(directory) / "result.json")
        self.assertEqual([], result["records"])

    @patch("jev_alpha.operating_sources.subprocess.run")
    def test_dry_run_immutable_output_and_complete_unattempted_matrix(self, run):
        with tempfile.TemporaryDirectory() as directory, Store(Path(directory) / "archive") as store:
            output = Path(directory) / "result.json"
            result = collect_operating_sources(store, protocol(), output)
            self.assertEqual("dry_run", result["status"])
            self.assertEqual(4, len(result["coverage"]))
            with self.assertRaises(FileExistsError):
                collect_operating_sources(store, protocol(), output)
        run.assert_not_called()


class BillingTransportTests(unittest.TestCase):
    @patch("jev_alpha.operating_sources.shutil.which", return_value="curl.exe")
    @patch("jev_alpha.operating_sources.subprocess.run")
    def test_credentials_stdin_only_and_cost_envelope_cached(self, run, which):
        envelope = {"spend_nanodollars": 12500000, "rows": [], "columns": [],
                    "coverage": [{"known_holes": ["partial tail"]}], "deadline_partial": True,
                    "completeness": "partial", "truncated": False}
        run.return_value = completed(envelope)
        with tempfile.TemporaryDirectory() as directory, Store(directory) as store:
            client = ScryClient(store, protocol(), key="synthetic-secret")
            try:
                result = client.request("POST", QUERY_URL, "SELECT 1 LIMIT 1")
                cached = client.request("POST", QUERY_URL, "SELECT 1 LIMIT 1")
                self.assertEqual(envelope, result["envelope"])
                self.assertTrue(cached["cache_hit"])
                self.assertAlmostEqual(0.0125, client.spending()["accounted_usd"])
                self.assertEqual(1, client.spending()["attempt_count"])
                archive = b"".join(p.read_bytes() for p in store.blobs.iterdir())
                self.assertNotIn(b"synthetic-secret", archive)
            finally:
                client.close()
        args, kwargs = run.call_args
        self.assertEqual(["curl.exe", "-q", "--config", "-"], args[0])
        self.assertNotIn("synthetic-secret", str(args))
        self.assertNotIn("env", kwargs)
        self.assertFalse(kwargs["shell"])
        self.assertIn(b"Authorization: Bearer synthetic-secret", kwargs["input"])
        self.assertIn(b"x-scry-max-exposure: 100000000", kwargs["input"])
        self.assertIn(b"retry = 0", kwargs["input"])
        self.assertEqual(subprocess.DEVNULL, kwargs["stderr"])
        run.assert_called_once()

    @patch("jev_alpha.operating_sources.shutil.which", return_value="curl")
    @patch("jev_alpha.operating_sources.subprocess.run")
    def test_uncertain_request_keeps_reservation_prevents_retry_and_enforces_cap(self, run, which):
        run.side_effect = subprocess.TimeoutExpired(["curl"], 65, output=b"sensitive output")
        with tempfile.TemporaryDirectory() as directory, Store(directory) as store:
            client = ScryClient(store, protocol(), max_scry_usd=0.10, key="synthetic-secret")
            try:
                with self.assertRaises(OperatingSourceError) as failure:
                    client.request("POST", QUERY_URL, "SELECT 1 LIMIT 1")
                self.assertNotIn("sensitive", str(failure.exception))
                self.assertAlmostEqual(0.10, client.spending()["accounted_usd"])
                with self.assertRaisesRegex(ValueError, "recorded attempt"):
                    client.request("POST", QUERY_URL, "SELECT 1 LIMIT 1")
                with self.assertRaisesRegex(ValueError, "budget exceeded"):
                    client.request("POST", QUERY_URL, "SELECT 2 LIMIT 1")
                run.assert_called_once()
            finally:
                client.close()

    @patch("jev_alpha.operating_sources.shutil.which", return_value="curl")
    @patch("jev_alpha.operating_sources.subprocess.run")
    def test_error_diagnostics_are_archived_with_key_redacted(self, run, which):
        key = 'synthetic-"secret'
        run.return_value = completed({"error": "failed", "echo": key}, 422)
        with tempfile.TemporaryDirectory() as directory, Store(directory) as store:
            client = ScryClient(store, protocol(), key=key)
            try:
                with self.assertRaises(OperatingSourceError) as caught:
                    client.request("POST", QUERY_URL, "SELECT 1 LIMIT 1")
                self.assertEqual("failed", caught.exception.manifest["envelope"]["error"])
                self.assertEqual("[REDACTED]", caught.exception.manifest["envelope"]["echo"])
                self.assertTrue(caught.exception.manifest["response_redacted"])
                self.assertNotIn(key, json.dumps(caught.exception.manifest))
                self.assertAlmostEqual(.10, client.spending()["accounted_usd"])
            finally:
                client.close()

    @patch("jev_alpha.operating_sources.shutil.which", return_value="curl")
    @patch("jev_alpha.operating_sources.subprocess.run")
    def test_model_ledger_does_not_consume_scry_budget_and_unknown_cost_is_reserved(self, run, which):
        run.return_value = completed({"surfaces": []})
        with tempfile.TemporaryDirectory() as directory, Store(directory) as store:
            store.reserve_attempt({"provider": "model"}, 9, 10)
            client = ScryClient(store, protocol(), key="synthetic-secret")
            try:
                result = client.request("GET", SCHEMA_URL, exposure_usd=.000000001)
                self.assertEqual("uncertain_reserved", result["billing_outcome"])
                self.assertEqual(.000000001, client.spending()["accounted_usd"])
                self.assertEqual(1, client.spending()["uncertain_attempts"])
            finally:
                client.close()

    @patch("jev_alpha.operating_sources.shutil.which", return_value="curl")
    @patch("jev_alpha.operating_sources.subprocess.run")
    @patch("jev_alpha.operating_sources.read_key", return_value="synthetic-secret")
    def test_live_mock_schema_precedes_queries_and_full_coverage_survives_failure(self, key, run, which):
        run.side_effect = [completed(dict(schema(), spend_nanodollars=0)),
                           completed({"error": "diagnostic", "spend_nanodollars": 0}, 500)]
        with tempfile.TemporaryDirectory() as directory, Store(directory) as store:
            result = collect_operating_sources(store, protocol(), Path(directory) / "result.json", live=True)
        self.assertEqual("stopped_source_error", result["status"])
        self.assertEqual(4, len(result["coverage"]))
        self.assertEqual(2, len(run.call_args_list))
        self.assertIn(SCHEMA_URL.encode(), run.call_args_list[0].kwargs["input"])
        self.assertIn(QUERY_URL.encode(), run.call_args_list[1].kwargs["input"])
        self.assertEqual("diagnostic", result["errors"][0]["manifest"]["envelope"]["error"])
        self.assertAlmostEqual(.10, result["billing"]["accounted_usd"])

    @patch("jev_alpha.operating_sources.subprocess.run")
    def test_bad_endpoints_unbounded_queries_and_invalid_caps_are_rejected(self, run):
        with tempfile.TemporaryDirectory() as directory, Store(directory) as store:
            client = ScryClient(store, protocol(), key="synthetic-secret")
            try:
                for query in ("SELECT * FROM reddit.posts", "SELECT 1 LIMIT 126", "SELECT 1; SELECT 2 LIMIT 1"):
                    with self.assertRaises(ValueError):
                        client.request("POST", QUERY_URL, query)
                with self.assertRaises(ValueError):
                    client.request("POST", "https://evil.invalid", "SELECT 1 LIMIT 1")
                for cap in (0, -1, float("nan"), float("inf"), True, 6):
                    with self.assertRaises(ValueError):
                        client.request("POST", QUERY_URL, "SELECT 1 LIMIT 1", exposure_usd=cap)
            finally:
                client.close()
        run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
