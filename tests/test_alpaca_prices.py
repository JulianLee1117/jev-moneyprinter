from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from urllib.parse import parse_qs, urlsplit

from jev_alpha.alpaca_prices import (AlpacaTransportError, CurlAlpacaTransport, STATUS_MARKER,
    alpaca_entitlement_smoke, collect_alpaca_prices)
from jev_alpha.store import Store


NOW = datetime(2026, 9, 18, 15, 0, tzinfo=timezone.utc)
KEYS = {"APCA_API_KEY_ID": "synthetic-alpaca-id", "APCA_API_SECRET_KEY": "synthetic-alpaca-secret"}


def query(kind="quotes"):
    return {"symbol": "AAPL", "kind": kind, "feed": "sip", "currency": "USD", "asof": "2026-09-17",
            "start": "2026-09-17T14:00:00Z", "end": "2026-09-17T14:01:00Z", "limit": 5, "max_pages": 2,
            "calendar": {"name": "XNYS", "timezone": "America/New_York", "date": "2026-09-17",
                         "open": "2026-09-17T09:30:00-04:00", "close": "2026-09-17T16:00:00-04:00",
                         "source_url": "https://www.nyse.com/markets/hours-calendars"}}


def quote(**changes):
    return {"t": "2026-09-17T14:00:00.123456789Z", "bp": 200.01, "ap": 200.02, "bs": 2, "as": 3,
            "bx": "P", "ax": "Q", "c": ["R"], "z": "C", **changes}


def response(rows=None, *, token=None, kind="quotes", status=200, **extra):
    return status, json.dumps({kind: {"AAPL": [quote()] if rows is None else rows}, "next_page_token": token, **extra}).encode()


class FakeTransport:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def get(self, url, key_id, secret_key):
        self.calls.append((url, key_id, secret_key))
        value = self.responses.pop(0)
        if isinstance(value, Exception):
            raise value
        return value


class AlpacaCaptureTests(unittest.TestCase):
    def run_capture(self, root, store, answers, request=None, *, smoke=False):
        client = FakeTransport(answers)
        function = alpaca_entitlement_smoke if smoke else collect_alpaca_prices
        with patch.dict(os.environ, KEYS, clear=True):
            receipt = function(store, request or query(), Path(root) / "capture", transport=client, now=NOW)
        records = json.loads((Path(root) / "capture/records.json").read_text())
        return receipt, records, client

    def test_missing_and_malformed_credentials_make_zero_requests_and_write_receipt(self):
        for values, expected in (({}, "missing_credentials"), ({"APCA_API_KEY_ID": "bad\nkey", "APCA_API_SECRET_KEY": "good"}, "invalid_credentials")):
            with self.subTest(expected=expected), TemporaryDirectory() as root, Store(Path(root) / "archive") as store:
                with patch.dict(os.environ, values, clear=True):
                    client = FakeTransport([])
                    receipt = collect_alpaca_prices(store, query(), Path(root) / "capture", env_path=Path(root) / "absent", transport=client, now=NOW)
                self.assertEqual(receipt["status"], expected)
                self.assertEqual(client.calls, [])
                self.assertEqual(receipt["entitlement_status"], "not_established")
                self.assertTrue((Path(root) / "capture/receipt.json").exists())

    def test_valid_capture_preserves_raw_bytes_nanoseconds_and_provenance(self):
        answer = response()
        with TemporaryDirectory() as root, Store(Path(root) / "archive") as store:
            receipt, records, client = self.run_capture(root, store, [answer])
            self.assertEqual(receipt["status"], "captured")
            self.assertTrue(receipt["complete"])
            self.assertEqual(receipt["entitlement_status"], "accepted_for_this_request")
            row = records[0]
            self.assertEqual(row["timestamp"], quote()["t"])
            self.assertEqual(row["timestamp_ns"] % 1_000_000_000, 123456789)
            self.assertEqual((row["bid"], row["ask"]), (200.01, 200.02))
            self.assertFalse(row["is_fill"])
            page = receipt["pages"][0]
            self.assertEqual((store.blobs / page["body_sha256"]).read_bytes(), answer[1])
            self.assertLessEqual(page["capture_started_at"], page["capture_finished_at"])
            params = parse_qs(urlsplit(client.calls[0][0]).query)
            self.assertEqual(params["feed"], ["sip"])
            self.assertEqual(params["currency"], ["USD"])
            self.assertEqual(params["asof"], ["2026-09-17"])
            for path in (Path(root) / "capture").iterdir():
                self.assertNotIn(KEYS["APCA_API_SECRET_KEY"], path.read_text())

    def test_invalid_query_rejected_before_network_or_output(self):
        cases = []
        for key, value in (("end", "2026-09-18T14:59:00Z"), ("end", "2026-09-17T14:01:00"),
                           ("start", "2026-09-17T14:00:00+00:60"), ("asof", None), ("asof", "2027-01-01"),
                           ("feed", "iex"), ("currency", "EUR"), ("symbol", "AAPL,MSFT"),
                           ("max_pages", 11), ("limit", True), ("end", "2026-09-17T16:00:00Z")):
            item = query()
            item[key] = value
            cases.append(item)
        for changes in ({"open": "2026-09-17T09:30:00-05:00"}, {"date": "2026-09-19"},
                        {"source_url": "https://user:password@example.com/calendar"}, {"timezone": "UTC"}):
            item = query()
            item["calendar"].update(changes)
            cases.append(item)
        for item in cases:
            with self.subTest(item=item), TemporaryDirectory() as root, patch("jev_alpha.alpaca_prices.read_key") as key:
                client = FakeTransport([])
                with self.assertRaises(ValueError):
                    collect_alpaca_prices(None, item, Path(root) / "capture", transport=client, now=NOW)
                key.assert_not_called()
                self.assertFalse((Path(root) / "capture").exists())

    def test_crossed_zero_locked_quotes_and_empty_size_are_retained_flagged(self):
        rows = [quote(bp=201), quote(bp=0), quote(ap=200.01), quote(bs=0), quote(bp=True)]
        with TemporaryDirectory() as root, Store(Path(root) / "archive") as store:
            receipt, records, _ = self.run_capture(root, store, [response(rows)])
            self.assertEqual(receipt["valid_numeric_record_count"], 0)
            self.assertIn("crossed_quote", records[0]["quality_flags"])
            self.assertIn("invalid_or_nonpositive_quote", records[1]["quality_flags"])
            self.assertIn("locked_quote", records[2]["quality_flags"])
            self.assertIn("invalid_or_empty_size", records[3]["quality_flags"])
            self.assertTrue(all(not r["is_fill"] for r in records))

    def test_exact_fifteen_minute_boundary_is_allowed_but_one_nanosecond_newer_is_rejected(self):
        request = query()
        request.update(start="2026-09-18T14:44:00Z", end="2026-09-18T14:45:00Z")
        request["calendar"].update(date="2026-09-18", open="2026-09-18T09:30:00-04:00", close="2026-09-18T16:00:00-04:00")
        with TemporaryDirectory() as root, Store(Path(root) / "archive") as store:
            receipt, records, _ = self.run_capture(root, store, [response([])], request)
            self.assertEqual(receipt["status"], "empty")
        request["end"] = "2026-09-18T14:45:00.000000001Z"
        with TemporaryDirectory() as root, patch("jev_alpha.alpaca_prices.read_key") as key:
            with self.assertRaisesRegex(ValueError, "15 minutes"):
                collect_alpaca_prices(None, request, Path(root) / "capture", now=NOW)
            key.assert_not_called()

    def test_pagination_retains_immutable_pages_and_stops_at_bound(self):
        for final_token, expected in ((None, "captured"), ("more", "truncated")):
            with self.subTest(expected=expected), TemporaryDirectory() as root, Store(Path(root) / "archive") as store:
                receipt, records, client = self.run_capture(root, store, [response(token="next"), response([quote(t="2026-09-17T14:00:01Z")], token=final_token)])
                self.assertEqual(receipt["status"], expected)
                self.assertEqual(len(records), 2)
                self.assertEqual(len(client.calls), 2)
                self.assertEqual(parse_qs(urlsplit(client.calls[1][0]).query)["page_token"], ["next"])
                self.assertEqual(receipt["complete"], final_token is None)

    def test_repeated_page_token_out_of_window_order_feed_and_symbol_are_rejected(self):
        bad = [response([quote(t="2026-09-17T13:59:59Z")]), response(feed="iex"), response(currency="EUR"),
               response([quote(), quote(t="2026-09-17T14:00:00Z")]),
               (200, b'{"quotes":{"MSFT":[]},"next_page_token":null}')]
        for answer in bad:
            with self.subTest(answer=answer), TemporaryDirectory() as root, Store(Path(root) / "archive") as store:
                receipt, records, client = self.run_capture(root, store, [answer])
                self.assertEqual(receipt["status"], "invalid_response")
                self.assertEqual(records, [])
                self.assertEqual(len(client.calls), 1)
        with TemporaryDirectory() as root, Store(Path(root) / "archive") as store:
            receipt, records, client = self.run_capture(root, store, [response(token="next"), response(token="next")])
            self.assertEqual(receipt["status"], "invalid_response")
            self.assertEqual(len(client.calls), 2)

    def test_denial_error_timeout_and_empty_have_distinct_status_and_no_retry(self):
        for answer, expected in (((403, b'{"message":"forbidden"}'), "provider_denied"),
                                 ((422, b'{"message":"subscription unavailable"}'), "provider_denied"),
                                 ((429, b'{"message":"rate limit"}'), "http_error"),
                                 ((200, b'{"code":42210000,"message":"error"}'), "invalid_response"),
                                 (AlpacaTransportError("redacted"), "transport_failed"),
                                 (response([]), "empty")):
            with self.subTest(expected=expected), TemporaryDirectory() as root, Store(Path(root) / "archive") as store:
                receipt, records, client = self.run_capture(root, store, [answer])
                self.assertEqual(receipt["status"], expected)
                self.assertEqual(len(client.calls), 1)
                self.assertEqual(records, [])

    def test_secret_echo_and_nonfinite_bodies_never_archived(self):
        secret = KEYS["APCA_API_SECRET_KEY"]
        encoded = "".join("\\u%04x" % ord(c) for c in secret)
        bodies = [json.dumps({"error": secret}).encode(), ('{"error":"' + encoded + '"}').encode(),
                  json.dumps({"error": json.dumps({"key": secret})}).encode(),
                  b'{"quotes":{"AAPL":[]},"unsafe":NaN}', b'{"quotes":{"AAPL":[]},"unsafe":1e400}']
        for body in bodies:
            with self.subTest(body=body), TemporaryDirectory() as root, Store(Path(root) / "archive") as store:
                receipt, records, _ = self.run_capture(root, store, [(200, body)])
                self.assertEqual(receipt["status"], "invalid_response")
                self.assertFalse(receipt["pages"][0]["body_archived"])
                self.assertEqual(list(store.blobs.iterdir()), [])
                self.assertEqual(store.db.execute("SELECT COUNT(*) FROM observations").fetchone()[0], 0)
                self.assertNotIn(secret, (Path(root) / "capture/receipt.json").read_text())

    def test_trades_and_raw_minute_bars_are_observations_not_fills(self):
        for kind, row in (("trades", {"t": "2026-09-17T14:00:01Z", "p": 200, "s": 100, "x": "Q", "c": ["@"], "i": 1}),
                          ("bars", {"t": "2026-09-17T14:00:00Z", "o": 200, "h": 202, "l": 199, "c": 201, "v": 100, "vw": 200.5, "n": 4})):
            with self.subTest(kind=kind), TemporaryDirectory() as root, Store(Path(root) / "archive") as store:
                receipt, records, client = self.run_capture(root, store, [response([row], kind=kind)], query(kind))
                self.assertEqual(receipt["status"], "captured")
                self.assertFalse(records[0]["is_fill"])
                self.assertTrue(records[0]["valid_numeric_observation"])
                if kind == "bars":
                    params = parse_qs(urlsplit(client.calls[0][0]).query)
                    self.assertEqual(params["adjustment"], ["raw"])
                    self.assertEqual(params["timeframe"], ["1Min"])

    def test_smoke_limits_one_page_and_does_not_claim_full_coverage(self):
        with TemporaryDirectory() as root, Store(Path(root) / "archive") as store:
            receipt, records, client = self.run_capture(root, store, [response(token="next")], smoke=True)
            self.assertEqual(receipt["purpose"], "entitlement_smoke")
            self.assertEqual(receipt["status"], "truncated")
            self.assertEqual(receipt["entitlement_status"], "accepted_for_this_request")
            self.assertEqual(len(client.calls), 1)
            self.assertFalse(receipt["complete"])

    def test_existing_capture_is_never_overwritten(self):
        with TemporaryDirectory() as root, patch("jev_alpha.alpaca_prices.read_key") as key:
            output = Path(root) / "capture"
            output.mkdir()
            with self.assertRaises(ValueError):
                collect_alpaca_prices(None, query(), output, now=NOW)
            key.assert_not_called()


class AlpacaTransportTests(unittest.TestCase):
    URL = "https://data.alpaca.markets/v2/stocks/quotes?symbols=AAPL&start=2026-01-02T15%3A00%3A00Z&end=2026-01-02T15%3A01%3A00Z&asof=-&feed=sip&currency=USD"

    def test_only_stdin_contains_headers_no_shell_redirects_or_retry(self):
        completed = subprocess.CompletedProcess([], 0, stdout=b'{}' + STATUS_MARKER + b'200')
        with patch("jev_alpha.alpaca_prices.shutil.which", return_value="curl.exe"), patch("jev_alpha.alpaca_prices.subprocess.run", return_value=completed) as run:
            self.assertEqual(CurlAlpacaTransport().get(self.URL, *KEYS.values()), (200, b'{}'))
        args, kwargs = run.call_args
        self.assertEqual(args[0], ["curl.exe", "-q", "--config", "-"])
        config = kwargs["input"].decode()
        self.assertIn("APCA-API-SECRET-KEY: synthetic-alpaca-secret", config)
        self.assertIn("retry = 0", config)
        self.assertIn("no-location", config)
        self.assertFalse(kwargs["shell"])
        self.assertEqual(kwargs["stderr"], subprocess.DEVNULL)
        self.assertNotIn(KEYS["APCA_API_SECRET_KEY"], repr(args))

    def test_transport_rejects_order_host_latest_feed_and_recent_end(self):
        for url in (self.URL.replace("data.alpaca.markets", "api.alpaca.markets"),
                    self.URL.replace("/quotes?", "/quotes/latest?"), self.URL.replace("feed=sip", "feed=iex"),
                    self.URL.replace("2026-01-02T15%3A01%3A00Z", "2999-01-02T15%3A01%3A00Z")):
            with self.subTest(url=url), patch("jev_alpha.alpaca_prices.subprocess.run") as run:
                with self.assertRaises(AlpacaTransportError):
                    CurlAlpacaTransport().get(url, *KEYS.values())
                run.assert_not_called()

    def test_transport_denial_returns_status_without_raising_body_or_retrying(self):
        completed = subprocess.CompletedProcess([], 0, stdout=b'{"message":"denied"}' + STATUS_MARKER + b'403')
        with patch("jev_alpha.alpaca_prices.shutil.which", return_value="curl.exe"), patch("jev_alpha.alpaca_prices.subprocess.run", return_value=completed) as run:
            status, body = CurlAlpacaTransport().get(self.URL, *KEYS.values())
            self.assertEqual(status, 403)
            self.assertEqual(run.call_count, 1)


if __name__ == "__main__":
    unittest.main()
