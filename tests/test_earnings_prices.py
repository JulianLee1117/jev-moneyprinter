from datetime import datetime, timezone
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from urllib.parse import parse_qs, urlsplit

from jev_alpha.earnings_prices import (ACTION_URL, CALENDAR_URL, CurlReferenceTransport,
    calendar_sessions, capture_market_inputs, event_sessions)
from jev_alpha.alpaca_prices import AlpacaTransportError
from jev_alpha.store import Store

NOW = datetime(2026, 9, 19, 12, tzinfo=timezone.utc)
KEYS = {"APCA_API_KEY_ID": "synthetic-study-id", "APCA_API_SECRET_KEY": "synthetic-study-secret"}


class Transport:
    def __init__(self, payloads):
        self.payloads = list(payloads)
        self.urls = []

    def get(self, url, *_keys):
        self.urls.append(url)
        return self.payloads.pop(0)


class EarningsPriceTests(unittest.TestCase):
    def capture(self, root, store, query, payloads):
        client = Transport([(status, json.dumps(body).encode()) for status, body in payloads])
        with patch.dict(os.environ, KEYS, clear=True):
            receipt = capture_market_inputs(store, query, Path(root) / "capture", transport=client, now=NOW)
        return receipt, client

    def test_daily_sip_adjustment_asof_and_pagination_remain_explicit(self):
        for adjustment in ("raw", "all"):
            with self.subTest(adjustment=adjustment), TemporaryDirectory() as root, Store(Path(root) / "archive") as store:
                receipt, client = self.capture(root, store,
                    {"kind": "daily_bars", "symbol": "AAPL", "start": "2026-01-01", "end": "2026-03-31",
                     "asof": "2026-03-31", "adjustment": adjustment},
                    [(200, {"bars": {"AAPL": [{"t": "2026-01-02T05:00:00Z", "c": 200}]}, "next_page_token": "next"}),
                     (200, {"bars": {"AAPL": []}, "next_page_token": None})])
                self.assertTrue(receipt["complete"])
                first, second = [parse_qs(urlsplit(url).query) for url in client.urls]
                self.assertEqual(first["feed"], ["sip"])
                self.assertEqual(first["currency"], ["USD"])
                self.assertEqual(first["timeframe"], ["1Day"])
                self.assertEqual(first["adjustment"], [adjustment])
                self.assertEqual(first["asof"], ["2026-03-31"])
                self.assertEqual(second["page_token"], ["next"])
                self.assertEqual(len(json.loads((Path(root) / "capture/payloads.json").read_text())), 2)

    def test_actions_preserve_process_date_semantics_and_truncation_is_unknown(self):
        with TemporaryDirectory() as root, Store(Path(root) / "archive") as store:
            receipt, client = self.capture(root, store,
                {"kind": "corporate_actions", "symbol": "AAPL", "start": "2016-01-01", "end": "2026-09-18", "max_pages": 1},
                [(200, {"corporate_actions": {"cash_dividends": []}, "next_page_token": "more"})])
            self.assertFalse(receipt["complete"])
            self.assertEqual(receipt["status"], "truncated")
            self.assertIn("process-date", receipt["corporate_action_coverage"])
            params = parse_qs(urlsplit(client.urls[0]).query)
            self.assertEqual(params["data_quality"], ["all"])
            self.assertNotIn("types", params)

    def test_calendar_handles_holiday_dst_and_exact_five_session_exit(self):
        dates = ["2026-03-06", "2026-03-09", "2026-03-10", "2026-03-11", "2026-03-12",
                 "2026-03-13", "2026-03-16", "2026-03-17"]
        rows = [{"date": d, "open": "09:30", "close": "16:00"} for d in dates]
        sessions = calendar_sessions(rows, dates[0], dates[-1])
        # Saturday UTC is still Friday NY; DST changes on Sunday.
        plan = event_sessions("2026-03-07T01:00:00Z", sessions)
        self.assertEqual(plan["acceptance_new_york_date"], "2026-03-06")
        self.assertEqual(plan["entry_at"], "2026-03-10T09:35:00-04:00")
        self.assertEqual(plan["exit_at"], "2026-03-17T09:35:00-04:00")
        with self.assertRaises(ValueError):
            event_sessions("2026-03-07T01:00:00Z", sessions[:-1])
        with TemporaryDirectory() as root, Store(Path(root) / "archive") as store:
            receipt, _ = self.capture(root, store,
                {"kind": "calendar", "start": dates[0], "end": dates[-1]}, [(200, rows)])
            self.assertTrue(receipt["complete"])
        # Thanksgiving omitted and the following session closes early.
        holiday = calendar_sessions([{"date": "2025-11-26", "open": "09:30", "close": "16:00"},
            {"date": "2025-11-28", "open": "09:30", "close": "13:00"}], "2025-11-26", "2025-11-28")
        self.assertTrue(holiday[-1]["close"].endswith("T13:00:00-05:00"))

    def test_missing_denied_and_credential_echo_do_not_become_empty_success(self):
        for mode in ("missing", "denied", "echo"):
            with self.subTest(mode=mode), TemporaryDirectory() as root, Store(Path(root) / "archive") as store:
                client = Transport([(403, b'{"message":"denied"}')] if mode == "denied" else
                                   [(200, json.dumps({"message": KEYS["APCA_API_SECRET_KEY"]}).encode())])
                with patch.dict(os.environ, {} if mode == "missing" else KEYS, clear=True):
                    receipt = capture_market_inputs(store,
                        {"kind": "calendar", "start": "2026-03-01", "end": "2026-03-31"},
                        Path(root) / "capture", env_path=Path(root) / "missing", transport=client, now=NOW)
                self.assertFalse(receipt["complete"])
                self.assertEqual(receipt["status"], {"missing": "missing_credentials", "denied": "provider_denied", "echo": "invalid_response"}[mode])
                if mode == "missing":
                    self.assertFalse(client.urls)
                if mode == "echo":
                    self.assertFalse(list(store.blobs.iterdir()))

    def test_reference_transport_only_gets_allowlisted_endpoints_keys_via_stdin(self):
        class Completed:
            returncode = 0
            stdout = b'[]\n__ALPACA_HTTP_STATUS__:200'
        client = CurlReferenceTransport()
        url = CALENDAR_URL + "?start=2026-03-01&end=2026-03-31&date_type=TRADING"
        with patch("jev_alpha.earnings_prices.shutil.which", return_value="curl"), patch("jev_alpha.earnings_prices.subprocess.run", return_value=Completed()) as run:
            self.assertEqual(client.get(url, *KEYS.values()), (200, b"[]"))
            args, kwargs = run.call_args
            self.assertEqual(args[0], ["curl", "-q", "--config", "-"])
            self.assertNotIn(KEYS["APCA_API_SECRET_KEY"], str(args))
            self.assertIn(KEYS["APCA_API_SECRET_KEY"].encode(), kwargs["input"])
            self.assertIn(b"no-location", kwargs["input"])
            for wrong in (url.replace("calendar", "orders"), url.replace("paper-api", "api"),
                          url.replace("date_type=TRADING", "date_type=SETTLEMENT"), ACTION_URL + "?symbols=AAPL&start=2026-03-01&end=2026-03-31"):
                with self.assertRaises(AlpacaTransportError):
                    client.get(wrong, *KEYS.values())
            self.assertEqual(run.call_count, 1)


if __name__ == "__main__":
    unittest.main()
