import contextlib
import copy
from datetime import datetime
import io
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from jev_alpha.cli import evaluate_daily_capture, main
from jev_alpha.experiment import digest
from jev_alpha.store import Store, canonical_json, read_json, write_new_json


def fixture():
    request = {"model": "typesafe/jev-1.13", "state": {}, "questions": {}}
    manifest = {"requests": [{"document_id": "2015-16067", "publication_date": "2025-07-03",
                              "request": request, "request_sha256": digest(request)}]}
    protocol = {"prediction_manifest_sha256": digest(manifest), "symbols": ["NUE", "STLD"], "benchmark": "SPY",
                "selected": [{"document_id": "2015-16067", "publication_date": "2025-07-03"}],
                "document_ids": ["2015-16067"], "arms": ["jev", "baseline", "always_long", "cash"],
                "horizon_sessions": 5, "primary_cost_bps": 25, "cost_scenarios_bps": [10, 25, 50],
                "model_policy": {"positive_probability_threshold": .70, "confidence_statistic_used": False,
                                 "provider_phase_budget_usd": 2}}
    signals = {"status": "complete", "frozen_at": "2026-09-19T00:00:00Z", "protocol_sha256": digest(protocol),
               "prediction_manifest_sha256": digest(manifest), "signals": [{"document_id": "2015-16067",
                    "publication_date": "2025-07-03", "arms": {arm: {"status": "completed", "side": "cash" if arm == "cash" else "long"}
                                                               for arm in protocol["arms"]}}]}
    return manifest, protocol, signals


def chart(symbol):
    days = ["2025-07-02", "2025-07-03", "2025-07-07", "2025-07-08", "2025-07-09",
            "2025-07-10", "2025-07-11", "2025-07-14"]
    return {"chart": {"error": None, "result": [{"meta": {"symbol": symbol, "currency": "USD",
            "exchangeTimezoneName": "America/New_York", "dataGranularity": "1d"},
            "timestamp": [int(datetime.fromisoformat(day + "T13:30:00+00:00").timestamp()) for day in days],
            "indicators": {"quote": [{"close": [100] * len(days)}],
                           "adjclose": [{"adjclose": [100] * len(days)}]}}]}}


def capture_fixture(store, directory, protocol, signals):
    records = []
    for symbol in ["NUE", "STLD", "SPY"]:
        payload = chart(symbol)
        raw_hash = store.put_blob(canonical_json(payload))
        write_new_json(directory / f"{symbol}.json", payload)
        records.append({"symbol": symbol, "status": "captured", "raw_sha256": raw_hash,
                        "source_url": f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"})
    capture = {"status": "complete", "protocol_sha256": digest(protocol), "signals_sha256": digest(signals),
               "signals_frozen_at": signals["frozen_at"], "symbols": records}
    write_new_json(directory / "capture.json", capture)
    return capture


class DailyCliTests(unittest.TestCase):
    def invoke(self, root, argv):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return main(["--data-dir", str(root / "archive"), *argv])

    def write_inputs(self, root):
        manifest, protocol, signals = fixture()
        for name, value in [("manifest", manifest), ("protocol", protocol), ("signals", signals)]:
            write_new_json(root / f"{name}.json", value)
        return protocol, signals

    def test_predict_default_is_dry_without_paid_calls(self):
        with TemporaryDirectory() as tmp, patch("jev_alpha.daily_signals.run_one") as paid:
            root = Path(tmp)
            self.write_inputs(root)
            code = self.invoke(root, ["daily-predict", "--manifest", str(root / "manifest.json"),
                                     "--protocol", str(root / "protocol.json"), "--out", str(root / "predictions")])
            self.assertEqual(code, 0)
            self.assertEqual(read_json(root / "predictions/signals.json")["status"], "dry_run")
            paid.assert_not_called()

    def test_prices_default_writes_plan_without_network(self):
        with TemporaryDirectory() as tmp, patch("jev_alpha.daily_prices.collect_daily_prices") as collector:
            root = Path(tmp)
            self.write_inputs(root)
            code = self.invoke(root, ["daily-prices", "--protocol", str(root / "protocol.json"),
                                     "--signals", str(root / "signals.json"), "--out", str(root / "price-plan")])
            self.assertEqual(code, 0)
            self.assertEqual(read_json(root / "price-plan/plan.json")["network_requests"], 0)
            collector.assert_not_called()

    def test_evaluate_valid_capture_offline_with_provenance_and_no_overwrite(self):
        with TemporaryDirectory() as tmp, patch("subprocess.run") as network:
            root = Path(tmp)
            protocol, signals = self.write_inputs(root)
            with Store(root / "archive") as store:
                capture_fixture(store, root / "prices", protocol, signals)
            args = ["daily-evaluate", "--protocol", str(root / "protocol.json"), "--signals", str(root / "signals.json"),
                    "--prices", str(root / "prices"), "--out", str(root / "report.json")]
            self.assertEqual(self.invoke(root, args), 0)
            report = read_json(root / "report.json")
            self.assertTrue(report["offline_evaluation"])
            self.assertEqual(report["protocol_sha256"], digest(protocol))
            self.assertEqual(len(report["source_provenance"]), 3)
            self.assertEqual(report["arms"]["cash"]["primary"]["all_cohort_mean"], 0)
            original = (root / "report.json").read_bytes()
            self.assertEqual(self.invoke(root, args), 2)
            self.assertEqual((root / "report.json").read_bytes(), original)
            network.assert_not_called()

    def test_evaluate_rejects_wrong_hash_incomplete_capture_and_forged_price_json(self):
        for issue in ["signal_hash", "protocol_hash", "capture_incomplete", "price_tamper", "raw_tamper"]:
            with self.subTest(issue=issue), TemporaryDirectory() as tmp:
                root = Path(tmp)
                _, protocol, signals = fixture()
                with Store(root / "archive") as store:
                    capture = capture_fixture(store, root / "prices", protocol, signals)
                    if issue == "signal_hash":
                        capture["signals_sha256"] = "f" * 64
                    elif issue == "protocol_hash":
                        capture["protocol_sha256"] = "f" * 64
                    elif issue == "capture_incomplete":
                        capture["status"] = "incomplete"
                    elif issue == "price_tamper":
                        changed = chart("NUE")
                        changed["chart"]["result"][0]["indicators"]["adjclose"][0]["adjclose"][-1] = 200
                        (root / "prices/NUE.json").write_bytes(canonical_json(changed))
                    else:
                        (store.blobs / capture["symbols"][0]["raw_sha256"]).write_bytes(b"{}")
                    (root / "prices/capture.json").write_bytes(canonical_json(capture))
                    with self.assertRaises(ValueError):
                        evaluate_daily_capture(store, protocol, signals, root / "prices")

    def test_rejects_changed_cohort_date_and_arm_before_reading_prices(self):
        for issue in ["date", "duplicate", "missing", "arm"]:
            with self.subTest(issue=issue), TemporaryDirectory() as tmp:
                _, protocol, signals = fixture()
                if issue == "date":
                    signals["signals"][0]["publication_date"] = "2025-07-04"
                elif issue == "duplicate":
                    signals["signals"].append(copy.deepcopy(signals["signals"][0]))
                elif issue == "missing":
                    signals["signals"] = []
                else:
                    del signals["signals"][0]["arms"]["jev"]
                with Store(Path(tmp) / "archive") as store, self.assertRaises(ValueError):
                    evaluate_daily_capture(store, protocol, signals, Path(tmp) / "nonexistent")

    def test_fully_attempted_failures_stay_unknown_in_primary(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, protocol, signals = fixture()
            signals["status"] = "complete_with_failures"
            signals["signals"][0]["arms"]["baseline"] = {"status": "failed", "side": None}
            with Store(root / "archive") as store:
                capture_fixture(store, root / "prices", protocol, signals)
                report = evaluate_daily_capture(store, protocol, signals, root / "prices")
            self.assertIsNone(report["arms"]["baseline"]["primary"]["all_cohort_mean"])
            self.assertEqual(report["arms"]["baseline"]["unknown_signal_count"], 1)

    def test_live_prices_rejects_unattempted_signals_before_network(self):
        with TemporaryDirectory() as tmp, patch("jev_alpha.daily_prices.collect_daily_prices") as collector:
            root = Path(tmp)
            self.write_inputs(root)
            signals = read_json(root / "signals.json")
            signals["status"] = "incomplete"
            (root / "signals.json").write_bytes(canonical_json(signals))
            code = self.invoke(root, ["daily-prices", "--live", "--protocol", str(root / "protocol.json"),
                                     "--signals", str(root / "signals.json"), "--out", str(root / "prices")])
            self.assertEqual(code, 2)
            collector.assert_not_called()

    def test_resume_argument_forwarded_explicitly(self):
        with TemporaryDirectory() as tmp, patch("jev_alpha.daily_signals.predict_daily") as predict:
            root = Path(tmp)
            _, signals = self.write_inputs(root)
            predict.return_value = {"status": "dry_run", "signals": []}
            code = self.invoke(root, ["daily-predict", "--manifest", str(root / "manifest.json"),
                                     "--protocol", str(root / "protocol.json"), "--out", str(root / "resumed"),
                                     "--resume-signals", str(root / "signals.json")])
            self.assertEqual(code, 0)
            self.assertEqual(predict.call_args.kwargs["resume"], signals)
            self.assertFalse(predict.call_args.kwargs["live"])


if __name__ == "__main__":
    unittest.main()
