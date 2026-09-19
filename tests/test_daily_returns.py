import copy
import math
import unittest
from datetime import datetime, timezone

from jev_alpha.daily_returns import (DailyReturnsValidationError, evaluate_daily,
                                     parse_yahoo_chart)


DATES = ["2025-07-02", "2025-07-03", "2025-07-07", "2025-07-08", "2025-07-09",
         "2025-07-10", "2025-07-11", "2025-07-14", "2025-07-15", "2025-07-16",
         "2025-07-17", "2025-07-18", "2025-07-21", "2025-07-22", "2025-07-23"]


def chart(symbol="NUE", dates=None):
    dates = dates or DATES
    return {"chart": {"error": None, "result": [{
        "meta": {"symbol": symbol, "currency": "USD", "exchangeTimezoneName": "America/New_York",
                 "dataGranularity": "1d"},
        "timestamp": [int(datetime.fromisoformat(d + "T13:30:00+00:00").timestamp()) for d in dates],
        "indicators": {"quote": [{"close": [100.0] * len(dates)}],
                       "adjclose": [{"adjclose": [100.0] * len(dates)}]}}]}}


def prices():
    result = {symbol: parse_yahoo_chart(chart(symbol), symbol) for symbol in ("NUE", "STLD", "SPY")}
    for symbol, gain in (("NUE", 0.10), ("STLD", 0.20), ("SPY", 0.05)):
        result[symbol][7]["adjusted_close"] = 100 * (1 + gain)
    return result


def signal(doc="a", publication="2025-07-03", side="long", status="completed"):
    return {"document_id": doc, "publication_date": publication,
            "arms": {"jev": {"status": status, "side": side, "positive_probability": 0.8},
                     "always_long": {"status": "completed", "side": "long"},
                     "cash": {"status": "completed", "side": "cash"}}}


class YahooParserTests(unittest.TestCase):
    def test_preserves_null_adjusted_and_raw_close(self):
        value = chart()
        value["chart"]["result"][0]["indicators"]["adjclose"][0]["adjclose"][1] = None
        value["chart"]["result"][0]["indicators"]["quote"][0]["close"][2] = None
        rows = parse_yahoo_chart(value, "NUE")
        self.assertEqual(len(rows), len(DATES))
        self.assertIsNone(rows[1]["adjusted_close"])
        self.assertIsNone(rows[2]["close"])

    def test_wrong_symbol_currency_timezone_interval_rejected(self):
        for key, value in (("symbol", "TS"), ("currency", "EUR"),
                           ("exchangeTimezoneName", "Europe/London"), ("dataGranularity", "1h")):
            with self.subTest(key=key):
                data = chart()
                data["chart"]["result"][0]["meta"][key] = value
                with self.assertRaises(DailyReturnsValidationError):
                    parse_yahoo_chart(data, "NUE")

    def test_duplicate_unordered_mismatched_arrays_rejected(self):
        for mutation in ("duplicate", "order", "length"):
            data = chart()
            result = data["chart"]["result"][0]
            if mutation == "duplicate":
                result["timestamp"][1] = result["timestamp"][0]
            elif mutation == "order":
                result["timestamp"].reverse()
            else:
                result["indicators"]["adjclose"][0]["adjclose"].pop()
            with self.subTest(mutation=mutation), self.assertRaises(DailyReturnsValidationError):
                parse_yahoo_chart(data, "NUE")

    def test_nonfinite_zero_negative_bool_prices_rejected(self):
        for invalid in (math.nan, math.inf, -1, 0, True, "12"):
            data = chart()
            data["chart"]["result"][0]["indicators"]["adjclose"][0]["adjclose"][0] = invalid
            with self.subTest(value=invalid), self.assertRaises(DailyReturnsValidationError):
                parse_yahoo_chart(data, "NUE")

    def test_midnight_utc_not_silently_assigned_to_wrong_ny_session(self):
        data = chart()
        data["chart"]["result"][0]["timestamp"][0] = int(datetime(2025, 7, 2, tzinfo=timezone.utc).timestamp())
        with self.assertRaises(DailyReturnsValidationError):
            parse_yahoo_chart(data, "NUE")

    def test_provider_error_and_missing_adjusted_array_rejected(self):
        data = chart()
        data["chart"]["error"] = {"description": "unavailable"}
        with self.assertRaises(DailyReturnsValidationError):
            parse_yahoo_chart(data, "NUE")
        data = chart()
        del data["chart"]["result"][0]["indicators"]["adjclose"]
        with self.assertRaises(DailyReturnsValidationError):
            parse_yahoo_chart(data, "NUE")


class DailyEvaluationTests(unittest.TestCase):
    def test_holiday_weekend_entry_and_five_sessions(self):
        report = evaluate_daily({}, [signal()], prices())
        doc = report["documents"][0]
        self.assertEqual(doc["entry_date"], "2025-07-07")
        self.assertEqual(doc["exit_date"], "2025-07-14")
        self.assertAlmostEqual(doc["gross_market_relative_return"], 0.10)
        self.assertAlmostEqual(report["arms"]["jev"]["primary"]["all_cohort_mean"], 0.0975)
        self.assertAlmostEqual(report["arms"]["jev"]["document_mean_by_cost_bps"]["10"]["all_cohort_mean"], 0.099)
        self.assertEqual(report["arms"]["cash"]["primary"]["all_cohort_mean"], 0)

    def test_entry_strictly_after_even_when_publication_is_session(self):
        doc = evaluate_daily({}, [signal(publication="2025-07-07")], prices())["documents"][0]
        self.assertEqual(doc["entry_date"], "2025-07-08")

    def test_adjusted_prices_control_split_not_raw_close(self):
        data = prices()
        for symbol in data:
            data[symbol][2]["close"] = 200
            data[symbol][7]["close"] = 50
        result = evaluate_daily({}, [signal()], data)
        self.assertAlmostEqual(result["documents"][0]["gross_market_relative_return"], 0.10)

    def test_absent_stock_session_never_shifts_entry(self):
        data = prices()
        data["NUE"].pop(2)
        result = evaluate_daily({}, [signal()], data)
        doc = result["documents"][0]
        self.assertEqual(doc["entry_date"], "2025-07-07")
        self.assertIn("NUE_endpoint_missing", doc["price_issues"])
        self.assertIsNone(result["arms"]["jev"]["primary"]["all_cohort_mean"])
        self.assertEqual(result["arms"]["cash"]["primary"]["all_cohort_mean"], 0)

    def test_null_spy_session_retained_for_calendar(self):
        data = prices()
        data["SPY"][2]["adjusted_close"] = None
        report = evaluate_daily({}, [signal()], data)
        doc = report["documents"][0]
        self.assertEqual(doc["entry_date"], "2025-07-07")
        self.assertEqual(doc["exit_date"], "2025-07-14")
        self.assertIsNone(doc["gross_market_relative_return"])

    def test_null_stock_adjusted_never_substitutes_raw_close(self):
        data = prices()
        data["STLD"][7]["adjusted_close"] = None
        result = evaluate_daily({}, [signal()], data)
        self.assertIsNone(result["documents"][0]["gross_market_relative_return"])

    def test_cash_stays_in_primary_denominator(self):
        report = evaluate_daily({}, [signal(), signal("b", side="cash")], prices())
        arm = report["arms"]["jev"]
        self.assertEqual(arm["primary"]["total_count"], 2)
        self.assertAlmostEqual(arm["primary"]["all_cohort_mean"], 0.0975 / 2)
        self.assertEqual(arm["cash_document_count"], 1)
        self.assertEqual(arm["active_document_count"], 1)

    def test_failed_skipped_missing_signals_are_unknown_not_cash(self):
        for status in ("failed", "skipped", "missing"):
            failed = signal("b", status="failed" if status == "missing" else status)
            if status == "missing":
                del failed["arms"]["jev"]
            result = evaluate_daily({}, [signal(), failed], prices())
            arm = result["arms"]["jev"]["primary"]
            self.assertEqual(arm["total_count"], 2)
            self.assertEqual(arm["computable_count"], 1)
            self.assertIsNone(arm["all_cohort_mean"])
            self.assertAlmostEqual(arm["conditional_computable_mean"], 0.0975)

    def test_transitive_overlap_clusters_and_document_equal_weight(self):
        # Use a two-session horizon: A [Jul3,Jul8], B [Jul7,Jul9], C [Jul9,Jul11].
        rows = [signal("a", "2025-07-02"), signal("b", "2025-07-03", side="cash"),
                signal("c", "2025-07-08"), signal("d", "2025-07-17")]
        report = evaluate_daily({"horizon_sessions": 2}, rows, prices())
        self.assertEqual(report["cluster_count"], 2)
        self.assertEqual(report["clusters"][0]["document_ids"], ["a", "b", "c"])
        self.assertEqual(report["clusters"][0]["arms"]["jev"]["member_observation_active_fraction"], 2/3)
        self.assertAlmostEqual(report["clusters"][0]["arms"]["jev"]["primary_net_excess_contribution"], -0.005/3)

    def test_incomplete_end_window_remains_in_cohort_not_cluster_mean(self):
        report = evaluate_daily({}, [signal(), signal("late", "2025-07-22")], prices())
        self.assertEqual(report["document_count"], 2)
        self.assertEqual(report["ungrouped_document_ids"], ["late"])
        self.assertIsNone(report["arms"]["jev"]["primary"]["all_cohort_mean"])
        self.assertIsNone(report["arms"]["jev"]["primary_cluster"]["all_cohort_mean"])

    def test_calendar_without_prior_session_does_not_jump_to_first_available(self):
        report = evaluate_daily({}, [signal(publication="2020-01-01")], prices())
        self.assertIsNone(report["documents"][0]["entry_date"])
        self.assertIn("calendar_starts_after_publication", report["documents"][0]["price_issues"])

    def test_duplicate_cohort_and_frozen_identity_mismatch_rejected(self):
        with self.assertRaises(DailyReturnsValidationError):
            evaluate_daily({}, [signal(), signal()], prices())
        with self.assertRaises(DailyReturnsValidationError):
            evaluate_daily({"document_ids": ["another"]}, [signal()], prices())

    def test_invalid_signal_cost_price_and_probability_rejected(self):
        for protocol in ({"horizon_sessions": True}, {"primary_cost_bps": 30},
                         {"cost_scenarios_bps": [25, 25]}, {"primary_cost_bps": math.inf}):
            with self.assertRaises(DailyReturnsValidationError):
                evaluate_daily(protocol, [signal()], prices())
        bad = signal(side="short")
        with self.assertRaises(DailyReturnsValidationError):
            evaluate_daily({}, [bad], prices())
        bad = signal()
        bad["arms"]["jev"]["positive_probability"] = 1.1
        with self.assertRaises(DailyReturnsValidationError):
            evaluate_daily({}, [bad], prices())
        data = prices()
        data["NUE"].append(data["NUE"][0])
        with self.assertRaises(DailyReturnsValidationError):
            evaluate_daily({}, [signal()], data)

    def test_finite_prices_with_overflowing_ratio_rejected(self):
        data = prices()
        data["NUE"][2]["adjusted_close"] = 1e-320
        data["NUE"][7]["adjusted_close"] = 1e308
        with self.assertRaises(DailyReturnsValidationError):
            evaluate_daily({}, [signal()], data)

    def test_pure_function_does_not_mutate_inputs(self):
        data, sig, protocol = prices(), [signal()], {"document_ids": ["a"]}
        before = copy.deepcopy((data, sig, protocol))
        evaluate_daily(protocol, sig, data)
        self.assertEqual((data, sig, protocol), before)


if __name__ == "__main__":
    unittest.main()
