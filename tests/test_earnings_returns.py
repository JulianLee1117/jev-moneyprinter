import unittest

from jev_alpha.earnings_returns import daily_proxy, quote_outcome, semantic_score, simulate_ledger


DAYS = ["2026-04-01", "2026-04-02", "2026-04-06", "2026-04-07", "2026-04-08",
        "2026-04-09", "2026-04-10", "2026-04-13", "2026-04-14", "2026-04-15",
        "2026-04-16", "2026-04-17", "2026-04-20"]


def event(event_id, entry=0, score=0.5, entry_price=100, exit_price=110, **extra):
    return {"event_id": event_id, "entry_date": DAYS[entry], "exit_date": DAYS[entry + 5],
            "entry_price": entry_price, "exit_price": exit_price, "scores": {"jev": score},
            "corporate_action_status": "none", **extra}


class EarningsReturnsTests(unittest.TestCase):
    def test_fixed_denominator_and_invalid_outputs_not_cash(self):
        ids = [str(i) for i in range(8)]
        answers = {i: {"choice": "unknown", "probabilities": {
            "favourable": 0, "adverse": 0, "unchanged": 0, "mixed": 0, "unknown": 1}} for i in ids}
        answers["0"] = {"choice": "favourable", "probabilities": {
            "favourable": .7, "adverse": .2, "unchanged": 0, "mixed": 0, "unknown": .1}}
        self.assertAlmostEqual(semantic_score(answers, ids), .5 / 8)
        self.assertEqual(semantic_score(answers, ids, False), 1 / 8)
        with self.assertRaises(ValueError):
            semantic_score({k: v for k, v in answers.items() if k != "7"}, ids)
        answers["0"]["probabilities"]["favourable"] = .69
        self.assertAlmostEqual(semantic_score(answers, ids), .49 / 8)
        self.assertEqual(answers["0"]["probabilities"]["favourable"], .69)
        answers["0"]["probabilities"]["favourable"] = .9
        with self.assertRaises(ValueError):
            semantic_score(answers, ids)

    def test_whole_share_costs_and_distinct_missing_daily_proxy(self):
        result = quote_outcome(100, 110, 50, 51, slippage_bps=0)
        self.assertEqual(result["stock"]["shares"], 9)
        self.assertEqual(result["stock"]["pnl_usd"], 89)
        self.assertEqual(result["benchmark"]["pnl_usd"], 18)
        self.assertAlmostEqual(result["excess_account_return"], .071)
        self.assertFalse(result["is_fill"])
        self.assertIsNone(quote_outcome(None, 110, 50, 51)["stock"])
        self.assertIsNone(daily_proxy(100, None, 50, 51)["excess_return"])
        self.assertAlmostEqual(daily_proxy(100, 110, 50, 51)["excess_return"], .08)

    def test_overlap_priority_settlement_and_dividend_not_reinvested(self):
        events = [event("b", score=.8), event("a", score=.8, corporate_action_status="cash_dividend",
                                                       cash_dividend_per_entry_share=1),
                  event("exit-day", entry=5, score=1), event("settled", entry=6, exit_price=90)]
        result = simulate_ledger(events, "jev", DAYS, slippage_bps=0)
        self.assertEqual([t["event_id"] for t in result["trades"]], ["a", "settled"])
        self.assertEqual(result["trades"][1]["shares"], 10)
        self.assertEqual(result["dividend_receivable_usd"], 9)
        self.assertEqual(result["final_settled_cash_usd"], 988)
        self.assertEqual(result["final_equity_usd"], 997)
        self.assertTrue(any(d["event_id"] == "exit-day" and d["status"] ==
                            "capital_unavailable_until_settlement" for d in result["decisions"]))

    def test_selected_future_missingness_halts_without_choosing_clean_alternative(self):
        for change, reason in (({"corporate_action_status": "unsupported"}, "corporate_actions"),
                               ({"exit_price": None}, "exit_price"),
                               ({"scores": {"jev": None}}, "signal")):
            result = simulate_ledger([event("high", score=.9, **change), event("low", score=.1)],
                                     "jev", DAYS)
            self.assertEqual(result["status"], "unknown")
            self.assertEqual(result["unknown_event_id"], "high")
            self.assertIn(reason, result["unknown_reason"])
            self.assertIsNone(result["pnl_usd"])
        negative = [event("negative", score=-.1)]
        self.assertEqual(simulate_ledger(negative, "jev", DAYS)["final_equity_usd"], 1000)
        self.assertEqual(len(simulate_ledger(negative, "jev", DAYS, force_long=True)["trades"]), 1)

    def test_displayed_sizes_cover_actual_compounded_quantity_not_fixed_account_proxy(self):
        first = event("first", exit_price=200)
        second = event("second", entry=6)
        complete = simulate_ledger([first, second], "jev", DAYS, slippage_bps=0)
        self.assertEqual(complete["trades"][1]["shares"], 18)
        self.assertEqual(quote_outcome(100, 110, 50, 51, slippage_bps=0)["stock"]["shares"], 9)
        for field in ("entry_available_shares", "exit_available_shares"):
            for size in (12, None, 0):
                with self.subTest(field=field, size=size):
                    result = simulate_ledger([first, {**second, field: size},
                                              event("lower-priority", entry=6, score=.1)],
                                             "jev", DAYS, slippage_bps=0)
                    self.assertEqual(result["status"], "unknown")
                    self.assertEqual(result["unknown_event_id"], "second")
                    self.assertIn(field.split("_")[0] + "_size", result["unknown_reason"])
                    self.assertIsNone(result["final_equity_usd"])
                    self.assertEqual([t["event_id"] for t in result["trades"]], ["first"])


if __name__ == "__main__":
    unittest.main()
