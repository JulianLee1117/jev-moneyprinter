import copy
from datetime import date, datetime, timedelta
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from jev_alpha.earnings_prices import calendar_sessions
from jev_alpha.procurement_returns import (UNIVERSE, _digest, _frozen, _quote, capture_prices,
                                          evaluate, plan_windows)


def calendar():
    rows, day = [], date(2025, 8, 1)
    while day <= date(2025, 12, 15):
        if day.weekday() < 5 and day.isoformat() not in {"2025-09-01", "2025-11-27"}:
            rows.append({"date": day.isoformat(), "open": "09:30",
                         "close": "13:00" if day.isoformat() == "2025-11-28" else "16:00"})
        day += timedelta(days=1)
    return calendar_sessions(rows, rows[0]["date"], rows[-1]["date"])


def event(name="one", day="2025-10-01", **updates):
    return {"event_id": name, "project_id": name, "symbol": "ORN", "source_id": name,
            "association_date": day, "availability": "unverified_historical", "split": "holdout",
            "arms": {"jev": {"status": "signal", "materiality": .03}}, **updates}


def fixture(events):
    signals = {"frozen_at": "2026-09-01T00:00:00Z", "events": events,
               "arm_costs_usd": {"jev": .25},
               "arm_costs_by_split_usd": {"development": {"jev": .05}, "holdout": {"jev": .2}},
               "model_execution_failures": {"jev": 0},
               "model_execution_failures_by_split": {"development": {"jev": 0}, "holdout": {"jev": 0}},
               "research_cost_usd": 2}
    sessions = calendar()
    market = {"signals_sha256": _frozen(signals), "calendar": sessions, "quotes": {}, "bars": {}, "actions": {}}
    for symbol in (*UNIVERSE, "IWM"):
        market["bars"][symbol] = {"status": "captured", "records": [
            {"date": row["date"], "close": 100, "volume": 20000} for row in sessions]}
    for window in plan_windows(events, sessions):
        if window["status"] != "planned":
            continue
        for symbol in (*UNIVERSE, "IWM"):
            for key in ("entry_at", "exit_at"):
                stamp = window[key]
                mid = 110 if symbol == "ORN" and key == "exit_at" else 100
                market["quotes"][symbol + "|" + stamp] = {"status": "captured", "records": [{
                    "timestamp": stamp, "bid": mid - .1, "ask": mid + .1,
                    "bid_size": 1000, "ask_size": 1000, "conditions": ["R"]}]}
            action_key = "|".join((symbol, window["entry_at"][:10], window["exit_at"][:10]))
            market["actions"][action_key] = {"status": "none", "coverage_complete": True,
                                            "evidence": ["offline independently reviewed fixture"]}
    return signals, market


class ProcurementReturnsTests(unittest.TestCase):
    def test_clock_boundaries_dst_halfday_and_no_invented_intraday(self):
        rows = plan_windows([event(day="20251031")], calendar())
        self.assertEqual(rows[0]["entry_at"], "2025-11-03T09:45:00-05:00")
        self.assertEqual(rows[0]["exit_at"], "2025-11-10T09:45:00-05:00")
        self.assertEqual(rows[0]["scope"], "historical_association")
        self.assertEqual(rows[1]["status"], "unavailable")
        late = event(day="2025-11-28", availability="prospective_first_seen",
                     published_at="2025-11-28T17:58:00Z",
                     arms={"jev": {"status": "signal", "materiality": .03,
                                   "ready_at": "2025-11-28T17:58:30Z"}})
        primary, intraday = plan_windows([late], calendar())
        self.assertEqual(primary["entry_at"], "2025-12-01T09:45:00-05:00")
        self.assertEqual(intraday["entry_at"], primary["entry_at"])
        self.assertEqual(intraday["exit_at"], primary["exit_at"])
        late["arms"]["jev"]["ready_at"] = "2025-11-28T17:57:00Z"
        self.assertEqual(plan_windows([late], calendar())[1]["status"], "unavailable")
        late.update(availability="verified_historical")
        late["arms"]["jev"]["ready_at"] = "2026-09-01T10:00:00Z"
        self.assertEqual(plan_windows([late], calendar())[0]["status"], "unavailable")
        late["arms"]["jev"]["counterfactual_ready_at"] = "2025-12-01T18:00:00Z"
        self.assertEqual(plan_windows([late], calendar())[0]["entry_at"], "2025-12-02T09:45:00-05:00")
        observed = event(day="2025-10-01", availability="prospective_first_seen", published_at=None,
                         first_seen_at="2025-10-01T15:00:00Z",
                         arms={"jev": {"status": "signal", "materiality": .03, "ready_at": "2025-10-01T15:01:00Z"}})
        primary, intraday = plan_windows([observed], calendar())
        self.assertEqual(intraday["entry_at"], "2025-10-01T11:02:00-04:00")
        self.assertIsNone(intraday["published_at"])
        self.assertEqual(intraday["first_seen_at"], "2025-10-01T15:00:00Z")
        observed["arms"]["jev"]["ready_at"] = "2025-10-01T14:59:00Z"
        self.assertEqual(plan_windows([observed], calendar())[0]["status"], "unavailable")
        observed.pop("first_seen_at")
        self.assertEqual(plan_windows([observed], calendar())[1]["status"], "unavailable")

    def test_dry_capture_never_uses_network_and_rejects_tampered_freeze(self):
        signals, _ = fixture([event()])
        with tempfile.TemporaryDirectory() as root, patch("jev_alpha.procurement_returns.capture_market_inputs") as network:
            result = capture_prices(root, signals, {"calendar": calendar()})
            self.assertEqual(result["status"], "planned")
            self.assertEqual(len(result["windows"]), 2)
            network.assert_not_called()
            self.assertFalse((Path(root) / "market").exists())
            signals["sha256"] = _digest(signals)
            signals["events"][0]["symbol"] = "TPC"
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                capture_prices(root, signals, {})

    def test_verified_historical_units_and_post_switch_first_sufficient_quote(self):
        for day, first_sufficient in (("2025-10-30", True), ("2025-10-31", False)):
            signals, market = fixture([event(day=day)])
            window = plan_windows(signals["events"], market["calendar"])[0]
            capture = market["quotes"]["ORN|" + window["entry_at"]]
            if first_sufficient:
                capture["size_unit_assessment"] = {"verified": True, "unit": "round_lots",
                    "round_lot_shares": 100, "evidence": ["offline explicitly verified unit fixture"]}
            capture["records"][0]["ask_size"] = 1
            later = copy.deepcopy(capture["records"][0])
            later.update(timestamp=(datetime.fromisoformat(window["entry_at"]) + timedelta(seconds=10)).isoformat(), ask_size=1000)
            capture["records"].append(later)
            result = evaluate(signals, market, {})
            account = result["accounts"]["jev|manual|all"]["10"]
            self.assertEqual(account["status"], "complete")
            expected = window["entry_at"] if first_sufficient else later["timestamp"]
            self.assertEqual(account["trades"][0]["entry_quote_at"], expected)
            if not first_sufficient:
                capture["records"].pop()
                account = evaluate(signals, market, {})["accounts"]["jev|manual|all"]["10"]
                self.assertEqual(account["reason"], "entry_quote_unavailable_or_insufficient_size")
                self.assertIsNone(account["pnl_usd"])

    def test_unverified_historical_units_allow_same_first_quote_without_invented_size(self):
        signals, market = fixture([event()])
        window = plan_windows(signals["events"], market["calendar"])[0]
        before = copy.deepcopy(market)
        diagnostic = {}
        quote = _quote(market, "ORN", window["entry_at"], capital=1000, bps=10, fee=1,
                       diagnostic=diagnostic)
        self.assertEqual(quote["timestamp"], window["entry_at"])
        self.assertIsNone(quote["size_multiplier"])
        self.assertIsNone(quote["ask_shares"])
        self.assertEqual(quote["size_multipliers_checked"], [1, 100])
        self.assertTrue(quote["first_sufficient_quote_invariant"])
        self.assertEqual(diagnostic["status"], "usable")
        account = evaluate(signals, market, {})["accounts"]["jev|manual|all"]["10"]
        self.assertEqual(account["status"], "complete")
        self.assertEqual(account["trades"][0]["entry_quote_at"], window["entry_at"])
        self.assertEqual(account["trades"][0]["entry_quote_size_unit_status"],
                         "unverified_historical_units_first_quote_invariant")
        self.assertEqual(market, before)

    def test_ambiguous_earlier_quote_is_not_replaced_by_later_sufficient_quote(self):
        signals, market = fixture([event()])
        window = plan_windows(signals["events"], market["calendar"])[0]
        capture = market["quotes"]["ORN|" + window["entry_at"]]
        first = capture["records"][0]
        first["ask_size"] = 1
        later = {**first, "timestamp": (datetime.fromisoformat(first["timestamp"]) + timedelta(seconds=10)).isoformat(),
                 "ask_size": 1000, "ask": 100.05}
        capture["records"].append(later)
        diagnostic = {}
        self.assertIsNone(_quote(market, "ORN", window["entry_at"], capital=1000, bps=10, fee=1,
                                 diagnostic=diagnostic))
        self.assertEqual(diagnostic["reason"], "quote_size_unit_ambiguous_first_sufficient")
        self.assertEqual(diagnostic["quote_at"], first["timestamp"])
        self.assertEqual(diagnostic["sufficient_by_interpretation"], [False, True])
        self.assertFalse(diagnostic["later_quote_substitution_permitted"])
        account = evaluate(signals, market, {})["accounts"]["jev|manual|all"]["10"]
        self.assertEqual(account["reason"], "entry_quote_size_unit_ambiguous_first_sufficient")
        self.assertEqual(account["trades"], [])
        self.assertIsNone(account["pnl_usd"])

    def test_missing_or_unverified_exit_units_leave_pnl_unknown_and_model_cost_intact(self):
        signals, market = fixture([event()])
        window = plan_windows(signals["events"], market["calendar"])[0]
        capture = market["quotes"]["ORN|" + window["exit_at"]]
        capture["records"][0]["bid_size"] = 1
        for assessment in (None, {"verified": False, "unit": "round_lots", "round_lot_shares": 100}):
            with self.subTest(assessment=assessment):
                if assessment is None:
                    capture.pop("size_unit_assessment", None)
                else:
                    capture["size_unit_assessment"] = assessment
                result = evaluate(signals, market, {})
                account = result["accounts"]["jev|manual|all"]["10"]
                self.assertEqual(account["status"], "unknown")
                self.assertEqual(account["reason"], "exit_quote_size_unit_ambiguous_first_sufficient")
                self.assertIsNone(account["pnl_usd"])
                self.assertIsNone(account["pnl_after_operating_cost_usd"])
                self.assertEqual(account["operating_cost_usd"], .25)
                self.assertEqual(account["quote_diagnostic"]["raw_bid_size"], 1)
                self.assertIsNone(result["windows"][0]["outcome"]["return"])

    def test_corporate_actions_and_missing_peer_are_not_zero_returns(self):
        signals, market = fixture([event()])
        window = plan_windows(signals["events"], market["calendar"])[0]
        peer_key = "GVA|" + window["exit_at"]
        del market["quotes"][peer_key]
        result = evaluate(signals, market, {})
        row = result["windows"][0]
        self.assertIsNone(row["peer_excess"])
        self.assertIsNotNone(row["iwm_excess"])
        self.assertIn("GVA", row["missing_benchmarks"])
        action_key = "|".join(("ORN", window["entry_at"][:10], window["exit_at"][:10]))
        market["actions"][action_key]["evidence"] = []
        result = evaluate(signals, market, {})
        self.assertEqual(result["accounts"]["jev|manual|all"]["10"]["status"], "unknown")
        self.assertIsNone(result["windows"][0]["outcome"]["return"])

    def test_t_plus_one_cash_priority_dividend_and_corpus_cost(self):
        events = [event("first"), event("exit-day", "2025-10-08"), event("settled", "2025-10-09")]
        signals, market = fixture(events)
        first = plan_windows(events, market["calendar"])[0]
        action_key = "|".join(("ORN", first["entry_at"][:10], first["exit_at"][:10]))
        market["actions"][action_key].update(status="cash_dividend", cash_dividend_per_entry_share=1)
        result = evaluate(signals, market, {})
        account = result["accounts"]["jev|manual|all"]["10"]
        self.assertEqual([t["event_id"] for t in account["trades"]], ["first", "settled"])
        self.assertTrue(any(r["event_id"] == "exit-day" and r["reason"] == "capital_unavailable" for r in account["skipped"]))
        self.assertEqual(account["dividend_receivable_usd"], 9)
        self.assertAlmostEqual(account["pnl_usd"] - account["pnl_after_operating_cost_usd"], .25)
        self.assertEqual(result["decisions"]["jev"]["decision"], "descriptive_insufficient_sample")
        self.assertFalse(result["decisions"]["jev"]["proven_alpha"])

    def test_unknown_high_priority_is_not_replaced_with_clean_low_priority(self):
        high, low = event("high"), event("low", symbol="TPC")
        high["arms"]["jev"]["materiality"] = .1
        low["arms"]["jev"]["materiality"] = .02
        signals, market = fixture([high, low])
        window = plan_windows([high], market["calendar"])[0]
        del market["quotes"]["ORN|" + window["exit_at"]]
        result = evaluate(signals, market, {})
        account = result["accounts"]["jev|manual|all"]["10"]
        self.assertEqual(account["status"], "unknown")
        self.assertEqual(account["unknown_event_id"], "high")
        self.assertEqual(account["trades"], [])

    def test_simultaneous_lower_priority_opportunity_is_retained(self):
        high, low = event("high"), event("low", symbol="TPC")
        high["arms"]["jev"]["materiality"] = .1
        low["arms"]["jev"]["materiality"] = .02
        signals, market = fixture([low, high])
        account = evaluate(signals, market, {})["accounts"]["jev|manual|all"]["10"]
        self.assertEqual(account["status"], "complete")
        self.assertEqual([t["event_id"] for t in account["trades"]], ["high"])
        self.assertEqual(account["skipped"], [
            {"event_id": "low", "reason": "simultaneous_lower_priority"}])

    def test_missing_operating_cost_and_unmapped_failure_cannot_clear_profit_gate(self):
        signals, market = fixture([event()])
        signals["arm_costs_usd"] = {}
        market["signals_sha256"] = _frozen(signals)
        account = evaluate(signals, market, {})["accounts"]["jev|manual|all"]["10"]
        self.assertIsNotNone(account["pnl_usd"])
        self.assertIsNone(account["pnl_after_operating_cost_usd"])
        signals["unmapped_unknown_packets"] = {"jev": 1}
        market["signals_sha256"] = _frozen(signals)
        account = evaluate(signals, market, {})["complete_evidence_accounts"]["jev|manual|all"]["10"]
        self.assertEqual(account["status"], "unknown")
        self.assertIsNone(account["pnl_usd"])

    def test_prior_close_eligibility_and_split_cost_unknown_scope(self):
        signals, market = fixture([event()])
        window = plan_windows(signals["events"], market["calendar"])[0]
        previous = [r for r in market["bars"]["ORN"]["records"] if r["date"] < window["entry_at"][:10]][-1]
        previous["close"] = 4.99
        report = evaluate(signals, market, {})
        self.assertEqual(report["windows"][0]["outcome"]["status"], "ineligible")
        previous["close"] = 100
        signals["unmapped_unknown_packets"] = {"jev": 1}
        signals["unmapped_unknown_by_split"] = {"development": {"jev": 1}, "holdout": {"jev": 0}}
        market["signals_sha256"] = _frozen(signals)
        report = evaluate(signals, market, {})
        self.assertEqual(report["complete_evidence_accounts"]["jev|manual|development"]["10"]["status"], "unknown")
        self.assertEqual(report["accounts"]["jev|manual|development"]["10"]["status"], "complete")
        holdout = report["accounts"]["jev|manual|holdout"]["10"]
        self.assertEqual(holdout["status"], "complete")
        self.assertAlmostEqual(holdout["pnl_usd"] - holdout["pnl_after_operating_cost_usd"], .2)

    def test_empty_census_still_reports_registered_arms(self):
        with tempfile.TemporaryDirectory() as tmp:
            empty = {"frozen_at":"2026-09-01T00:00:00Z", "events":[]}
            with patch("jev_alpha.procurement_returns.capture_market_inputs") as network:
                receipt = capture_prices(Path(tmp), empty, {}, live=True)
                self.assertEqual(receipt["status"], "no_events")
                self.assertTrue(Path(tmp,"market","manifest.json").exists())
                network.assert_not_called()
        signals, market = fixture([])
        report = evaluate(signals, market, {"arms": ["rules", "nano", "mini", "jev"]})
        self.assertEqual(set(report["decisions"]), {"rules", "nano", "mini", "jev"})
        self.assertEqual(report["decisions"]["jev"]["census_projects"], 0)
        self.assertEqual(report["decisions"]["jev"]["decision"], "descriptive_insufficient_sample")

    def test_observed_decisions_abstain_semantic_unknown_but_not_missing_selected_price(self):
        unresolved = event("uncertain", arms={"jev": {"status": "unknown", "materiality": None}})
        signals, market = fixture([unresolved, event()])
        signals["unmapped_unknown_packets"] = {"jev": 2}
        market["signals_sha256"] = _frozen(signals)
        report = evaluate(signals, market, {})
        self.assertEqual(report["complete_evidence_accounts"]["jev|manual|all"]["10"]["status"], "unknown")
        observed = report["accounts"]["jev|manual|all"]["10"]
        self.assertEqual(observed["status"], "complete")
        self.assertEqual(observed["semantic_unknown_abstention_count"], 1)
        self.assertEqual(observed["unmapped_unknown_count"], 2)
        self.assertEqual([r["event_id"] for r in observed["trades"]], ["one"])
        self.assertAlmostEqual(observed["pnl_usd"] - observed["pnl_after_operating_cost_usd"], .25)
        self.assertFalse(observed["research_gate_eligible_by_itself"])
        self.assertIsNone(next(w for w in report["windows"] if w["event_id"] == "uncertain")["outcome"])
        signals["model_execution_failures"]["jev"] = 1
        signals["model_execution_failures_by_split"]["holdout"]["jev"] = 1
        market["signals_sha256"] = _frozen(signals)
        failed = evaluate(signals, market, {"minimum_projects": 1, "minimum_issuers": 1})
        self.assertEqual(failed["accounts"]["jev|manual|all"]["10"]["status"], "complete")
        self.assertEqual(failed["decisions"]["jev"]["decision"], "inconclusive_model_execution_coverage")
        selected = plan_windows([event()], market["calendar"])[0]
        del market["quotes"]["ORN|" + selected["exit_at"]]
        observed = evaluate(signals, market, {})["observed_decision_accounts"]["jev|manual|all"]["10"]
        self.assertEqual(observed["status"], "unknown")
        self.assertIsNone(observed["pnl_usd"])

    def test_stress_costs_do_not_cancel_against_gross_benchmark_reference(self):
        signals, market = fixture([event()])
        window = plan_windows(signals["events"], market["calendar"])[0]
        # Every security has the same 10% gross midpoint appreciation.
        for symbol in (*UNIVERSE, "IWM"):
            market["quotes"][symbol + "|" + window["exit_at"]]["records"][0].update(bid=109.9, ask=110.1)
        row = evaluate(signals, market, {})["windows"][0]
        self.assertAlmostEqual(row["benchmark_references"]["IWM"]["return"], .1)
        self.assertEqual(row["benchmark_references"], row["stress_benchmark_references"])
        self.assertLess(row["iwm_excess"], 0)
        self.assertLess(row["stress_iwm_excess"], row["iwm_excess"])
        self.assertAlmostEqual(row["peer_excess"], row["iwm_excess"])
        self.assertAlmostEqual(row["stress_peer_excess"], row["stress_iwm_excess"])

    def test_positive_peer_excess_cannot_replace_registered_iwm_benchmark(self):
        events = [event("one", "2025-10-01"), event("two", "2025-10-15", symbol="TPC")]
        signals, market = fixture(events)
        windows = [w for w in plan_windows(events, market["calendar"]) if w["horizon"] == "manual"]
        for window in windows:
            for symbol in (*UNIVERSE, "IWM"):
                mid = 120 if symbol == "IWM" else 110 if symbol == window["symbol"] else 100
                market["quotes"][symbol + "|" + window["exit_at"]]["records"][0].update(
                    bid=mid - .1, ask=mid + .1)
        protocol = {"benchmark": "IWM", "minimum_projects": 2, "minimum_issuers": 2}
        decision = evaluate(signals, market, protocol)["decisions"]["jev"]
        self.assertGreater(decision["base_holdout_account_pnl_after_operating_cost_usd"], 0)
        self.assertGreater(decision["base_holdout_mean_peer_excess"], .005)
        self.assertLess(decision["base_holdout_mean_iwm_excess"], 0)
        self.assertEqual(decision["primary_benchmark"], "IWM")
        self.assertEqual(decision["decision"], "reject_frozen_rule")
        del market["quotes"]["IWM|" + windows[0]["exit_at"]]
        missing = evaluate(signals, market, protocol)["decisions"]["jev"]
        self.assertEqual(missing["decision"], "inconclusive_missing_data")

    def test_full_census_counts_and_stress_concentration_are_explicit(self):
        development = event("development", "2025-09-02", split="development", symbol="TPC")
        signals, market = fixture([development, event()])
        protocol = {"minimum_projects": 2, "minimum_issuers": 2}
        report = evaluate(signals, market, protocol)
        decision = report["decisions"]["jev"]
        self.assertEqual(decision["census_projects"], 2)
        self.assertEqual(decision["census_issuers"], 2)
        self.assertEqual(decision["holdout_projects"], 1)
        self.assertEqual(decision["decision"], "inconclusive_effect_or_concentration")
        self.assertIsNotNone(report["statistics"]["jev|manual|holdout"]["stress_peer_excess"]["mean"])
        loo = report["accounts"]["jev|manual|holdout"]["25"]["leave_one_issuer_out_pnl_after_operating_cost_usd"]
        self.assertEqual(loo, {"ORN": -.2})
        signals["reviewed_qualifying_projects"] = [{"source_id": "one", "project_id": "one", "symbol": "ORN"}]
        signals["reviewed_qualifying_projects_complete"] = True
        market["signals_sha256"] = _frozen(signals)
        reviewed = evaluate(signals, market, protocol)
        self.assertEqual(reviewed["decisions"]["jev"]["census_projects"], 1)
        self.assertEqual(reviewed["decisions"]["jev"]["decision"], "descriptive_insufficient_sample")
        self.assertEqual(reviewed["accounts"], report["accounts"])
        signals["reviewed_qualifying_projects"].append({"source_id": "development", "project_id": "development", "symbol": "TPC"})
        signals["reviewed_qualifying_projects_complete"] = False
        market["signals_sha256"] = _frozen(signals)
        incomplete = evaluate(signals, market, protocol)["decisions"]["jev"]
        self.assertEqual(incomplete["decision"], "inconclusive_effect_or_concentration")
        self.assertTrue(incomplete["census_is_lower_bound"])
        signals, market = fixture([event()])
        window = plan_windows(signals["events"], market["calendar"])[0]
        exit_quote = market["quotes"]["ORN|" + window["exit_at"]]["records"][0]
        exit_quote.update(bid=100.6, ask=100.8)
        decision = evaluate(signals, market, {"minimum_projects": 1, "minimum_issuers": 1})["decisions"]["jev"]
        self.assertGreater(decision["base_holdout_account_pnl_after_operating_cost_usd"], 0)
        self.assertLess(decision["stress_holdout_account_pnl_after_operating_cost_usd"], 0)
        self.assertEqual(decision["decision"], "inconclusive_stress_failure")
        exit_quote.update(bid=100.1, ask=100.3)
        decision = evaluate(signals, market, {"minimum_projects": 1, "minimum_issuers": 1})["decisions"]["jev"]
        self.assertEqual(decision["decision"], "reject_frozen_rule")


if __name__ == "__main__":
    unittest.main()
