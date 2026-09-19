"""Small offline earnings pilot arithmetic; quotes are hypothetical, never fills.

The caller supplies verified exchange sessions, historical symbol identity and
corporate actions. Missing outcomes are unknown, not zero-return observations.
"""

from __future__ import annotations

import math
from datetime import date
from itertools import groupby

from .jev import choice_probability_mass_delta


CATEGORIES = {"favourable", "unchanged", "adverse", "mixed", "unknown"}
WEIGHTS = {name: (1 if name == "favourable" else -1 if name == "adverse" else 0)
           for name in CATEGORIES}


def _number(value, *, positive=False):
    return (type(value) in (int, float) and math.isfinite(value)
            and (value > 0 if positive else value >= 0))


def semantic_score(answers: dict, feature_ids, probabilistic: bool = True) -> float:
    """Fixed eight-feature mean; unknown/mixed retain their denominator weight.

Native answers have choice and a complete five-category probabilities mapping.
Categorical answers may instead be category strings. Invalid output raises.
"""
    if (not isinstance(feature_ids, (list, tuple)) or len(feature_ids) != 8
            or any(not isinstance(x, str) or not x for x in feature_ids)
            or len(set(feature_ids)) != 8 or not isinstance(answers, dict)
            or set(answers) != set(feature_ids) or type(probabilistic) is not bool):
        raise ValueError("Exactly eight unique feature IDs and every answer are required")
    values = []
    for feature_id in feature_ids:
        answer = answers[feature_id]
        choice = answer if isinstance(answer, str) else answer.get("choice") if isinstance(answer, dict) else None
        if choice not in CATEGORIES:
            raise ValueError("Invalid or missing category")
        if not probabilistic:
            values.append(WEIGHTS[choice])
            continue
        probabilities = answer.get("probabilities") if isinstance(answer, dict) else None
        if not isinstance(probabilities, dict) or set(probabilities) != CATEGORIES:
            raise ValueError("Complete probability categories required")
        # Match the provider adapter's bounded rounding tolerance. Keep the raw
        # masses: this is a semantic score, not a normalized return probability.
        choice_probability_mass_delta(probabilities)
        values.append(probabilities["favourable"] - probabilities["adverse"])
    return math.fsum(values) / 8


def _costs(capital, bps, fee):
    if not _number(capital, positive=True) or not _number(bps) or bps >= 10000 or not _number(fee):
        raise ValueError("Finite positive capital, nonnegative fee and 0..9999 bps required")


def _trade(capital, entry, exit_price, bps, fee):
    if not _number(entry, positive=True) or not _number(exit_price, positive=True):
        raise ValueError("Finite positive quote prices required")
    entry_effective, exit_effective = entry * (1 + bps / 10000), exit_price * (1 - bps / 10000)
    # Reserve the round-trip allowance so modeled costs never require borrowing.
    shares = max(0, math.floor((capital - fee) / entry_effective))
    applied_fee = fee if shares else 0.0
    spent, proceeds = shares * entry_effective, shares * exit_effective - applied_fee
    final = capital - spent + proceeds
    return {"shares": shares, "entry_effective": entry_effective, "exit_effective": exit_effective,
            "entry_spend": spent, "exit_proceeds_after_fee": proceeds,
            "fee_allowance_applied": applied_fee, "final_equity_usd": final,
            "pnl_usd": final - capital, "account_return": final / capital - 1}


def quote_outcome(stock_entry_ask, stock_exit_bid, benchmark_entry_ask, benchmark_exit_bid,
                  *, capital_usd=1000, slippage_bps=10, fee_allowance_usd=1) -> dict:
    """Two separate whole-share accounts, stock minus IWM; price-only diagnostic.

Caller must verify quote eligibility and corporate actions before interpreting
this as an event-return observation. No quote is asserted to be executable.
"""
    _costs(capital_usd, slippage_bps, fee_allowance_usd)
    result = {"status": "missing", "hypothetical": True, "is_fill": False,
              "scope": "quote_price_only_before_corporate_action_validation",
              "requires_corporate_action_validation": True,
              "stock": None, "benchmark": None, "excess_account_return": None}
    if any(p is None for p in (stock_entry_ask, stock_exit_bid, benchmark_entry_ask, benchmark_exit_bid)):
        return result
    stock = _trade(capital_usd, stock_entry_ask, stock_exit_bid, slippage_bps, fee_allowance_usd)
    benchmark = _trade(capital_usd, benchmark_entry_ask, benchmark_exit_bid, slippage_bps, fee_allowance_usd)
    result.update(status="computed_price_only", stock=stock, benchmark=benchmark,
                  excess_account_return=stock["account_return"] - benchmark["account_return"])
    return result


def daily_proxy(entry_adjusted_close, exit_adjusted_close,
                benchmark_entry_adjusted_close, benchmark_exit_adjusted_close) -> dict:
    """Separate close-to-close association proxy, not the intraday trade outcome.

Caller supplies the endpoints of the frozen five-session window. This function
does not infer dates, validate provider adjustments or add execution costs.
"""
    prices = (entry_adjusted_close, exit_adjusted_close,
              benchmark_entry_adjusted_close, benchmark_exit_adjusted_close)
    result = {"status": "missing", "scope": "adjusted_daily_close_proxy_not_execution",
              "stock_return": None, "benchmark_return": None, "excess_return": None}
    if any(p is None for p in prices):
        return result
    if any(not _number(p, positive=True) for p in prices):
        raise ValueError("Adjusted closes must be finite positive prices or missing")
    stock = exit_adjusted_close / entry_adjusted_close - 1
    benchmark = benchmark_exit_adjusted_close / benchmark_entry_adjusted_close - 1
    result.update(status="computed_proxy", stock_return=stock, benchmark_return=benchmark,
                  excess_return=stock - benchmark)
    return result


def _day(value):
    if not isinstance(value, str) or date.fromisoformat(value).isoformat() != value:
        raise ValueError("Canonical ISO dates required")
    return value


def simulate_ledger(events: list[dict], arm: str, sessions: list[str], *,
                    initial_cash_usd=1000, slippage_bps=10, fee_allowance_usd=1,
                    force_long=False) -> dict:
    """One position, fixed priority, settled cash only; unknown selected trade halts.

Each event requires event_id, entry_date, exit_date, entry_price (ask), exit_price
    (bid), scores[arm], and corporate_action_status. Status 'none' requires no
    cash dividend; 'cash_dividend' requires cash_dividend_per_entry_share >= 0.
    All other action statuses halt a selected trade, including split/merger.
    Entry/exit must be supplied regular sessions and exactly five sessions apart.
    The calendar must extend one session after exit. Dividend receivables count
    in ending equity but are never reinvested without payment-date information.
    force_long selects even nonpositive scores using the identical ranking rule.
    Optional entry_available_shares / exit_available_shares are checked against
    the actual ledger quantity. Absent fields retain the price-only diagnostic;
    present but missing/invalid/insufficient sizes halt without an alternative.
    """
    _costs(initial_cash_usd, slippage_bps, fee_allowance_usd)
    if (not isinstance(events, list) or not isinstance(arm, str) or not arm
            or not isinstance(sessions, list) or not sessions or type(force_long) is not bool):
        raise ValueError("Events, arm and an explicit exchange-session calendar required")
    days = [_day(day) for day in sessions]
    if days != sorted(set(days)):
        raise ValueError("Sessions must be unique and strictly increasing")
    indices, seen = {day: i for i, day in enumerate(days)}, set()
    for event in events:
        if not isinstance(event, dict) or not isinstance(event.get("event_id"), str) or not event["event_id"]:
            raise ValueError("Each event needs an ID")
        if event["event_id"] in seen:
            raise ValueError("Duplicate event ID")
        seen.add(event["event_id"])
        entry, exit_day = _day(event.get("entry_date")), _day(event.get("exit_date"))
        if entry not in indices or exit_day not in indices or indices[exit_day] - indices[entry] != 5:
            raise ValueError("Entry and exit must be exactly five exchange sessions apart")
        if indices[exit_day] + 1 >= len(days):
            raise ValueError("Calendar must include the next settlement session")
    ordered = sorted(events, key=lambda e: (e["entry_date"], e["event_id"]))
    cash, dividends, position = float(initial_cash_usd), 0.0, None
    trades, decisions = [], []
    result = {"status": "complete", "arm": arm, "force_long": force_long,
              "hypothetical": True, "is_fill": False, "initial_cash_usd": initial_cash_usd,
              "cash_control_final_equity_usd": initial_cash_usd, "trades": trades,
              "decisions": decisions, "final_equity_usd": None, "pnl_usd": None,
              "account_return": None, "dividend_policy": "receivable_not_reinvested",
              "fees": "fixed_round_trip_allowance_not_verified_actual_broker_fees"}

    def halt(event_id, reason):
        result.update(status="unknown", unknown_event_id=event_id, unknown_reason=reason,
                      known_settled_cash_usd=cash, dividend_receivable_usd=dividends)
        return result

    for day, group in groupby(ordered, key=lambda e: e["entry_date"]):
        group = list(group)
        if position is not None and day >= position["settlement_date"]:
            cash += position["exit_proceeds_after_fee"]
            dividends += position["dividend_receivable_usd"]
            position = None
        if position is not None:
            decisions.extend({"event_id": e["event_id"], "status": "capital_unavailable_until_settlement"}
                             for e in group)
            continue
        for event in group:
            score = event.get("scores", {}).get(arm) if isinstance(event.get("scores"), dict) else None
            if type(score) not in (int, float) or not math.isfinite(score) or not -1 <= score <= 1:
                return halt(event["event_id"], "missing_or_invalid_signal_cannot_rank")
        ranked = sorted(group, key=lambda e: (-e["scores"][arm], e["event_id"]))
        chosen = ranked[0]
        if not force_long and chosen["scores"][arm] <= 0:
            decisions.extend({"event_id": e["event_id"], "status": "cash_nonpositive_signal"} for e in group)
            continue
        decisions.extend({"event_id": e["event_id"], "status": "lower_same_entry_priority"} for e in ranked[1:])
        entry = chosen.get("entry_price")
        if not _number(entry, positive=True):
            return halt(chosen["event_id"], "selected_entry_price_unavailable")
        effective_entry = entry * (1 + slippage_bps / 10000)
        shares = max(0, math.floor((cash - fee_allowance_usd) / effective_entry))
        if not shares:
            decisions.append({"event_id": chosen["event_id"], "status": "insufficient_cash_for_one_share"})
            continue
        if "entry_available_shares" in chosen:
            available = chosen["entry_available_shares"]
            if not _number(available, positive=True) or available < shares:
                return halt(chosen["event_id"], "selected_entry_size_unavailable_or_insufficient")
        action = chosen.get("corporate_action_status")
        dividend = chosen.get("cash_dividend_per_entry_share", 0 if action == "none" else None)
        if action not in {"none", "cash_dividend"} or not _number(dividend) or action == "none" and dividend != 0:
            return halt(chosen["event_id"], "selected_corporate_actions_unresolved_or_unsupported")
        if not _number(chosen.get("exit_price"), positive=True):
            return halt(chosen["event_id"], "selected_exit_price_unavailable")
        if "exit_available_shares" in chosen:
            available = chosen["exit_available_shares"]
            if not _number(available, positive=True) or available < shares:
                return halt(chosen["event_id"], "selected_exit_size_unavailable_or_insufficient")
        trade = _trade(cash, entry, chosen["exit_price"], slippage_bps, fee_allowance_usd)
        trade.update(event_id=chosen["event_id"], entry_date=day, exit_date=chosen["exit_date"],
                     settlement_date=days[indices[chosen["exit_date"]] + 1], score=chosen["scores"][arm],
                     dividend_receivable_usd=shares * dividend)
        cash -= trade["entry_spend"]
        position = trade
        trades.append(trade)
        decisions.append({"event_id": chosen["event_id"], "status": "hypothetical_long"})
    if position is not None:
        cash += position["exit_proceeds_after_fee"]
        dividends += position["dividend_receivable_usd"]
    equity = cash + dividends
    result.update(final_settled_cash_usd=cash, dividend_receivable_usd=dividends,
                  final_equity_usd=equity, pnl_usd=equity - initial_cash_usd,
                  account_return=equity / initial_cash_usd - 1)
    return result
