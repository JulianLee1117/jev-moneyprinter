"""Offline, date-only association diagnostics; never executable trade returns.

SPY's observed daily dates supply the session index. Null prices retain their
session; absent stock dates are never filled or moved. Current adjusted history
and source-only model judgments cannot reproduce historical information sets.
"""

from __future__ import annotations

import bisect
import math
from datetime import date, datetime, timezone


class DailyReturnsValidationError(ValueError):
    """Input cannot support an unambiguous daily diagnostic."""


def _number(value, *, positive=False):
    return (type(value) in (int, float) and math.isfinite(value)
            and (value > 0 if positive else value >= 0))


def _date(value):
    if not isinstance(value, str):
        raise DailyReturnsValidationError("Session/publication date must be ISO YYYY-MM-DD.")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise DailyReturnsValidationError("Invalid ISO date.") from exc
    if parsed.isoformat() != value:
        raise DailyReturnsValidationError("Date must be canonical YYYY-MM-DD.")
    return value


def parse_yahoo_chart(payload: dict, symbol: str) -> list[dict]:
    """Validate one USD/New York daily chart; preserve every timestamp and null.

    Yahoo regular daily bar timestamps are in Unix seconds. UTC dates are used
    only when the timestamp is at least 05:00 UTC, so UTC and New York have the
    same civil date under either -04:00 or -05:00. Reject ambiguous midnight bars
    instead of guessing a session. No optional timezone package is required.
    """
    if not isinstance(symbol, str) or not symbol or symbol != symbol.upper():
        raise DailyReturnsValidationError("Expected a nonempty uppercase symbol.")
    if not isinstance(payload, dict) or not isinstance(payload.get("chart"), dict):
        raise DailyReturnsValidationError("Expected Yahoo chart object.")
    chart = payload["chart"]
    if chart.get("error") is not None:
        raise DailyReturnsValidationError("Chart reports a provider error.")
    result = chart.get("result")
    if not isinstance(result, list) or len(result) != 1 or not isinstance(result[0], dict):
        raise DailyReturnsValidationError("Chart must contain exactly one result.")
    result = result[0]
    meta = result.get("meta", {})
    if not isinstance(meta, dict) or meta.get("symbol") != symbol:
        raise DailyReturnsValidationError("Chart symbol does not match requested symbol.")
    if meta.get("currency") != "USD" or meta.get("exchangeTimezoneName") != "America/New_York":
        raise DailyReturnsValidationError("Only USD New York exchange dates are supported.")
    if meta.get("dataGranularity", "1d") != "1d":
        raise DailyReturnsValidationError("Only daily bars are supported.")
    timestamps = result.get("timestamp")
    try:
        indicators = result["indicators"]
        quote = indicators["quote"]
        adjusted = indicators["adjclose"]
        if len(quote) != 1 or len(adjusted) != 1:
            raise DailyReturnsValidationError("One close and adjusted-close series required.")
        closes, adjcloses = quote[0]["close"], adjusted[0]["adjclose"]
    except (KeyError, TypeError, IndexError) as exc:
        raise DailyReturnsValidationError("Close and adjusted-close arrays required.") from exc
    if (not isinstance(timestamps, list) or not timestamps
            or not isinstance(closes, list) or not isinstance(adjcloses, list)
            or len(timestamps) != len(closes) or len(timestamps) != len(adjcloses)):
        raise DailyReturnsValidationError("Daily price arrays must have matching nonzero lengths.")
    rows, previous = [], None
    for timestamp, close, adjusted_close in zip(timestamps, closes, adjcloses):
        if type(timestamp) is not int or timestamp <= 0:
            raise DailyReturnsValidationError("Unix timestamps must be positive integer seconds.")
        try:
            at = datetime.fromtimestamp(timestamp, timezone.utc)
        except (ValueError, OverflowError, OSError) as exc:
            raise DailyReturnsValidationError("Timestamp is outside supported date range.") from exc
        if at.hour < 5:
            raise DailyReturnsValidationError("Daily timestamp has ambiguous UTC/New York date.")
        day = at.date().isoformat()
        if previous is not None and day <= previous:
            raise DailyReturnsValidationError("Duplicate or unordered session dates.")
        previous = day
        for price in (close, adjusted_close):
            if price is not None and not _number(price, positive=True):
                raise DailyReturnsValidationError("Prices must be null or finite positive numbers.")
        rows.append({"date": day, "close": close, "adjusted_close": adjusted_close})
    return rows


def _series(rows, symbol):
    if not isinstance(rows, list) or not rows:
        raise DailyReturnsValidationError(f"Nonempty daily rows required for {symbol}.")
    result, previous = {}, None
    for row in rows:
        if not isinstance(row, dict):
            raise DailyReturnsValidationError("Price rows must be objects.")
        day = _date(row.get("date"))
        if previous is not None and day <= previous:
            raise DailyReturnsValidationError("Duplicate or unordered price session dates.")
        previous = day
        for field in ("close", "adjusted_close"):
            if field not in row:
                raise DailyReturnsValidationError("Each row requires close and adjusted_close.")
            if row[field] is not None and not _number(row[field], positive=True):
                raise DailyReturnsValidationError("Prices must be null or finite positive numbers.")
        result[day] = row
    return result


def _aggregate(values, total):
    known = [v for v in values if v is not None]
    complete = len(known) == total
    return {"total_count": total, "computable_count": len(known),
            "unknown_count": total - len(known),
            "coverage": len(known) / total if total else None,
            "status": "complete" if complete else "inconclusive_missing",
            "all_cohort_mean": math.fsum(known) / total if complete and total else None,
            "conditional_computable_mean": math.fsum(known) / len(known) if known else None}


def evaluate_daily(protocol: dict, signals: list[dict], prices: dict) -> dict:
    """Evaluate fixed long/cash document arms; report transitive overlap clusters.

    Active excess = equal-weight NUE/STLD adjusted close change minus SPY change.
    Flat illustrative round-trip basis points apply only to active observations.
    Cash is zero; failed/skipped/missing model outputs are UNKNOWN, never cash.
    Cluster means average member observation contributions. They are not a
    simulated portfolio, and overlapping members never become independent trades.
    """
    if not isinstance(protocol, dict) or not isinstance(signals, list) or not signals:
        raise DailyReturnsValidationError("Protocol object and nonempty signal list required.")
    horizon = protocol.get("horizon_sessions", 5)
    primary = protocol.get("primary_cost_bps", 25)
    costs = protocol.get("cost_scenarios_bps", [10, 25, 50])
    if type(horizon) is not int or horizon <= 0:
        raise DailyReturnsValidationError("Horizon must be positive integer sessions.")
    if (not _number(primary) or not isinstance(costs, list) or not costs
            or any(not _number(c) for c in costs) or len(set(costs)) != len(costs)
            or primary not in costs):
        raise DailyReturnsValidationError("Unique finite nonnegative costs including primary required.")
    if not isinstance(prices, dict) or not {"NUE", "STLD", "SPY"} <= set(prices):
        raise DailyReturnsValidationError("NUE, STLD and SPY price rows are required.")
    series = {symbol: _series(prices[symbol], symbol) for symbol in ("NUE", "STLD", "SPY")}
    sessions = list(series["SPY"])
    ids, arm_names = set(), set()
    for signal in signals:
        if not isinstance(signal, dict) or not isinstance(signal.get("document_id"), str) or not signal["document_id"]:
            raise DailyReturnsValidationError("Signals require document IDs.")
        if signal["document_id"] in ids:
            raise DailyReturnsValidationError("Duplicate document ID.")
        ids.add(signal["document_id"])
        _date(signal.get("publication_date"))
        arms = signal.get("arms")
        if not isinstance(arms, dict) or not arms:
            raise DailyReturnsValidationError("Signals require named arms.")
        for name, arm in arms.items():
            if not isinstance(name, str) or not name or not isinstance(arm, dict):
                raise DailyReturnsValidationError("Each arm must be a named object.")
            if arm.get("status") not in ("completed", "failed", "skipped"):
                raise DailyReturnsValidationError("Unknown model signal status.")
            if arm.get("status") == "completed" and arm.get("side") not in ("long", "cash"):
                raise DailyReturnsValidationError("Completed primary signals must be long or cash.")
            p = arm.get("positive_probability")
            if p is not None and (not _number(p) or p > 1):
                raise DailyReturnsValidationError("Positive probability must be finite in [0,1].")
        arm_names.update(arms)
    expected = protocol.get("document_ids")
    if expected is not None:
        if not isinstance(expected, list) or len(expected) != len(set(expected)) or set(expected) != ids:
            raise DailyReturnsValidationError("Signal cohort differs from frozen document IDs.")
    arm_names = sorted(arm_names)
    documents = []
    for signal in signals:
        entry_index = bisect.bisect_right(sessions, signal["publication_date"])
        exit_index = entry_index + horizon
        prior_calendar_known = entry_index > 0
        entry = sessions[entry_index] if prior_calendar_known and entry_index < len(sessions) else None
        exit_day = sessions[exit_index] if prior_calendar_known and exit_index < len(sessions) else None
        reasons, returns = [], {}
        if not prior_calendar_known:
            reasons.append("calendar_starts_after_publication")
        if entry is None:
            reasons.append("entry_session_unavailable")
        if exit_day is None:
            reasons.append("exit_session_unavailable")
        for symbol in ("NUE", "STLD", "SPY"):
            start = series[symbol].get(entry) if entry else None
            end = series[symbol].get(exit_day) if exit_day else None
            if (start is None or end is None or start["adjusted_close"] is None
                    or end["adjusted_close"] is None):
                returns[symbol] = None
                if entry is not None and exit_day is not None:
                    reasons.append(symbol + "_endpoint_missing")
            else:
                returns[symbol] = end["adjusted_close"] / start["adjusted_close"] - 1
                if not math.isfinite(returns[symbol]):
                    raise DailyReturnsValidationError("Adjusted price ratio is not finite.")
        complete_prices = all(value is not None for value in returns.values())
        basket = returns["NUE"] / 2 + returns["STLD"] / 2 if complete_prices else None
        excess = basket - returns["SPY"] if complete_prices else None
        if excess is not None and not math.isfinite(excess):
            raise DailyReturnsValidationError("Market-relative return is not finite.")
        arm_results = {}
        for name in arm_names:
            arm = signal["arms"].get(name)
            status = arm.get("status") if arm else "missing"
            side = arm.get("side") if arm and status == "completed" else None
            if status != "completed":
                eligibility, gross, active = "unknown_signal", None, None
            elif side == "cash":
                eligibility, gross, active = "cash", 0.0, False
            elif excess is None:
                eligibility, gross, active = "unknown_prices", None, True
            else:
                eligibility, gross, active = "active", excess, True
            net = {str(cost): (gross - cost / 10000 if active else gross)
                   if gross is not None else None for cost in costs}
            arm_results[name] = {"signal_status": status, "side": side, "status": eligibility,
                                 "active": active, "gross_excess_contribution": gross,
                                 "net_excess_by_cost_bps": net,
                                 "primary_net_excess_contribution": net[str(primary)]}
        documents.append({"document_id": signal["document_id"],
                          "publication_date": signal["publication_date"],
                          "entry_date": entry, "exit_date": exit_day,
                          "entry_session_index": entry_index if entry else None,
                          "exit_session_index": exit_index if exit_day else None,
                          "price_status": "complete" if complete_prices else "missing",
                          "price_issues": reasons, "symbol_adjusted_returns": returns,
                          "basket_adjusted_return": basket, "gross_market_relative_return": excess,
                          "arms": arm_results})

    # Inclusive interval intersection groups touching endpoints conservatively.
    known_windows = sorted((d for d in documents if d["entry_date"] and d["exit_date"]),
                           key=lambda d: (d["entry_date"], d["exit_date"], d["document_id"]))
    groups = []
    for doc in known_windows:
        if groups and doc["entry_date"] <= groups[-1]["exit_date"]:
            groups[-1]["members"].append(doc)
            groups[-1]["exit_date"] = max(groups[-1]["exit_date"], doc["exit_date"])
        else:
            groups.append({"entry_date": doc["entry_date"], "exit_date": doc["exit_date"], "members": [doc]})
    ungrouped = [d["document_id"] for d in documents if not d["entry_date"] or not d["exit_date"]]
    clusters = []
    for i, group in enumerate(groups, 1):
        members = group["members"]
        arms = {}
        for name in arm_names:
            scenarios = {}
            for cost in costs:
                values = [d["arms"][name]["net_excess_by_cost_bps"][str(cost)] for d in members]
                scenarios[str(cost)] = _aggregate(values, len(members))
            active = [d["arms"][name]["active"] for d in members]
            arms[name] = {"net_excess_by_cost_bps": scenarios,
                          "primary_net_excess_contribution": scenarios[str(primary)]["all_cohort_mean"],
                          "member_observation_active_fraction": (sum(v is True for v in active) / len(active)
                                                                 if all(v is not None for v in active) else None)}
        clusters.append({"cluster_id": f"overlap-{i:04d}", "entry_date": group["entry_date"],
                         "exit_date": group["exit_date"], "document_ids": [d["document_id"] for d in members],
                         "document_count": len(members), "arms": arms})
    summary = {}
    for name in arm_names:
        scenarios, cluster_scenarios = {}, {}
        for cost in costs:
            key = str(cost)
            scenarios[key] = _aggregate([d["arms"][name]["net_excess_by_cost_bps"][key]
                                         for d in documents], len(documents))
            cluster_scenarios[key] = _aggregate([c["arms"][name]["net_excess_by_cost_bps"][key]["all_cohort_mean"]
                                                 for c in clusters], len(clusters))
            if ungrouped:
                cluster_scenarios[key]["status"] = "inconclusive_ungrouped_windows"
                cluster_scenarios[key]["all_cohort_mean"] = None
        summary[name] = {"document_mean_by_cost_bps": scenarios,
                         "cluster_mean_by_cost_bps": cluster_scenarios,
                         "primary": scenarios[str(primary)],
                         "primary_cluster": cluster_scenarios[str(primary)],
                         "active_document_count": sum(d["arms"][name]["active"] is True for d in documents),
                         "cash_document_count": sum(d["arms"][name]["status"] == "cash" for d in documents),
                         "unknown_signal_count": sum(d["arms"][name]["status"] == "unknown_signal" for d in documents)}
    return {"schema_version": "daily-association-report-v1", "horizon_sessions": horizon,
            "primary_cost_bps": primary, "cost_scenarios_bps": costs,
            "document_count": len(documents), "priced_document_count": sum(d["price_status"] == "complete" for d in documents),
            "cluster_count": len(clusters), "ungrouped_document_ids": ungrouped,
            "documents": documents, "clusters": clusters, "arms": summary,
            "method": {"entry": "First observed SPY session strictly after publication date, close.",
                       "exit": "Close horizon_sessions after entry on observed SPY session index.",
                       "return": "Equal-weight NUE/STLD adjusted returns minus SPY adjusted return, for long; cash=0.",
                       "costs": "Flat illustrative round-trip bps per active document observation; not measured execution costs.",
                       "clustering": "Transitive inclusive interval overlap; equal mean of member observation contributions, then equal cluster mean.",
                       "primary_denominator": "Every frozen document, including cash=0; missing signals/prices make complete-cohort mean unavailable."},
            "limitations": ["Historical date-only association proxy, not an executable strategy backtest or evidence of alpha.",
                            "Observed SPY dates are the session index; independently verified exchange-calendar completeness is not asserted.",
                            "Current adjusted history may incorporate later corporate-action revisions; no raw-close substitution or forward fill.",
                            "Market subtraction is a descriptive control, not a hedge fill or estimated beta adjustment.",
                            "Cluster averages are descriptive, not capital-weighted portfolio returns; documents and stock legs are not independent trades.",
                            "No Sharpe ratio, statistical significance, borrow feasibility, or real execution is inferred."]}
