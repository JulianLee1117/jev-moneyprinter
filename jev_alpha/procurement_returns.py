"""Frozen procurement signals -> read-only market receipts -> offline diagnostics.

Market fixtures/captures use ``calendar`` (normalized Alpaca sessions), ``quotes``
keyed by ``SYMBOL|RFC3339 window start``, ``bars`` keyed by symbol, and ``actions``
keyed by ``SYMBOL|entry date|exit date``. Quote values contain records and status;
bar values contain records. Action assessments require status, evidence, and an
explicit coverage_complete assertion from independent review. Empty provider
action responses never become proof of no action. No function places orders.
"""
from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
import hashlib
import math
from pathlib import Path
import random
from statistics import mean

from .alpaca_prices import collect_alpaca_prices
from .earnings_prices import capture_market_inputs, calendar_sessions
from .store import Store, canonical_json, read_json, utc_now, write_new_json

UNIVERSE = ("GVA", "ROAD", "STRL", "TPC", "PRIM", "ORN")


def _settings(protocol):
    values = {"initial_cash_usd": 1000, "slippage_bps": 10, "stress_slippage_bps": 25,
              "fee_allowance_usd": 1, "min_price_usd": 5, "min_adv_usd": 1_000_000,
              "adv_sessions": 20, "minimum_projects": 30, "minimum_issuers": 4,
              "promising_mean_excess_return": .005, "bootstrap_seed": 20260918,
              "bootstrap_resamples": 2000}
    values.update({k: protocol[k] for k in values if k in protocol})
    if any(not _number(v) for v in values.values()) or values["initial_cash_usd"] <= 0:
        raise ValueError("Invalid finite nonnegative protocol parameter")
    if any(type(values[k]) is not int or values[k] < 1 for k in
           ("adv_sessions", "minimum_projects", "minimum_issuers", "bootstrap_resamples")):
        raise ValueError("Protocol counts must be positive integers")
    if max(values["slippage_bps"], values["stress_slippage_bps"]) >= 10000:
        raise ValueError("Invalid slippage")
    # Window definition is deliberately fixed, rather than silently ignoring changes.
    for key, value in (("benchmark", "IWM"), ("holding_sessions", 5), ("entry_time_et", "09:45"),
                       ("exit_time_et", "09:45"), ("intraday_execution_delay_seconds", 60),
                       ("intraday_last_entry_et", "15:45")):
        if protocol.get(key, value) != value:
            raise ValueError("Unsupported window policy: " + key)
    return values


def _symbols(protocol):
    values = protocol.get("symbols", protocol.get("issuers", UNIVERSE))
    if not isinstance(values, (list, tuple)) or len(set(values)) != len(values) or not values:
        raise ValueError("Unique explicit symbol universe required")
    return tuple(values)


def _number(x, positive=False):
    return type(x) in (float, int) and math.isfinite(x) and (x > 0 if positive else x >= 0)


def _stamp(value):
    if not isinstance(value, str):
        raise ValueError("Timestamp must be a timezone-aware string")
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.utcoffset() is None:
        raise ValueError("Timezone-aware timestamp required")
    return result


def _day(value):
    if isinstance(value, str) and len(value) == 8 and value.isdigit():
        value = value[:4] + "-" + value[4:6] + "-" + value[6:]
    return date.fromisoformat(value).isoformat()


def _ny_day(stamp):
    """Modern US DST date boundary without a platform tzdata dependency."""
    value = _stamp(stamp).astimezone(timezone.utc)
    year = value.year
    march, november = date(year, 3, 8), date(year, 11, 1)
    march += timedelta(days=(6 - march.weekday()) % 7)
    november += timedelta(days=(6 - november.weekday()) % 7)
    start = datetime.combine(march, time(7), timezone.utc)
    end = datetime.combine(november, time(6), timezone.utc)
    offset = timezone(timedelta(hours=-4 if start <= value < end else -5))
    return value.astimezone(offset).date().isoformat()


def _calendar(rows):
    if not rows:
        return []
    if rows[0].get("open") == "09:30":
        rows = calendar_sessions(rows, rows[0]["date"], rows[-1]["date"])
    days = [_day(r["date"]) for r in rows]
    if days != sorted(set(days)):
        raise ValueError("Exchange calendar must be sorted and unique")
    for row in rows:
        opened, closed = _stamp(row["open"]), _stamp(row["close"])
        if opened.date().isoformat() != row["date"] or closed <= opened:
            raise ValueError("Invalid exchange session bounds")
    return rows


def _at(session, hour=9, minute=45):
    return _stamp(session["open"]).replace(hour=hour, minute=minute, second=0, microsecond=0)


def _digest(value):
    return hashlib.sha256(canonical_json(value)).hexdigest()


def _verify_market_hash(market):
    if "sha256" in market and market["sha256"] != _digest({k: v for k, v in market.items() if k != "sha256"}):
        raise ValueError("Market receipt hash mismatch")


def _frozen(signals):
    stamp = _stamp(signals["frozen_at"])
    if stamp > datetime.now(timezone.utc):
        raise ValueError("Signal freeze cannot be in the future")
    body = {k: v for k, v in signals.items() if k != "sha256"}
    digest = _digest(body)
    if signals.get("sha256", digest) != digest:
        raise ValueError("Frozen signal hash mismatch")
    for event in signals.get("events", []):
        for signal in event.get("arms", {}).values():
            if signal.get("status") not in {"signal", "no_signal", "unknown"}:
                raise ValueError("Unknown signal status must not become cash")
    return digest


def plan_windows(events: list, calendar: list) -> list:
    """One primary and secondary row per event/arm; unavailable clocks stay explicit.

    Historical date-only inputs receive next-session association windows only.
    Intraday requires verified/prospective publication and arm readiness. Its
    exit is that arm's primary exit, so cross-horizon differences are not a
    causal model-speed estimate. All entries use actual supplied exchange days.
    """
    sessions = _calendar(calendar)
    output, seen = [], set()
    for event in events:
        if event["event_id"] in seen:
            raise ValueError("Duplicate event ID")
        seen.add(event["event_id"])
        availability = event.get("availability", "unverified_historical")
        for arm, signal in sorted(event.get("arms", {}).items()):
            base = {"event_id": event["event_id"], "project_id": event.get("project_id", event["event_id"]),
                    "symbol": event["symbol"], "arm": arm, "availability": availability,
                    "signal_status": signal.get("status", "unknown"),
                    "first_seen_at": event.get("first_seen_at"), "published_at": event.get("published_at"),
                    "materiality": signal.get("materiality", event.get("materiality")),
                    "split": event.get("split", "development" if _day(event["association_date"]) < "2025-10-01" else "holdout"),
                    "entry_at": None, "exit_at": None,
                    "hypothetical": True, "is_fill": False}
            try:
                if availability == "prospective_first_seen":
                    available = event.get("first_seen_at") or event.get("published_at")
                    if available is not None and _stamp(signal["ready_at"]) < _stamp(available):
                        raise ValueError("Ready before observed source availability")
                    day = _ny_day(signal["ready_at"])
                elif availability == "verified_historical":
                    # A replay's present-day wall clock is not historical readiness.
                    ready = _stamp(signal["counterfactual_ready_at"])
                    if ready < _stamp(event["published_at"]):
                        raise ValueError("Counterfactual readiness before source")
                    day = _ny_day(signal["counterfactual_ready_at"])
                else:
                    day = _day(event["association_date"])
                index = next(i for i, row in enumerate(sessions) if row["date"] > day)
                # Include settlement after exit; otherwise the ledger is incomplete.
                entry, exit_session = sessions[index], sessions[index + 5]
                sessions[index + 6]
                primary = {**base, "horizon": "manual", "status": "planned",
                           "entry_at": _at(entry).isoformat(), "exit_at": _at(exit_session).isoformat(),
                           "scope": ("historical_association" if availability == "unverified_historical" else
                                     "prospective_quote_illustration" if availability == "prospective_first_seen" else
                                     "historical_counterfactual")}
            except (ValueError, KeyError, StopIteration, IndexError, TypeError):
                primary = {**base, "horizon": "manual", "status": "unavailable",
                           "scope": "unknown", "reason": "missing_clock_or_calendar_coverage"}
            secondary = {**primary, "horizon": "intraday", "status": "unavailable",
                         "entry_at": None, "reason": "unverified_or_missing_intraday_clock"}
            if primary["status"] == "planned" and availability in {"verified_historical", "prospective_first_seen"}:
                try:
                    ready_key = "counterfactual_ready_at" if availability == "verified_historical" else "ready_at"
                    available_at = (event["published_at"] if availability == "verified_historical" else
                                    event.get("first_seen_at") or event.get("published_at"))
                    available, ready = _stamp(available_at), _stamp(signal[ready_key])
                    if ready < available:
                        raise ValueError("Ready before source")
                    earliest = ready + timedelta(seconds=60)
                    chosen = None
                    for session in sessions:
                        first = _at(session)
                        # A half-day session cannot accept an entry at 15:45.
                        last = min(_at(session, 15, 45), _stamp(session["close"]) - timedelta(minutes=1))
                        candidate = max(first, earliest.astimezone(first.tzinfo))
                        if candidate <= last:
                            chosen = candidate
                            break
                    if chosen is None or chosen >= _stamp(primary["exit_at"]):
                        raise ValueError("No mature intraday window")
                    secondary.update(status="planned", entry_at=chosen.isoformat(),
                                     scope=primary["scope"])
                    secondary.pop("reason", None)
                except (ValueError, TypeError, KeyError):
                    pass
            for row in (primary, secondary):
                row["window_id"] = event["event_id"] + "|" + arm + "|" + row["horizon"]
                output.append(row)
    return output


def capture_prices(root, signals_manifest, protocol, *, live=False):
    """Plan by default; explicit live=True calls only existing read-only collectors.

    An existing capture directory is never overwritten/retried. ``calendar`` in
    protocol can supply verified sessions; otherwise live capture first obtains
    a bounded historical calendar. No quote query can occur before signal freeze.
    Independent corporate-action assessments can be supplied in protocol.actions.
    """
    digest = _frozen(signals_manifest)
    _settings(protocol)
    if signals_manifest.get("protocol_sha256", _digest(protocol)) != _digest(protocol):
        raise ValueError("Protocol differs from signal freeze")
    events = signals_manifest.get("events", [])
    sessions = _calendar(protocol.get("calendar", []))
    manifest = {"schema_version": "procurement-market-v1", "signals_sha256": digest,
                "signal_frozen_at": signals_manifest["frozen_at"], "started_at": utc_now(),
                "status": "planned", "calendar": sessions, "windows": [], "quotes": {}, "bars": {},
                "raw_actions": {}, "actions": protocol.get("actions", {}),
                "interpretation": "Hypothetical quote diagnostics, not fills or proven historical availability."}
    if not events:
        manifest.update(status="no_events", finished_at=utc_now())
        if live:
            target = Path(root) / "market"
            target.mkdir(parents=True, exist_ok=False)
            manifest["root"] = str(target.resolve())
            write_new_json(target / "manifest.json", manifest)
        return manifest
    dates = sorted(_day(e["association_date"]) for e in events)
    start = (date.fromisoformat(dates[0]) - timedelta(days=60)).isoformat()
    end = min(date.fromisoformat(dates[-1]) + timedelta(days=45),
              datetime.now(timezone.utc).date() - timedelta(days=1)).isoformat()
    if not live:
        manifest["windows"] = plan_windows(events, sessions)
        manifest["calendar_query"] = {"kind": "calendar", "start": start, "end": end}
        return manifest
    target = Path(root) / "market"
    target.mkdir(parents=True, exist_ok=False)
    manifest["root"] = str(target.resolve())
    cap = protocol.get("max_market_captures", 1000)
    if type(cap) is not int or cap < 1:
        raise ValueError("Positive market capture cap required")
    captures = 0
    with Store(target / "archive") as store:
        if not sessions:
            receipt = capture_market_inputs(store, {"kind": "calendar", "start": start, "end": end}, target / "calendar")
            captures += 1
            if receipt["complete"]:
                payload = read_json(target / "calendar" / "payloads.json")
                sessions = calendar_sessions(payload[0], start, end)
            else:
                manifest["calendar_failure"] = receipt["status"]
        manifest["calendar"] = sessions
        manifest["windows"] = plan_windows(events, sessions)
        symbols = sorted(set(_symbols(protocol)) | {"IWM"} | {e["symbol"] for e in events})
        for symbol in symbols:
            for kind in ("daily_bars", "corporate_actions"):
                if captures >= cap:
                    break
                query = {"kind": kind, "symbol": symbol, "start": start, "end": end, "max_pages": 2}
                if kind == "daily_bars":
                    query.update(adjustment="raw", asof="-")
                else:
                    query.update(start="2016-01-01", end=(datetime.now(timezone.utc).date() - timedelta(days=1)).isoformat())
                folder = target / kind / symbol
                receipt = capture_market_inputs(store, query, folder)
                captures += 1
                value = {"path": str(folder.resolve()), "status": receipt["status"], "complete": receipt["complete"]}
                manifest["bars" if kind == "daily_bars" else "raw_actions"][symbol] = value
        by_day = {s["date"]: s for s in sessions}
        wanted = set()
        for window in manifest["windows"]:
            if window["status"] == "planned" and window["signal_status"] == "signal":
                for symbol in symbols:
                    wanted.update((symbol, window[key]) for key in ("entry_at", "exit_at"))
        for symbol, stamp in sorted(wanted):
            if captures >= cap:
                break
            moment = _stamp(stamp)
            if moment + timedelta(minutes=16) > datetime.now(timezone.utc):
                continue
            query = {"kind": "quotes", "symbol": symbol, "start": stamp,
                     "end": (moment + timedelta(minutes=1)).isoformat(), "asof": moment.date().isoformat(),
                     "feed": "sip", "currency": "USD", "limit": 1000, "max_pages": 2,
                     "calendar": by_day[moment.date().isoformat()]}
            folder = target / "quotes" / _digest(query)[:24]
            receipt = collect_alpaca_prices(store, query, folder)
            captures += 1
            manifest["quotes"][symbol + "|" + stamp] = {"path": str(folder.resolve()),
                "status": receipt["status"], "complete": receipt["complete"]}
    manifest.update(status="captured" if captures < cap else "capture_cap_reached",
                    capture_count=captures, finished_at=utc_now())
    write_new_json(target / "manifest.json", manifest)
    return manifest


def _quote(market, symbol, stamp, *, shares=None, capital=None, bps=0, fee=0, diagnostic=None):
    diagnostic = diagnostic if diagnostic is not None else {}
    diagnostic.update(status="unknown", reason="quote_unavailable_or_insufficient_size")
    value = market.get("quotes", {}).get(symbol + "|" + stamp, {})
    if value.get("status") not in {"captured", "truncated"}:
        return None
    records = value.get("records")
    if records is None and value.get("path"):
        records = read_json(Path(value["path"]) / "records.json")
    first, last = _stamp(stamp), _stamp(stamp) + timedelta(minutes=1)
    previous = None
    for index, record in enumerate(records or []):
        try:
            instant = _stamp(record["timestamp"])
        except (ValueError, KeyError, TypeError):
            return None
        if previous is not None and instant < previous:
            return None
        previous = instant
        if not first <= instant < last:
            continue
        if (record.get("conditions") != ["R"] or
                not all(_number(record.get(k), True) for k in ("bid", "ask", "bid_size", "ask_size")) or
                record["bid"] >= record["ask"]):
            continue
        assessment = record.get("size_unit_assessment", value.get("size_unit_assessment", {}))
        verified = (isinstance(assessment, dict) and assessment.get("verified") is True
                    and bool(assessment.get("evidence")))
        if verified:
            if assessment.get("unit") == "shares":
                factors = (1,)
            elif (assessment.get("unit") == "round_lots"
                  and type(assessment.get("round_lot_shares")) is int
                  and assessment["round_lot_shares"] > 0):
                factors = (assessment["round_lot_shares"],)
            else:
                diagnostic.update(reason="quote_size_unit_metadata_invalid", quote_index=index)
                return None
            unit_status = "verified_size_unit_metadata"
        elif instant.date().isoformat() >= "2025-11-03":
            factors, unit_status = (1,), "shares_per_alpaca_2025_11_03_change"
        else:
            # The switch notice does not document current replay units for old
            # records. Compare both interpretations without inventing share size.
            factors, unit_status = (1, 100), "unverified_historical_units_first_quote_invariant"
        needed = (max(0, math.floor((capital - fee) / (record["ask"] * (1 + bps / 10000))))
                  if capital is not None else None)
        sufficient = [((needed is None or record["ask_size"] * factor >= needed)
                       and (shares is None or record["bid_size"] * factor >= shares))
                      for factor in factors]
        if any(sufficient) and not all(sufficient):
            # This is already the first sufficient quote in one interpretation.
            # A later quote cannot restore identical first-quote selection.
            diagnostic.update(reason="quote_size_unit_ambiguous_first_sufficient", quote_index=index,
                              quote_at=record["timestamp"], bid=record["bid"], ask=record["ask"],
                              raw_bid_size=record["bid_size"], raw_ask_size=record["ask_size"],
                              required_ask_shares=needed, required_bid_shares=shares,
                              size_multipliers=list(factors), sufficient_by_interpretation=sufficient,
                              later_quote_substitution_permitted=False)
            return None
        if not all(sufficient):
            continue
        factor = factors[0] if len(factors) == 1 else None
        candidate = {**record, "bid_shares": record["bid_size"] * factor if factor is not None else None,
                     "ask_shares": record["ask_size"] * factor if factor is not None else None,
                     "size_multiplier": factor, "size_unit_status": unit_status,
                     "size_multipliers_checked": list(factors), "first_sufficient_quote_invariant": True}
        diagnostic.update(status="usable", reason=None, quote_index=index, quote_at=record["timestamp"],
                          size_unit_status=unit_status, size_multipliers_checked=list(factors))
        return candidate
    return None


def _adv(market, symbol, entry, sessions, count):
    days = [r["date"] for r in sessions if r["date"] < entry][-count:]
    if len(days) != count:
        return None
    source = market.get("bars", {}).get(symbol, {})
    if source.get("status") != "captured":
        return None
    records = source.get("records")
    if records is None and source.get("path"):
        payloads = read_json(Path(source["path"]) / "payloads.json")
        records = [r for p in payloads for r in p.get("bars", {}).get(symbol, [])]
    by_day = {}
    for row in records or []:
        try:
            day = _day(row.get("date", row.get("t", "")[:10]))
        except (ValueError, TypeError):
            return None
        close, volume = row.get("close", row.get("c")), row.get("volume", row.get("v"))
        if day in by_day or not _number(close, True) or not _number(volume):
            return None
        by_day[day] = (close, close * volume)
    if not all(day in by_day for day in days):
        return None
    return {"adv_usd": mean(by_day[day][1] for day in days), "prior_close": by_day[days[-1]][0]}


def _action(market, symbol, entry, exit_day):
    assessment = market.get("actions", {}).get("|".join((symbol, entry, exit_day)), {})
    if not assessment.get("coverage_complete") or not assessment.get("evidence"):
        return None
    if assessment.get("status") == "none":
        return 0.0 if assessment.get("cash_dividend_per_entry_share", 0) == 0 else None
    dividend = assessment.get("cash_dividend_per_entry_share")
    if assessment.get("status") == "cash_dividend" and _number(dividend):
        return dividend
    return None


def _trade(market, symbol, window, capital, bps, fee, *, check_liquidity=True, settings=None):
    settings = settings or _settings({})
    entry_day, exit_day = window["entry_at"][:10], window["exit_at"][:10]
    result = {"status": "unknown", "return": None, "pnl_usd": None, "reason": None}
    if check_liquidity:
        adv = _adv(market, symbol, entry_day, market["calendar"], settings["adv_sessions"])
        if adv is None:
            return {**result, "reason": "pre_entry_volume_unavailable"}
        if adv["prior_close"] < settings["min_price_usd"] or adv["adv_usd"] < settings["min_adv_usd"]:
            return {**result, "status": "ineligible", "reason": "pre_entry_price_or_liquidity"}
    entry_diagnostic = {}
    entry = _quote(market, symbol, window["entry_at"], capital=capital, bps=bps, fee=fee,
                   diagnostic=entry_diagnostic)
    if entry is None:
        return {**result, "reason": "entry_" + entry_diagnostic["reason"],
                "quote_diagnostic": entry_diagnostic}
    effective = entry["ask"] * (1 + bps / 10000)
    shares = max(0, math.floor((capital - fee) / effective))
    if not shares:
        return {**result, "status": "ineligible", "reason": "whole_share_unaffordable"}
    dividend = _action(market, symbol, entry_day, exit_day)
    if dividend is None:
        return {**result, "reason": "corporate_actions_unresolved"}
    exit_diagnostic = {}
    exit_quote = _quote(market, symbol, window["exit_at"], shares=shares, diagnostic=exit_diagnostic)
    if exit_quote is None:
        return {**result, "reason": "exit_" + exit_diagnostic["reason"],
                "quote_diagnostic": exit_diagnostic}
    proceeds = shares * exit_quote["bid"] * (1 - bps / 10000) - fee
    spent = shares * effective
    receivable = shares * dividend
    pnl = proceeds - spent + receivable
    return {"status": "computed", "shares": shares, "pnl_usd": pnl, "return": pnl / capital,
            "entry_spend": spent, "exit_proceeds": proceeds, "dividend_receivable_usd": receivable,
            "entry_ask": entry["ask"], "exit_bid": exit_quote["bid"], "fee_usd": fee,
            "entry_quote_at": entry["timestamp"], "exit_quote_at": exit_quote["timestamp"],
            "entry_quote_size_unit_status": entry["size_unit_status"],
            "exit_quote_size_unit_status": exit_quote["size_unit_status"],
            "reason": None}


def _ledger(rows, market, cost, bps, settings):
    initial = settings["initial_cash_usd"]
    cash, dividends, settlement, pending = float(initial), 0.0, None, None
    trades, skipped = [], []
    days = [r["date"] for r in market["calendar"]]
    result = {"status": "complete", "trades": trades, "skipped": skipped,
              "operating_cost_usd": cost, "pnl_usd": None, "pnl_after_operating_cost_usd": None,
              "final_equity_usd": None, "hypothetical": True, "is_fill": False}
    groups = {}
    for row in rows:
        if row["status"] != "planned":
            if row["signal_status"] != "no_signal":
                return {**result, "status": "unknown", "reason": "selected_clock_unavailable"}
            continue
        groups.setdefault(row["entry_at"], []).append(row)
    for stamp, group in sorted(groups.items(), key=lambda item: _stamp(item[0])):
        if settlement is not None and stamp[:10] >= settlement:
            cash += pending["exit_proceeds"]
            dividends += pending["dividend_receivable_usd"]
            settlement, pending = None, None
        if pending is not None:
            skipped.extend({"event_id": r["event_id"], "reason": "capital_unavailable"} for r in group)
            continue
        if any(r["signal_status"] == "unknown" for r in group):
            return {**result, "status": "unknown", "reason": "unknown_signal_cannot_rank"}
        candidates = [r for r in group if r["signal_status"] == "signal"]
        if any(not _number(r.get("materiality"), True) for r in candidates):
            return {**result, "status": "unknown", "reason": "missing_priority"}
        ranked = sorted(candidates, key=lambda r: (-r["materiality"], r["event_id"]))
        for rank, row in enumerate(ranked):
            trade = _trade(market, row["symbol"], row, cash, bps, settings["fee_allowance_usd"], settings=settings)
            if trade["status"] == "ineligible":
                skipped.append({"event_id": row["event_id"], "reason": trade["reason"]})
                continue
            if trade["status"] == "unknown":
                return {**result, "status": "unknown", "reason": trade["reason"], "unknown_event_id": row["event_id"],
                        "quote_diagnostic": trade.get("quote_diagnostic")}
            index = days.index(row["exit_at"][:10])
            if index + 1 >= len(days):
                return {**result, "status": "unknown", "reason": "settlement_calendar_unavailable"}
            settlement = days[index + 1]
            pending = trade
            cash -= trade["entry_spend"]
            trades.append({**trade, "event_id": row["event_id"], "symbol": row["symbol"],
                           "entry_at": stamp, "exit_at": row["exit_at"], "settlement_date": settlement})
            skipped.extend({"event_id": r["event_id"], "reason": "simultaneous_lower_priority"}
                           for r in ranked[rank + 1:])
            break
    if pending:
        cash += pending["exit_proceeds"]
        dividends += pending["dividend_receivable_usd"]
    equity = cash + dividends
    result.update(final_equity_usd=equity, pnl_usd=equity - initial,
                  pnl_after_operating_cost_usd=equity - initial - cost,
                  final_equity_after_operating_cost_usd=equity - cost,
                  dividend_receivable_usd=dividends)
    return result


def _statistics(rows, key, settings):
    valid = [r for r in rows if r.get(key) is not None]
    result = {"n": len(valid), "missing": len(rows) - len(valid), "mean": None,
              "week_block_bootstrap_95": None, "leave_one_issuer_out": {}, "leave_one_event_out_min": None}
    if not valid:
        return result
    result["mean"] = mean(r[key] for r in valid)
    symbols = sorted({r["symbol"] for r in valid})
    result["leave_one_issuer_out"] = {symbol: mean(r[key] for r in valid if r["symbol"] != symbol)
        for symbol in symbols if any(r["symbol"] != symbol for r in valid)}
    if len(valid) > 1:
        result["leave_one_event_out_min"] = min((sum(r[key] for r in valid) - row[key]) / (len(valid) - 1) for row in valid)
    blocks = {}
    for row in valid:
        iso = date.fromisoformat(row["entry_at"][:10]).isocalendar()
        blocks.setdefault((iso.year, iso.week), []).append(row[key])
    if len(blocks) > 1:
        groups, rng, samples = list(blocks.values()), random.Random(settings["bootstrap_seed"]), []
        for _ in range(settings["bootstrap_resamples"]):
            sample = [x for __ in groups for x in rng.choice(groups)]
            samples.append(mean(sample))
        samples.sort()
        result["week_block_bootstrap_95"] = [samples[int(.025 * (len(samples) - 1))], samples[int(.975 * (len(samples) - 1))]]
    result["independent_weeks"] = len(blocks)
    return result


def _split_value(signals, field, arm, split, legacy, *, default=None):
    if split == "all":
        return signals.get(legacy, {}).get(arm, default)
    values = signals.get(field, {})
    # Accept either documented split->arm or arm->split layout.
    if arm in values.get(split, {}):
        return values[split][arm]
    if split in values.get(arm, {}):
        return values[arm][split]
    return default


def _account(rows, market, cost, unknown_count, bps, settings):
    account = _ledger(rows, market, cost or 0, bps, settings)
    if cost is None:
        account.update(operating_cost_usd=None, pnl_after_operating_cost_usd=None,
                       final_equity_after_operating_cost_usd=None, cost_status="unknown")
    if unknown_count:
        account.update(status="unknown", reason="unmapped_unknown_packets", pnl_usd=None,
                       pnl_after_operating_cost_usd=None, final_equity_usd=None,
                       final_equity_after_operating_cost_usd=None)
    return account


def _benchmark_reference(market, symbol, window):
    """Gross total-return reference, not a simulated benchmark trade/account."""
    result = {"status": "unknown", "return": None, "reason": None}
    entry = _quote(market, symbol, window["entry_at"])
    exit_quote = _quote(market, symbol, window["exit_at"])
    if entry is None or exit_quote is None:
        return {**result, "reason": "benchmark_quote_unavailable"}
    dividend = _action(market, symbol, window["entry_at"][:10], window["exit_at"][:10])
    if dividend is None:
        return {**result, "reason": "benchmark_corporate_actions_unresolved"}
    entry_midpoint = (entry["bid"] + entry["ask"]) / 2
    exit_midpoint = (exit_quote["bid"] + exit_quote["ask"]) / 2
    return {"status": "computed", "return": (exit_midpoint + dividend) / entry_midpoint - 1,
            "entry_midpoint": entry_midpoint, "exit_midpoint": exit_midpoint,
            "cash_dividend_per_share": dividend, "entry_quote_at": entry["timestamp"],
            "exit_quote_at": exit_quote["timestamp"], "gross_reference": True, "reason": None}


def _comparison(window, market, universe, settings, bps):
    stock = _trade(market, window["symbol"], window, settings["initial_cash_usd"],
                   bps, settings["fee_allowance_usd"], settings=settings)
    result = {"outcome": stock, "iwm_excess": None, "peer_excess": None,
              "missing_benchmarks": [], "benchmark_references": {}}
    if stock["status"] != "computed":
        return result
    peers = []
    for symbol in ("IWM", *(s for s in universe if s != window["symbol"])):
        benchmark = _benchmark_reference(market, symbol, window)
        result["benchmark_references"][symbol] = benchmark
        if benchmark["status"] != "computed":
            result["missing_benchmarks"].append(symbol)
        elif symbol == "IWM":
            result["iwm_excess"] = stock["return"] - benchmark["return"]
        else:
            peers.append(benchmark["return"])
    if len(peers) == len(universe) - 1 and peers:
        result["peer_excess"] = stock["return"] - mean(peers)
    return result


def evaluate(signals_manifest, market_manifest, protocol):
    """Offline comparison. Missing selected observations invalidate deployment gates."""
    digest = _frozen(signals_manifest)
    _verify_market_hash(market_manifest)
    settings = _settings(protocol)
    if signals_manifest.get("protocol_sha256", _digest(protocol)) != _digest(protocol):
        raise ValueError("Protocol differs from signal freeze")
    if market_manifest.get("signals_sha256") != digest:
        raise ValueError("Market receipt is not bound to these frozen signals")
    market = {**market_manifest, "calendar": _calendar(market_manifest.get("calendar", []))}
    events = signals_manifest.get("events", [])
    windows = plan_windows(events, market["calendar"])
    universe = _symbols(protocol)
    results = []
    for window in windows:
        result = {**window, "outcome": None, "iwm_excess": None, "peer_excess": None,
                  "missing_benchmarks": [], "stress_outcome": None, "stress_iwm_excess": None,
                  "stress_peer_excess": None, "stress_missing_benchmarks": []}
        if window["status"] == "planned" and window["signal_status"] == "signal":
            result.update(_comparison(window, market, universe, settings, settings["slippage_bps"]))
            result.update({"stress_" + key: value for key, value in
                           _comparison(window, market, universe, settings, settings["stress_slippage_bps"]).items()})
        results.append(result)
    accounts, complete_evidence_accounts, statistics, decisions = {}, {}, {}, {}
    arms = sorted(set(protocol.get("arms", ["rules", "nano", "mini", "jev"])) | {w["arm"] for w in windows})
    reviewed_census = signals_manifest.get("reviewed_qualifying_projects")
    if reviewed_census is None:
        census = [e for e in events if any(s.get("status") == "signal" for s in e.get("arms", {}).values())]
        census_projects = {(e.get("source_id"), e.get("project_id", e["event_id"])) for e in census}
        census_issuers = {e["symbol"] for e in census}
        census_basis, census_complete = "model_signal_union_legacy_not_independent", True
    else:
        if not isinstance(reviewed_census, list) or any(
                not isinstance(row, dict) or row.get("symbol") not in universe
                or not all(isinstance(row.get(k), str) and row[k].strip() for k in ("source_id", "project_id"))
                for row in reviewed_census):
            raise ValueError("Reviewed qualifying census requires source/project/issuer identities")
        census_projects = {(r["source_id"], r["project_id"]) for r in reviewed_census}
        census_issuers = {r["symbol"] for r in reviewed_census}
        census_basis = "price_blind_independently_reviewed_qualifying_projects"
        census_complete = signals_manifest.get("reviewed_qualifying_projects_complete") is True
    for arm in arms:
        for horizon in ("manual", "intraday"):
            for split in ("all", "development", "holdout"):
                chosen = [w for w in windows if w["arm"] == arm and w["horizon"] == horizon
                          and (split == "all" or w["split"] == split)]
                key = "|".join((arm, horizon, split))
                raw_cost = _split_value(signals_manifest, "arm_costs_by_split_usd", arm, split, "arm_costs_usd")
                cost = raw_cost if _number(raw_cost) else None
                # Legacy unknown counts have no reliable split and contaminate both.
                unknown_count = _split_value(signals_manifest, "unmapped_unknown_by_split", arm, split,
                                             "unmapped_unknown_packets", default=signals_manifest.get(
                                                 "unmapped_unknown_packets", {}).get(arm, 0))
                complete_evidence_accounts[key] = {
                    str(bps): _account(chosen, market, cost, unknown_count, bps, settings)
                    for bps in (settings["slippage_bps"], settings["stress_slippage_bps"])}
                failures = _split_value(signals_manifest, "model_execution_failures_by_split", arm,
                                        split, "model_execution_failures", default=0 if arm == "rules" else None)
                known_decisions = [w for w in chosen if w["signal_status"] in {"signal", "no_signal"}]
                abstentions = [{"event_id": w["event_id"], "window_id": w["window_id"],
                                "entry_at": w["entry_at"], "status": "unknown"}
                               for w in chosen if w["signal_status"] == "unknown"]
                accounts[key] = {
                    str(bps): {**_account(known_decisions, market, cost, 0, bps, settings),
                               "semantic_unknown_abstentions": abstentions,
                               "semantic_unknown_abstention_count": len(abstentions),
                               "unmapped_unknown_count": unknown_count,
                               "model_execution_failures": failures,
                               "model_execution_coverage": "complete" if failures == 0 else (
                                   "unknown" if failures is None else "incomplete"),
                               "research_gate_eligible_by_itself": False}
                    for bps in (settings["slippage_bps"], settings["stress_slippage_bps"])}
                sample = [r for r in results if r["arm"] == arm and r["horizon"] == horizon
                          and r["signal_status"] == "signal" and (split == "all" or r["split"] == split)]
                statistics[key] = {kind: _statistics(sample, kind, settings) for kind in
                                   ("iwm_excess", "peer_excess", "stress_iwm_excess", "stress_peer_excess")}
                if horizon == "manual" and split == "holdout":
                    for bps, account in accounts[key].items():
                        account["leave_one_issuer_out_pnl_after_operating_cost_usd"] = {
                            symbol: _account([w for w in known_decisions if w["symbol"] != symbol], market,
                                             cost, 0, float(bps), settings)["pnl_after_operating_cost_usd"]
                            for symbol in sorted({r["symbol"] for r in sample})}
        all_signals = [r for r in results if r["arm"] == arm and r["horizon"] == "manual" and r["signal_status"] == "signal"]
        manual = [r for r in all_signals if r["split"] == "holdout"]
        holdout_stats = statistics[arm + "|manual|holdout"]
        stats = holdout_stats["iwm_excess"]
        stress_stats = holdout_stats["stress_iwm_excess"]
        base_account = accounts[arm + "|manual|holdout"][str(settings["slippage_bps"])]
        stress = accounts[arm + "|manual|holdout"][str(settings["stress_slippage_bps"])]
        projects, issuers = {r["project_id"] for r in manual}, {r["symbol"] for r in manual}
        complete = (stats["missing"] == 0 and manual and
                    all(r["outcome"] and r["outcome"]["status"] == "computed" for r in manual))
        stress_complete = (stress_stats["missing"] == 0 and manual and
                           all(r["stress_outcome"] and r["stress_outcome"]["status"] == "computed" for r in manual))
        verdict = "descriptive_insufficient_sample"
        execution_failures = _split_value(signals_manifest, "model_execution_failures_by_split", arm,
                                          "all", "model_execution_failures", default=0 if arm == "rules" else None)
        holdout_execution_failures = base_account["model_execution_failures"]
        if len(census_projects) >= settings["minimum_projects"] and len(census_issuers) >= settings["minimum_issuers"]:
            if execution_failures != 0 or holdout_execution_failures != 0:
                verdict = "inconclusive_model_execution_coverage"
            elif not manual:
                verdict = "inconclusive_no_holdout_signals"
            elif not complete or base_account["pnl_after_operating_cost_usd"] is None:
                verdict = "inconclusive_missing_data"
            elif stats["mean"] <= 0 or base_account["pnl_after_operating_cost_usd"] <= 0:
                verdict = "reject_frozen_rule"
            elif not stress_complete or stress["pnl_after_operating_cost_usd"] is None:
                verdict = "inconclusive_missing_stress_data"
            elif stress_stats["mean"] <= 0 or stress["pnl_after_operating_cost_usd"] <= 0:
                verdict = "inconclusive_stress_failure"
            else:
                loo_excess = list(stress_stats["leave_one_issuer_out"].values())
                loo_accounts = list(stress["leave_one_issuer_out_pnl_after_operating_cost_usd"].values())
                robust = (loo_excess and min(loo_excess) > 0 and loo_accounts and
                          all(x is not None and x > 0 for x in loo_accounts))
                if stats["mean"] > settings["promising_mean_excess_return"] and robust:
                    verdict = ("promising_association_only" if any(r["scope"] == "historical_association" for r in manual)
                               else "promising_requires_prospective_confirmation")
                else:
                    verdict = "inconclusive_effect_or_concentration"
        decisions[arm] = {"decision": verdict, "census_projects": len(census_projects),
                          "census_issuers": len(census_issuers), "projects": len(projects), "issuers": len(issuers),
                          "census_basis": census_basis, "reviewed_census_complete": census_complete if reviewed_census is not None else None,
                          "census_is_lower_bound": not census_complete,
                          "arm_full_period_projects": len({r["project_id"] for r in all_signals}),
                          "holdout_projects": len(projects), "holdout_issuers": len(issuers),
                          "holdout_independent_weeks": stats.get("independent_weeks", 0),
                          "primary_benchmark": "IWM",
                          "base_holdout_mean_iwm_excess": stats["mean"],
                          "stress_holdout_mean_iwm_excess": stress_stats["mean"],
                          "base_holdout_mean_peer_excess": holdout_stats["peer_excess"]["mean"],
                          "stress_holdout_mean_peer_excess": holdout_stats["stress_peer_excess"]["mean"],
                          "base_holdout_account_pnl_after_operating_cost_usd": base_account["pnl_after_operating_cost_usd"],
                          "stress_holdout_account_pnl_after_operating_cost_usd": stress["pnl_after_operating_cost_usd"],
                          "model_execution_failures": execution_failures,
                          "holdout_model_execution_failures": holdout_execution_failures,
                          "holdout_uncertainty": "A sparse holdout cannot establish profitable performance even if the full census meets its size target.",
                          "proven_alpha": False, "live_trading_authorized": False}
    return {"schema_version": "procurement-return-report-v1", "signals_sha256": digest,
            "hypothetical": True, "is_fill": False, "windows": results, "accounts": accounts,
            "observed_decision_accounts": accounts,
            "complete_evidence_accounts": complete_evidence_accounts,
            "accounts_policy": (
                "Primary hypothetical account acts only on frozen positive decisions. Semantic unknowns are "
                "explicit abstentions and retain unknown economic outcomes; they are not zero-return investments. "
                "Acceptance requires zero full-period and holdout model execution failures. Missing selected "
                "prices, clocks or corporate actions still make account returns unknown. Full split costs apply."),
            "observed_decision_accounts_interpretation": (
                "Compatibility alias of primary accounts. Complete-evidence diagnostic accounts additionally "
                "halt on any semantic unknown; that diagnostic is not the frozen abstention policy or acceptance gate."),
            "statistics": statistics, "decisions": decisions,
            "source_coverage": {"sources":signals_manifest.get("source_results", []),
                "preparation_failures":signals_manifest.get("source_failures", []),
                "independent_quality_observations":signals_manifest.get("source_quality", {})},
            "combined_research_cost_usd": signals_manifest.get("research_cost_usd"),
            "speed_claim": "not_established_by_horizon_comparison_or_historical_replay",
            "benchmark_method": (
                "Net stock trade return on initial capital minus gross matched-window total-return reference. "
                "IWM and each other contractor use the first eligible regular two-sided quote midpoint at each "
                "endpoint plus independently verified cash dividends. Peer reference is the equal-weight mean "
                "of all five other contractors. References have no trading slippage, fees, whole-share sizing "
                "or simulated benchmark portfolio; the same gross reference applies in base and stress cases. "
                "Missing quotes/actions remain unknown."),
            "cost_policy": "Each arm pays the actual full-corpus operating cost for the evaluated date split; all-period and split scenarios are not summed.",
            "uncertainty": "Week bootstrap and leave-one-out are descriptive, not independent-trade proof or a deployment gate."}
