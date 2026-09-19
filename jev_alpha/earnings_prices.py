"""Bounded read-only inputs for earnings studies; quote capture stays unchanged.

``capture_market_inputs`` accepts one of three query shapes:
  daily_bars: kind, symbol, start, end (ISO dates), asof, adjustment (raw/all);
              optional max_pages.
  calendar: kind, start, end (ISO dates).
  corporate_actions: kind, symbol, start, end (ISO dates); optional max_pages.

Daily bars explicitly request USD SIP and raw/all adjustment. Corporate-action dates filter
*process dates*, not ex-dates: use a sufficiently broad history and inspect all
effective dates. A successful empty response is NOT proof of no action.
All raw safe payloads and receipts are immutable. No account/order endpoints,
feed substitutions, retries, price inference, or corporate-action arithmetic.

Contracts: https://docs.alpaca.markets/us/reference/stockbars
https://docs.alpaca.markets/us/reference/legacycalendar
https://docs.alpaca.markets/us/reference/corporateactions-1
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
import hashlib
from pathlib import Path
import re
import shutil
import subprocess
from urllib.parse import parse_qs, urlencode, urlsplit

from .alpaca_prices import (AlpacaTransportError, CurlAlpacaTransport,
    MAX_RESPONSE_BYTES, STATUS_MARKER, _SYMBOL, _safe_json, _timestamp)
from .credentials import read_key
from .store import canonical_json, utc_now, write_new_json
from .transport import _config_quote

CALENDAR_URL = "https://paper-api.alpaca.markets/v2/calendar"
ACTION_URL = "https://data.alpaca.markets/v1/corporate-actions"
BAR_URL = "https://data.alpaca.markets/v2/stocks/bars"


def _day(value):
    if not isinstance(value, str) or date.fromisoformat(value).isoformat() != value:
        raise ValueError("An ISO date is required")
    return date.fromisoformat(value)


def _query(query, now):
    if not isinstance(query, dict):
        raise ValueError("A query object is required")
    kind = query.get("kind")
    allowed = {"kind", "start", "end"}
    if kind in {"daily_bars", "corporate_actions"}:
        allowed |= {"symbol", "max_pages"}
    if kind == "daily_bars":
        allowed |= {"asof", "adjustment"}
    if kind not in {"daily_bars", "calendar", "corporate_actions"} or set(query) - allowed:
        raise ValueError("Unsupported earnings market-data query")
    start, end = _day(query.get("start")), _day(query.get("end"))
    if not isinstance(now, datetime) or now.utcoffset() is None:
        raise ValueError("now must be timezone-aware")
    today = now.astimezone(timezone.utc).date()
    # Daily bar intervals must end before today's UTC date. Calendar is a
    # historical session census here, not a future scheduler.
    if start.year < 2007 or not start <= end < today:
        raise ValueError("Historical dates from 2007 through yesterday required")
    if (end - start).days > (366 if kind != "corporate_actions" else 7305):
        raise ValueError("Query exceeds bounded history")
    pages = query.get("max_pages", 1 if kind == "calendar" else 2)
    if type(pages) is not int or not 1 <= pages <= 10:
        raise ValueError("max_pages must be 1..10")
    result = {**query, "max_pages": pages}
    if kind != "calendar" and (not isinstance(query.get("symbol"), str) or not _SYMBOL.fullmatch(query["symbol"])):
        raise ValueError("One explicit historical symbol is required")
    if kind == "daily_bars" and query.get("asof") != "-" and _day(query.get("asof")) > today:
        raise ValueError("Historical asof date or '-' required")
    if kind == "daily_bars" and query.get("adjustment") not in {"raw", "all"}:
        raise ValueError("Explicit raw or all bar adjustment required")
    return result


class CurlReferenceTransport:
    """Only two read-only reference endpoints; credentials use curl stdin."""

    def get(self, url, key_id, secret_key, timeout=30):
        parsed, params = urlsplit(url), parse_qs(urlsplit(url).query)
        base = parsed.scheme + "://" + parsed.netloc + parsed.path
        allowed = ({"start", "end", "date_type"} if base == CALENDAR_URL else
                   {"start", "end", "symbols", "region", "data_quality", "sort", "limit", "page_token"})
        if (base not in {CALENDAR_URL, ACTION_URL} or parsed.fragment
                or set(params) - allowed or not {"start", "end"}.issubset(params)
                or any(len(v) != 1 for v in params.values())):
            raise AlpacaTransportError("Unsupported read-only reference endpoint")
        try:
            kind = "calendar" if base == CALENDAR_URL else "corporate_actions"
            _query({"kind": kind, "start": params["start"][0], "end": params["end"][0],
                    **({"symbol": params.get("symbols", [None])[0]} if kind != "calendar" else {})},
                   datetime.now(timezone.utc))
            if kind == "calendar" and params.get("date_type") != ["TRADING"]:
                raise ValueError()
            if kind == "corporate_actions" and any(params.get(k) != [v] for k, v in
                    (("region", "us"), ("data_quality", "all"), ("sort", "asc"), ("limit", "1000"))):
                raise ValueError()
        except (ValueError, TypeError):
            raise AlpacaTransportError("Invalid bounded reference query") from None
        if any(not isinstance(k, str) or not k or any(ord(c) < 33 or ord(c) > 126 for c in k)
               for k in (key_id, secret_key)) or type(timeout) not in (int, float) or not 0.1 <= timeout <= 120:
            raise AlpacaTransportError("Invalid credentials or timeout")
        executable = shutil.which("curl.exe") or shutil.which("curl")
        if not executable:
            raise AlpacaTransportError("curl is unavailable")
        config = "\n".join([
            "url = " + _config_quote(url), 'request = "GET"', 'proto = "=https"',
            'proto-redir = "=https"', "no-location", "max-redirs = 0", "retry = 0", "silent",
            "max-time = " + str(timeout), "connect-timeout = " + str(min(timeout, 10)),
            "max-filesize = " + str(MAX_RESPONSE_BYTES), 'header = "Accept: application/json"',
            "header = " + _config_quote("APCA-API-KEY-ID: " + key_id),
            "header = " + _config_quote("APCA-API-SECRET-KEY: " + secret_key),
            "write-out = " + _config_quote(STATUS_MARKER.decode() + "%{http_code}"), ""]).encode()
        try:
            answer = subprocess.run([executable, "-q", "--config", "-"], input=config,
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=timeout + 5,
                check=False, shell=False, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except (subprocess.TimeoutExpired, OSError):
            raise AlpacaTransportError("Reference transfer failed; no retry attempted") from None
        raw = answer.stdout
        if answer.returncode or not isinstance(raw, bytes) or len(raw) > MAX_RESPONSE_BYTES + len(STATUS_MARKER) + 3:
            raise AlpacaTransportError("Reference transfer failed or exceeded size limit")
        parts = raw.rsplit(STATUS_MARKER, 1)
        if len(parts) != 2 or not re.fullmatch(rb"[1-5][0-9]{2}", parts[1]):
            raise AlpacaTransportError("Reference response lacks HTTP status")
        return int(parts[1]), parts[0]


def capture_market_inputs(store, query, out, *, env_path=".env", transport=None, now=None):
    """Archive bounded raw input pages; ``complete`` means transport coverage only.

    Read ``payloads.json`` alongside receipt.json. Quote/economic eligibility
    and effective corporate-action dates require separate checks. A denied,
    failed, malformed, or truncated capture is unknown, never an empty dataset.
    """
    request = _query(query, now or datetime.now(timezone.utc))
    target = Path(out)
    target.mkdir(parents=True, exist_ok=False)
    payloads = []
    receipt = {"schema_version": "earnings-market-inputs-v1", "query": request,
        "query_sha256": hashlib.sha256(canonical_json(request)).hexdigest(),
        "started_at": utc_now(), "pages": [], "complete": False,
        "interpretation": "Current historical observations; not fills or point-in-time availability.",
        "corporate_action_coverage": "process-date interval only; no absence or economic adjustment inferred"}

    def finish(status):
        receipt.update(status=status, finished_at=utc_now(), complete=status == "captured",
                       payloads_sha256=hashlib.sha256(canonical_json(payloads)).hexdigest())
        write_new_json(target / "payloads.json", payloads)
        write_new_json(target / "receipt.json", receipt)
        return receipt

    try:
        keys = tuple(read_key(k, env_path) for k in ("APCA_API_KEY_ID", "APCA_API_SECRET_KEY"))
    except (OSError, UnicodeError, ValueError):
        return finish("invalid_credentials")
    if not all(keys):
        return finish("missing_credentials")
    if any(any(ord(c) < 33 or ord(c) > 126 for c in k) for k in keys):
        return finish("invalid_credentials")
    kind = request["kind"]
    client = transport or (CurlAlpacaTransport() if kind == "daily_bars" else CurlReferenceTransport())
    params = {k: request[k] for k in ("start", "end")}
    if kind == "daily_bars":
        # Alpaca interval endpoints are inclusive. Include the entire end date.
        params.update(start=request["start"] + "T00:00:00Z", end=request["end"] + "T23:59:59Z",
            symbols=request["symbol"], asof=request["asof"], feed="sip", currency="USD",
            timeframe="1Day", adjustment=request["adjustment"], sort="asc", limit=10000)
    elif kind == "corporate_actions":
        params.update(symbols=request["symbol"], region="us", data_quality="all", sort="asc", limit=1000)
    else:
        params["date_type"] = "TRADING"
    base = {"daily_bars": BAR_URL, "calendar": CALENDAR_URL, "corporate_actions": ACTION_URL}[kind]
    seen = set()
    for page in range(1, request["max_pages"] + 1):
        url = base + "?" + urlencode(params)
        row = {"page": page, "source_url": url, "started_at": utc_now()}
        receipt["pages"].append(row)
        try:
            status, body = client.get(url, *keys)
        except (AlpacaTransportError, OSError, subprocess.TimeoutExpired):
            return finish("transport_failed")
        row.update(http_status=status, finished_at=utc_now())
        if type(status) is not int or not 100 <= status <= 599 or not isinstance(body, bytes) or len(body) > MAX_RESPONSE_BYTES:
            return finish("invalid_response")
        try:
            # Wrapping also permits the calendar's top-level list, while using
            # the existing recursive credential-echo/nonfinite/depth guards.
            payload = _safe_json(b'{"response":' + body + b'}', keys)["response"]
        except (ValueError, UnicodeError, RecursionError):
            row["body_withheld"] = True
            return finish("provider_denied" if status in {401, 403, 422} else "invalid_response")
        row.update(body_sha256=hashlib.sha256(body).hexdigest(),
            observation_id=store.observe(url, body, kind="earnings_" + kind, status=status))
        if status != 200:
            return finish("provider_denied" if status in {401, 403, 422} else "http_error")
        if kind == "calendar":
            if not isinstance(payload, list):
                return finish("invalid_response")
            try:
                calendar_sessions(payload, request["start"], request["end"])
            except (ValueError, TypeError, KeyError):
                return finish("invalid_response")
            token = None
        else:
            group = "bars" if kind == "daily_bars" else "corporate_actions"
            if (not isinstance(payload, dict) or any(k in payload for k in ("error", "code", "message"))
                    or not isinstance(payload.get(group), dict)):
                return finish("invalid_response")
            if kind == "daily_bars" and (set(payload[group]) - {request["symbol"]}
                    or any(not isinstance(v, list) for v in payload[group].values())
                    or any(k in payload and payload[k] != v for k, v in
                           (("feed", "sip"), ("currency", "USD"), ("timeframe", "1Day"),
                            ("adjustment", request["adjustment"])))):
                return finish("invalid_response")
            token = payload.get("next_page_token")
            if token is not None and (not isinstance(token, str) or not 1 <= len(token) <= 4096
                    or any(ord(c) < 33 or ord(c) > 126 for c in token) or token in seen):
                return finish("invalid_response")
        payloads.append(payload)
        if token is None:
            return finish("captured")
        seen.add(token)
        params["page_token"] = token
    return finish("truncated")


def _dst_bounds(year):
    march, november = date(year, 3, 8), date(year, 11, 1)
    return (march + timedelta(days=(6 - march.weekday()) % 7),
            november + timedelta(days=(6 - november.weekday()) % 7))


def calendar_sessions(rows, start, end):
    """Normalize Alpaca session rows to existing quote-capture calendar input."""
    first, last, previous = _day(start), _day(end), None
    sessions = []
    for row in rows:
        day = _day(row["date"])
        if day.year < 2007 or day.weekday() >= 5 or not first <= day <= last or previous is not None and day <= previous:
            raise ValueError("Invalid or unordered calendar session")
        previous = day
        if row["open"] != "09:30" or row["close"] not in {"13:00", "16:00"}:
            raise ValueError("Unsupported regular session bounds")
        dst_start, dst_end = _dst_bounds(day.year)
        offset = "-04:00" if dst_start <= day < dst_end else "-05:00"
        sessions.append({"name": "XNYS", "timezone": "America/New_York", "date": day.isoformat(),
            "open": day.isoformat() + "T09:30:00" + offset,
            "close": day.isoformat() + "T" + row["close"] + ":00" + offset,
            "source_url": "https://docs.alpaca.markets/us/reference/legacycalendar"})
    return sessions


def event_sessions(accepted_at, sessions, *, holding_sessions=5):
    """Second full session strictly after the SEC acceptance NY date, then +5.

    Supplied calendar must include a session on/before acceptance and the exit.
    No inference from observed prices or weekday-only holiday guesses.
    """
    if type(holding_sessions) is not int or not 1 <= holding_sessions <= 20:
        raise ValueError("holding_sessions must be 1..20")
    _, accepted = _timestamp(accepted_at)
    instant = datetime.fromisoformat(accepted.replace("Z", "+00:00"))
    if instant.year < 2007:
        raise ValueError("Unsupported historical timezone rules")
    dst_start, dst_end = _dst_bounds(instant.year)
    start = datetime.combine(dst_start, datetime.min.time(), timezone.utc).replace(hour=7)
    end = datetime.combine(dst_end, datetime.min.time(), timezone.utc).replace(hour=6)
    local_day = (instant - timedelta(hours=4 if start <= instant < end else 5)).date().isoformat()
    dates = [s["date"] for s in sessions]
    if not dates or dates != sorted(set(dates)) or dates[0] > local_day:
        raise ValueError("Incomplete or unordered event calendar")
    following = [s for s in sessions if s["date"] > local_day]
    if len(following) <= 1 + holding_sessions:
        raise ValueError("Calendar does not reach exit")
    entry, exit_session = following[1], following[1 + holding_sessions]
    return {"acceptance_new_york_date": local_day, "entry_session": entry, "exit_session": exit_session,
            "entry_at": entry["open"].replace("T09:30:00", "T09:35:00"),
            "exit_at": exit_session["open"].replace("T09:30:00", "T09:35:00")}
