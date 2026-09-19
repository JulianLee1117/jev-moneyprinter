"""Bounded, read-only historical SIP captures; never submit orders or infer fills.

Public API: ``collect_alpaca_prices(store, query, out, env_path='.env')``.
``out`` must not exist. Query example (one caller-verified regular session):

    {"symbol": "AAPL", "kind": "quotes", "feed": "sip", "currency": "USD",
     "start": "2026-09-17T14:00:00Z", "end": "2026-09-17T14:01:00Z",
     "asof": "2026-09-17", "limit": 1000, "max_pages": 2,
     "calendar": {"name": "XNYS", "timezone": "America/New_York",
       "date": "2026-09-17", "open": "2026-09-17T09:30:00-04:00",
       "close": "2026-09-17T16:00:00-04:00",
       "source_url": "https://www.nyse.com/markets/hours-calendars"}}

Calendar bounds are validated but holiday/source authenticity is the caller's
responsibility. Only one session and <=1 hour per request are supported. Bars
are unadjusted 1Min bars; no split/dividend adjustment is silently applied.
asof is mandatory: a historical date or '-' to disable Alpaca symbol mapping.
Receipts and normalized records are immutable files; raw safe JSON responses
are archived in Store, including explicit failed/empty/truncated statuses.

Provider contracts: https://docs.alpaca.markets/us/reference/stockquotes-1
https://docs.alpaca.markets/us/reference/stockbars
https://docs.alpaca.markets/us/docs/market-data-faq
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
from urllib.parse import parse_qs, urlencode, urlsplit

from .credentials import read_key
from .store import Store, canonical_json, utc_now, write_new_json
from .transport import _config_quote

BASE_URL = "https://data.alpaca.markets/v2/stocks/"
MAX_RESPONSE_BYTES = 5_000_000
MAX_PAGES = 10
STATUS_MARKER = b"\n__ALPACA_HTTP_STATUS__:"
_TIME = re.compile(r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})(?:\.(\d{1,9}))?(Z|[+-]\d{2}:\d{2})\Z")
_SYMBOL = re.compile(r"[A-Z][A-Z0-9.\-]{0,14}\Z")
_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


class AlpacaTransportError(RuntimeError):
    """Redacted failure. No automatic retries are performed."""


def _timestamp(value: object) -> tuple[int, str]:
    """Retain nanosecond precision instead of rounding quotes to microseconds."""
    match = _TIME.fullmatch(value) if isinstance(value, str) else None
    if not match:
        raise ValueError("Expected a timezone-aware RFC3339 timestamp")
    base, fraction, offset = match.groups()
    if offset != "Z" and (int(offset[1:3]) > 23 or int(offset[4:6]) > 59):
        raise ValueError("Invalid RFC3339 offset")
    try:
        instant = datetime.fromisoformat(base + ("+00:00" if offset == "Z" else offset)).astimezone(timezone.utc)
    except ValueError:
        raise ValueError("Invalid RFC3339 timestamp") from None
    delta = instant - _EPOCH
    nanos = int((fraction or "").ljust(9, "0") or "0")
    return ((delta.days * 86400 + delta.seconds) * 1_000_000_000 + nanos,
            instant.strftime("%Y-%m-%dT%H:%M:%S") + ("." + f"{nanos:09d}" if nanos else "") + "Z")


def _now_ns(now: datetime | None) -> int:
    instant = now or datetime.now(timezone.utc)
    if not isinstance(instant, datetime) or instant.utcoffset() is None:
        raise ValueError("now must be timezone-aware")
    return _timestamp(instant.isoformat())[0]


def _positive(value: object, *, zero: bool = False) -> bool:
    try:
        return type(value) in (int, float) and math.isfinite(value) and (value >= 0 if zero else value > 0)
    except OverflowError:
        return False


def _query(query: dict, now: datetime | None) -> dict:
    if not isinstance(query, dict):
        raise ValueError("A historical query object is required")
    allowed = {"symbol", "kind", "start", "end", "asof", "feed", "currency", "calendar", "limit", "max_pages", "timeframe", "adjustment"}
    if set(query) - allowed:
        raise ValueError("Unsupported historical query field")
    symbol, kind = query.get("symbol"), query.get("kind", "quotes")
    if not isinstance(symbol, str) or not _SYMBOL.fullmatch(symbol) or not isinstance(kind, str) or kind not in {"quotes", "trades", "bars"}:
        raise ValueError("Expected one explicit historical symbol and quotes, trades, or bars")
    if query.get("feed", "sip") != "sip" or query.get("currency") != "USD":
        raise ValueError("Only explicit USD historical SIP prices are supported")
    start, start_text = _timestamp(query.get("start"))
    end, end_text = _timestamp(query.get("end"))
    if not start < end or end - start > 3600 * 1_000_000_000:
        raise ValueError("Historical window must be positive and at most one hour")
    now_ns = _now_ns(now)
    if end > now_ns - 900 * 1_000_000_000:
        raise ValueError("Historical SIP end must be at least 15 minutes old")
    asof = query.get("asof")
    if asof != "-":
        try:
            if not isinstance(asof, str) or date.fromisoformat(asof).isoformat() != asof:
                raise ValueError()
            if _timestamp(asof + "T00:00:00Z")[0] > now_ns:
                raise ValueError()
        except (TypeError, ValueError):
            raise ValueError("Explicit historical asof date or '-' is required") from None
    limit, max_pages = query.get("limit", 1000), query.get("max_pages", 2)
    if type(limit) is not int or not 1 <= limit <= 10000 or type(max_pages) is not int or not 1 <= max_pages <= MAX_PAGES:
        raise ValueError("limit must be 1..10000 and max_pages 1..10")
    calendar = query.get("calendar")
    if not isinstance(calendar, dict) or set(calendar) != {"name", "timezone", "date", "open", "close", "source_url"}:
        raise ValueError("An explicit calendar session with source provenance is required")
    if calendar["name"] not in ("XNYS", "XNAS") or calendar["timezone"] != "America/New_York":
        raise ValueError("Unsupported US equity calendar")
    try:
        session_date = date.fromisoformat(calendar["date"])
        session_open = datetime.fromisoformat(calendar["open"])
        session_close = datetime.fromisoformat(calendar["close"])
        open_ns, _ = _timestamp(calendar["open"])
        close_ns, _ = _timestamp(calendar["close"])
        source = urlsplit(calendar["source_url"])
        # US DST rules have been stable since 2007. Earlier calendar rules need
        # a separately supported calendar, rather than guessing an offset.
        march8, november1 = date(session_date.year, 3, 8), date(session_date.year, 11, 1)
        dst_start = march8 + timedelta(days=(6 - march8.weekday()) % 7)
        dst_end = november1 + timedelta(days=(6 - november1.weekday()) % 7)
        expected_offset = timedelta(hours=-4 if dst_start <= session_date < dst_end else -5)
        valid = (session_date.weekday() < 5
                 and session_date.year >= 2007
                 and session_open.date() == session_close.date() == session_date
                 and session_open.utcoffset() == expected_offset
                 and session_close.utcoffset() == session_open.utcoffset()
                 and open_ns % 1_000_000_000 == close_ns % 1_000_000_000 == 0
                 and (session_open.hour, session_open.minute, session_open.second, session_open.microsecond) == (9, 30, 0, 0)
                 and (session_close.hour, session_close.minute, session_close.second, session_close.microsecond) in {(13, 0, 0, 0), (16, 0, 0, 0)}
                 and open_ns <= start < end <= close_ns
                 and source.scheme == "https" and source.hostname and not source.username
                 and not source.password and not source.query and not source.fragment)
    except (TypeError, ValueError, AttributeError):
        valid = False
    if not valid:
        raise ValueError("Invalid calendar provenance or window outside supplied regular session")
    if kind == "bars":
        if query.get("timeframe", "1Min") != "1Min" or query.get("adjustment", "raw") != "raw":
            raise ValueError("Only unadjusted 1Min historical bars are supported")
    elif "timeframe" in query or "adjustment" in query:
        raise ValueError("Bar options are invalid for quotes or trades")
    return {"symbol": symbol, "kind": kind, "start": start_text, "end": end_text,
            "asof": asof, "feed": "sip", "currency": "USD", "limit": limit,
            "max_pages": max_pages, "calendar": dict(calendar),
            **({"timeframe": "1Min", "adjustment": "raw"} if kind == "bars" else {})}


class CurlAlpacaTransport:
    """GET allowlisted data endpoints; credentials travel only through curl stdin."""

    def get(self, url: str, key_id: str, secret_key: str, timeout: float = 30) -> tuple[int, bytes]:
        parsed = urlsplit(url)
        params = parse_qs(parsed.query)
        if (parsed.scheme != "https" or parsed.netloc != "data.alpaca.markets"
                or parsed.path not in {"/v2/stocks/quotes", "/v2/stocks/trades", "/v2/stocks/bars"}
                or parsed.fragment or params.get("feed") != ["sip"]
                or not {"start", "end", "asof", "currency", "symbols"}.issubset(params)):
            raise AlpacaTransportError("Unsupported historical market-data endpoint")
        try:
            if (set(params) - {"symbols", "start", "end", "asof", "feed", "currency", "limit", "sort", "page_token", "timeframe", "adjustment"}
                    or any(len(v) != 1 for v in params.values())
                    or params["currency"] != ["USD"]
                    or not _SYMBOL.fullmatch(params["symbols"][0])
                    or not _timestamp(params["start"][0])[0] < _timestamp(params["end"][0])[0] <= _now_ns(None) - 900 * 1_000_000_000):
                raise ValueError()
        except (ValueError, TypeError, IndexError):
            raise AlpacaTransportError("Invalid bounded historical SIP query") from None
        if any(not isinstance(key, str) or not key or any(ord(c) < 33 or ord(c) > 126 for c in key) for key in (key_id, secret_key)):
            raise AlpacaTransportError("Invalid market-data credential characters")
        if not _positive(timeout) or not 0.1 <= timeout <= 120:
            raise AlpacaTransportError("Invalid market-data timeout")
        executable = shutil.which("curl.exe") or shutil.which("curl")
        if not executable:
            raise AlpacaTransportError("curl is unavailable")
        config = "\n".join([
            "url = " + _config_quote(url), 'request = "GET"', 'proto = "=https"',
            'proto-redir = "=https"', "no-location", "max-redirs = 0", "retry = 0", "silent",
            "max-time = " + _config_quote(str(timeout)),
            "connect-timeout = " + _config_quote(str(min(timeout, 10))),
            "max-filesize = " + str(MAX_RESPONSE_BYTES), 'header = "Accept: application/json"',
            "header = " + _config_quote("APCA-API-KEY-ID: " + key_id),
            "header = " + _config_quote("APCA-API-SECRET-KEY: " + secret_key),
            "write-out = " + _config_quote(STATUS_MARKER.decode() + "%{http_code}"), "",
        ]).encode()
        try:
            result = subprocess.run([executable, "-q", "--config", "-"], input=config,
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=timeout + 5,
                check=False, shell=False, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except (subprocess.TimeoutExpired, OSError):
            raise AlpacaTransportError("Historical data transfer failed; no retry attempted") from None
        raw = result.stdout
        if result.returncode != 0 or not isinstance(raw, bytes) or len(raw) > MAX_RESPONSE_BYTES + len(STATUS_MARKER) + 3:
            raise AlpacaTransportError("Historical data transfer failed or exceeded the size limit")
        pieces = raw.rsplit(STATUS_MARKER, 1)
        if len(pieces) != 2 or not re.fullmatch(rb"[1-5][0-9]{2}", pieces[1]):
            raise AlpacaTransportError("Historical data response has no valid HTTP status")
        return int(pieces[1]), pieces[0]


def _safe_json(body: bytes, credentials: tuple[str, str]) -> dict:
    def invalid(_):
        raise ValueError("Nonfinite JSON")
    text = body.decode("utf-8")
    if any(key in text for key in credentials):
        raise ValueError("Credential echo")
    value = json.loads(text, parse_constant=invalid)

    def inspect(item, depth=0):
        if depth > 30:
            raise ValueError("JSON nesting")
        if isinstance(item, str):
            if any(key in item for key in credentials):
                raise ValueError("Credential echo")
            if item.startswith(("{", "[")):
                try:
                    nested = json.loads(item, parse_constant=invalid)
                except ValueError:
                    return
                inspect(nested, depth + 1)
        elif isinstance(item, dict):
            for key, child in item.items():
                inspect(key, depth + 1)
                inspect(child, depth + 1)
        elif isinstance(item, list):
            for child in item:
                inspect(child, depth + 1)
        elif type(item) is float and not math.isfinite(item):
            raise ValueError("Nonfinite JSON")
    inspect(value)
    if not isinstance(value, dict):
        raise ValueError("Object expected")
    return value


def _records(payload: dict, query: dict, page: int) -> tuple[list[dict], str | None]:
    kind, symbol = query["kind"], query["symbol"]
    if "code" in payload or "message" in payload or "error" in payload:
        raise ValueError("Provider error payload")
    for field, expected in (("feed", "sip"), ("currency", "USD")):
        if field in payload and payload[field] != expected:
            raise ValueError("Unexpected provider metadata")
    group = payload.get(kind)
    if not isinstance(group, dict) or set(group) - {symbol}:
        raise ValueError("Unexpected provider symbol schema")
    entries = group.get(symbol, [])
    if entries is None:
        entries = []
    if not isinstance(entries, list) or len(entries) > query["limit"]:
        raise ValueError("Invalid or oversized page")
    token = payload.get("next_page_token")
    if token is not None and (not isinstance(token, str) or not 1 <= len(token) <= 4096 or any(ord(c) < 33 or ord(c) > 126 for c in token)):
        raise ValueError("Invalid pagination token")
    start, end = _timestamp(query["start"])[0], _timestamp(query["end"])[0]
    normalized = []
    previous = None
    for index, row in enumerate(entries):
        if not isinstance(row, dict):
            raise ValueError("Invalid record")
        stamp, timestamp = _timestamp(row.get("t"))
        if not start <= stamp <= end or previous is not None and stamp < previous:
            raise ValueError("Out-of-window or unordered record")
        previous = stamp
        flags = []
        item = {"symbol": symbol, "kind": kind, "timestamp": timestamp, "timestamp_ns": stamp,
                "feed_requested": "sip", "currency_requested": "USD", "page": page,
                "source_record_index": index, "is_fill": False}
        if kind == "quotes":
            item.update(bid=row.get("bp"), ask=row.get("ap"), bid_size=row.get("bs"), ask_size=row.get("as"),
                        bid_exchange=row.get("bx"), ask_exchange=row.get("ax"), conditions=row.get("c"), tape=row.get("z"),
                        # Alpaca's dated CTA/UTP change applies from Nov 3,
                        # 2025. Preserve raw sizes; never multiply them by 100.
                        size_unit=("shares" if timestamp[:10] >= "2025-11-03"
                                   else "provider_round_lots; lot size not inferred"),
                        size_unit_source_url="https://docs.alpaca.markets/us/v1.1/changelog/marketdata-bid-and-ask-size-display-change")
            if not _positive(item["bid"]) or not _positive(item["ask"]):
                flags.append("invalid_or_nonpositive_quote")
            elif item["bid"] > item["ask"]:
                flags.append("crossed_quote")
            elif item["bid"] == item["ask"]:
                flags.append("locked_quote")
            if not _positive(item["bid_size"]) or not _positive(item["ask_size"]):
                flags.append("invalid_or_empty_size")
            # Conditions require a strategy-specific eligibility rule. Even an
            # otherwise sound quote is evidence, never an assumed execution.
            if not isinstance(row.get("c"), list) or any(not isinstance(c, str) for c in row["c"]):
                flags.append("missing_or_invalid_conditions")
        elif kind == "trades":
            item.update(price=row.get("p"), size=row.get("s"), exchange=row.get("x"),
                        trade_id=row.get("i"), conditions=row.get("c"), tape=row.get("z"))
            if not _positive(item["price"]) or not _positive(item["size"]):
                flags.append("invalid_trade")
        else:
            item.update(open=row.get("o"), high=row.get("h"), low=row.get("l"), close=row.get("c"),
                        volume=row.get("v"), vwap=row.get("vw"), trade_count=row.get("n"),
                        timeframe="1Min", adjustment="raw")
            if not all(_positive(item[k]) for k in ("open", "high", "low", "close")):
                flags.append("invalid_bar_price")
            elif not item["low"] <= min(item["open"], item["close"]) <= max(item["open"], item["close"]) <= item["high"]:
                flags.append("inconsistent_ohlc")
            if not _positive(item["volume"], zero=True):
                flags.append("invalid_volume")
            if stamp + 60 * 1_000_000_000 > end:
                flags.append("bar_extends_past_query_end")
        item["quality_flags"] = flags
        item["valid_numeric_observation"] = not flags
        normalized.append(item)
    return normalized, token


def collect_alpaca_prices(store: Store, query: dict, out: str | Path, *, env_path: str | Path = ".env",
                          transport=None, now: datetime | None = None, _purpose: str = "historical_capture") -> dict:
    """Write receipt.json and records.json, returning the receipt (never keys).

    Missing keys -> missing_credentials with zero network calls. All captures
    end after denial/error; no implicit retries, endpoint/feed changes, or orders.
    Transport injection is for offline tests: get(url, key_id, secret_key) returns
    (HTTP status, raw bytes). now is an aware datetime for deterministic tests.
    """
    request = _query(query, now)
    target = Path(out)
    if target.exists():
        raise ValueError("Preserve existing captures; choose a new output directory")
    target.mkdir(parents=True, exist_ok=False)
    receipt = {"schema_version": "alpaca-historical-capture-v1", "purpose": _purpose,
               "provider": "Alpaca Market Data", "started_at": utc_now(), "query": request,
               "query_sha256": hashlib.sha256(canonical_json(request)).hexdigest(),
               "interval_semantics": "start and end inclusive; provider event timestamps",
               "feed_verification": "explicit SIP request; provider response may not echo feed",
               "currency_verification": "explicit USD request; provider response may not echo currency",
               "calendar_validation": "caller-provided session bounds; holiday status and source not independently verified",
               "interpretation": "Historical observations captured now; not proof of real-time availability, strategy eligibility, or fills.",
               "pages": [], "status": "started", "record_count": 0, "complete": False}
    records = []

    def finish(status):
        receipt.update(status=status, finished_at=utc_now(), record_count=len(records),
                       valid_numeric_record_count=sum(r["valid_numeric_observation"] for r in records),
                       complete=status in {"captured", "empty"})
        receipt["entitlement_status"] = ("denied" if status == "provider_denied" else
            "accepted_for_this_request" if any(p.get("status") == "captured" for p in receipt["pages"]) else "not_established")
        receipt["records_sha256"] = hashlib.sha256(canonical_json(records)).hexdigest()
        write_new_json(target / "records.json", records)
        write_new_json(target / "receipt.json", receipt)
        return receipt

    try:
        keys = (read_key("APCA_API_KEY_ID", env_path), read_key("APCA_API_SECRET_KEY", env_path))
    except (OSError, UnicodeError, ValueError):
        return finish("invalid_credentials")
    if not all(keys):
        receipt["missing_credentials"] = [name for name, key in zip(("APCA_API_KEY_ID", "APCA_API_SECRET_KEY"), keys) if not key]
        return finish("missing_credentials")
    if any(any(ord(c) < 33 or ord(c) > 126 for c in key) for key in keys):
        return finish("invalid_credentials")
    client = transport or CurlAlpacaTransport()
    token, seen_tokens = None, set()
    for page in range(1, request["max_pages"] + 1):
        params = {k: request[k] for k in ("start", "end", "asof", "feed", "currency", "limit")}
        params.update(symbols=request["symbol"], sort="asc")
        if request["kind"] == "bars":
            params.update(timeframe="1Min", adjustment="raw")
        if token:
            params["page_token"] = token
        url = BASE_URL + request["kind"] + "?" + urlencode(params)
        capture = {"page": page, "source_url": url, "capture_started_at": utc_now()}
        receipt["pages"].append(capture)
        try:
            status, body = client.get(url, *keys)
        except (AlpacaTransportError, OSError, subprocess.TimeoutExpired):
            capture.update(status="transport_failed", capture_finished_at=utc_now())
            return finish("transport_failed")
        capture.update(http_status=status, capture_finished_at=utc_now())
        if type(status) is not int or not 100 <= status <= 599 or not isinstance(body, bytes) or len(body) > MAX_RESPONSE_BYTES:
            capture["status"] = "invalid_response"
            return finish("invalid_response")
        capture["body_sha256"] = hashlib.sha256(body).hexdigest()
        try:
            payload = _safe_json(body, keys)
        except (ValueError, UnicodeError, RecursionError):
            capture.update(status="body_withheld", body_archived=False)
            return finish("provider_denied" if status in {401, 403, 422} else "invalid_response")
        capture.update(observation_id=store.observe(url, body, kind="alpaca_historical_" + request["kind"], status=status),
                       body_archived=True)
        if status != 200:
            capture["status"] = "provider_denied" if status in {401, 403, 422} else "http_error"
            return finish(capture["status"])
        try:
            current, next_token = _records(payload, request, page)
            if records and current and current[0]["timestamp_ns"] < records[-1]["timestamp_ns"]:
                raise ValueError("Unordered pagination")
            if next_token and next_token in seen_tokens:
                raise ValueError("Repeated pagination token")
        except (ValueError, TypeError, OverflowError):
            capture["status"] = "invalid_response"
            return finish("invalid_response")
        records.extend(current)
        capture.update(status="captured", record_count=len(current), has_next_page=next_token is not None)
        if next_token is None:
            return finish("captured" if records else "empty")
        seen_tokens.add(next_token)
        token = next_token
    return finish("truncated")


def alpaca_entitlement_smoke(store: Store, query: dict, out: str | Path, **kwargs) -> dict:
    """One-page, at most five-record SIP quote probe, not a complete price sample.

    Uses the same explicit historical/calendar contract. A successful 200 page
    tests this request's access only; an empty page does not establish coverage.
    """
    smoke = dict(query)
    smoke.update(kind="quotes", limit=5, max_pages=1)
    if "timeframe" in smoke or "adjustment" in smoke:
        raise ValueError("Entitlement smoke requires a quote query")
    return collect_alpaca_prices(store, smoke, out, _purpose="entitlement_smoke", **kwargs)
