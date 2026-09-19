"""Bounded public daily-price snapshots for a registered development diagnostic.

The provider's adjusted closes are not executable quotes. This module neither
computes nor prints returns. It does not retry, authenticate or bypass denials.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
import json
from pathlib import Path
import subprocess
from urllib.parse import urlencode

from .experiment import digest
from .store import Store, utc_now, write_new_json


def collect_daily_prices(store: Store, protocol: dict, signals: dict, out: Path) -> dict:
    if out.exists():
        raise ValueError("Preserve existing price snapshots; choose a new directory")
    if (signals.get("protocol_sha256") != digest(protocol)
            or signals.get("status") not in {"complete", "complete_with_failures"}
            or not signals.get("signals")
            or any(row["arms"][arm]["status"] not in {"completed", "failed"}
                   for row in signals["signals"] for arm in ("jev", "baseline"))):
        raise ValueError("Finalized frozen predictions are required before opening prices")
    if protocol.get("symbols") != ["NUE", "STLD"] or protocol.get("benchmark") != "SPY":
        raise ValueError("This bounded collector supports only the registered CORE basket")
    # The collector is also called directly, outside the CLI. Enforce the
    # pre-outcome cohort contract here so a caller cannot accidentally open
    # prices after freezing only part of the registered predictions.
    selected = protocol.get("selected")
    rows = signals.get("signals")
    if not isinstance(selected, list) or not selected or not isinstance(rows, list):
        raise ValueError("Registered and predicted daily cohorts are required")
    expected = {row.get("document_id"): row.get("publication_date") for row in selected}
    actual = {row.get("document_id"): row.get("publication_date") for row in rows}
    registered_ids = protocol.get("document_ids")
    if (any(not isinstance(doc, str) or not doc for doc in expected)
            or len(expected) != len(selected) or len(actual) != len(rows)
            or actual != expected or not isinstance(registered_ids, list)
            or len(registered_ids) != len(expected) or set(registered_ids) != set(expected)):
        raise ValueError("Predicted documents and dates must match the entire frozen cohort")
    arms = ["jev", "baseline", "always_long", "cash"]
    if protocol.get("arms") != arms or any(set(row.get("arms", {})) != set(arms) for row in rows):
        raise ValueError("Predicted arms differ from the registered daily comparison")
    manifest_hash = protocol.get("prediction_manifest_sha256")
    if not isinstance(manifest_hash, str) or not manifest_hash or signals.get("prediction_manifest_sha256") != manifest_hash:
        raise ValueError("Frozen signal manifest differs from the registered prediction manifest")
    dates = [date.fromisoformat(row["publication_date"]) for row in protocol["selected"]]
    begin, end = min(dates) - timedelta(days=10), max(dates) + timedelta(days=40)
    if (end - begin).days > 5000:
        raise ValueError("Daily data range exceeds the bounded pilot")
    timestamp = lambda d: int(datetime.combine(d, datetime.min.time(), timezone.utc).timestamp())
    report = {"schema_version": "daily-price-capture-v1", "created_at": utc_now(),
              "protocol_sha256": digest(protocol), "signals_sha256": digest(signals),
              "signals_frozen_at": signals["frozen_at"], "provider": "Yahoo Finance public chart",
              "start_inclusive": begin.isoformat(), "end_exclusive": end.isoformat(),
              "symbols": [], "interpretation": "Adjusted daily close proxy; no executable-price claim."}
    out.mkdir(parents=True)
    for symbol in ["NUE", "STLD", "SPY"]:
        query = urlencode({"period1": timestamp(begin), "period2": timestamp(end), "interval": "1d",
                           "events": "div,splits", "includeAdjustedClose": "true"})
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?{query}"
        # Public endpoint, no credentials or arbitrary user URL. Default curl
        # configuration and retries disabled; HTTP denials remain failures.
        try:
            result = subprocess.run(["curl.exe", "-q", "--silent", "--show-error", "--fail",
                                     "--retry", "0", "--max-time", "30", "--max-filesize", "15000000", url],
                                    capture_output=True, timeout=35)
        except (subprocess.TimeoutExpired, OSError) as exc:
            report["symbols"].append({"symbol": symbol, "source_url": url, "status": "failed",
                                       "error_type": type(exc).__name__})
            continue
        if result.returncode:
            report["symbols"].append({"symbol": symbol, "source_url": url, "status": "failed",
                                       "curl_exit_code": result.returncode})
            continue
        try:
            payload = json.loads(result.stdout)
            if not isinstance(payload, dict) or payload.get("chart", {}).get("error"):
                raise ValueError("Provider error payload")
        except (ValueError, UnicodeError, AttributeError):
            report["symbols"].append({"symbol": symbol, "source_url": url, "status": "invalid_response"})
            continue
        observation = store.observe(url, result.stdout, kind="daily_price_chart")
        write_new_json(out / f"{symbol}.json", payload)
        report["symbols"].append({"symbol": symbol, "source_url": url, "status": "captured",
                                   "observation_id": observation, "raw_sha256": store.put_blob(result.stdout)})
    report["status"] = "complete" if all(r["status"] == "captured" for r in report["symbols"]) else "incomplete"
    write_new_json(out / "capture.json", report)
    return report
