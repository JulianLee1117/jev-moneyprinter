"""Point-in-time fiscal revenue selection from archived public SEC facts.

Current SEC metadata is a retrieval source, not proof of as-filed document bytes.
Subsidiary mappings require separate dated source evidence supplied by reviewers.
"""
from __future__ import annotations

from datetime import date
import hashlib
import json
from pathlib import Path
import urllib.request

from .store import Store, read_json, utc_now, write_new_json

CIKS = {"GVA": 861459, "ROAD": 1718227, "STRL": 874238,
        "TPC": 77543, "PRIM": 1361538, "ORN": 1402829}
REVENUE_TAGS = ("RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues",
                "SalesRevenueNet", "ContractRevenue", "SalesRevenueGoodsNet")


def annual_revenues(facts: dict, symbol: str) -> list[dict]:
    """Keep original FY 10-K observations filed before the experiment end.

    No future-filed comparative/restated facts may replace a past observation.
    Conflicting revenue tags for the same filing require review.
    """
    grouped = {}
    for tag in REVENUE_TAGS:
        for row in facts.get("facts", {}).get("us-gaap", {}).get(tag, {}).get("units", {}).get("USD", []):
            try:
                duration = (date.fromisoformat(row["end"]) - date.fromisoformat(row["start"])).days
                if (row.get("form") != "10-K" or row.get("fp") != "FY" or not 330 <= duration <= 380
                        or not "2024-01-01" <= row["filed"] <= "2025-12-31"
                        or int(row["end"][:4]) != row.get("fy") or row["val"] <= 0):
                    continue
                key = (row["filed"], row["end"], row["accn"])
                grouped.setdefault(key, []).append((tag, row["val"]))
            except (KeyError, TypeError, ValueError):
                continue
    output = []
    for (filed, end, accn), values in sorted(grouped.items()):
        amounts = {value for _, value in values}
        output.append({"symbol": symbol, "published_date": filed, "period_end": end,
            "accession": accn, "revenue_usd": next(iter(amounts)) if len(amounts) == 1 else None,
            "status": "sec_fact" if len(amounts) == 1 else "conflicting_tags",
            "tags": values, "historical_bytes_verified": False})
    return output


def revenue_asof(records: list[dict], symbol: str, asof: str) -> dict | None:
    # A filing dated the event day has no exact release clock here; use next day.
    eligible = [r for r in records if r.get("symbol") == symbol
                and r.get("published_date", "9999") < asof]
    if not eligible:
        return None
    last = max(eligible, key=lambda r: (r["period_end"], r["published_date"]))
    return last if isinstance(last.get("revenue_usd"), (int, float)) and last["revenue_usd"] > 0 else None


def capture_references(root: Path, *, live=False) -> dict:
    out = root / "references" / "sec-facts.json"
    if out.exists():
        return read_json(out)
    if not live:
        return {"status": "dry_run", "symbols": list(CIKS), "network_requests": 0}
    captures, revenues = [], []
    with Store(root / "references" / "archive") as store:
        for symbol, cik in CIKS.items():
            for kind, url in (
                ("facts", f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"),
                ("submissions", f"https://data.sec.gov/submissions/CIK{cik:010d}.json")):
                row = {"symbol": symbol, "cik": cik, "kind": kind, "url": url, "captured_at": utc_now()}
                try:
                    request = urllib.request.Request(url, headers={"User-Agent": "jev-alpha-research/0.1 public-research github.com/JulianLee1117/jev-moneyprinter"})
                    with urllib.request.urlopen(request, timeout=45) as response:
                        raw = response.read(20_000_001)
                        if len(raw) > 20_000_000:
                            raise ValueError("Reference body exceeds cap")
                    payload = json.loads(raw)
                    store.observe(url, raw, kind="procurement_reference_" + kind)
                    row.update(status="captured", sha256=hashlib.sha256(raw).hexdigest(), name=payload.get("entityName", payload.get("name")))
                    if kind == "facts":
                        rows = annual_revenues(payload, symbol)
                        for record in rows:
                            record.update(source_url=url, source_sha256=row["sha256"])
                        revenues.extend(rows)
                    else:
                        row.update(tickers=payload.get("tickers"), exchanges=payload.get("exchanges"))
                except Exception as exc:
                    row.update(status="unavailable", error_type=type(exc).__name__)
                captures.append(row)
    report = {"schema_version": "procurement-reference-v1", "captured_at": utc_now(),
              "captures": captures, "annual_revenues": revenues,
              "current_identity_is_not_historical_listing_proof": True}
    write_new_json(out, report)
    return report
