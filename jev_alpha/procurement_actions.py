"""Attach independent corporate-action reviews to immutable market receipts.

This offline CLI never fetches prices/actions, edits a protocol, or infers an
action-free interval from an empty provider response. Review templates contain
only unresolved records. Resolved assessments require explicit review, complete
interval coverage, and a hashed local capture of the cited primary evidence.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import date
import hashlib
import json
from pathlib import Path
from urllib.parse import urlsplit

from .procurement_returns import _digest, _frozen, _number, _settings, _symbols, _verify_market_hash, plan_windows
from .store import read_json, utc_now, write_new_json


def _bound(signals, market, protocol):
    signal_hash = _frozen(signals)
    _verify_market_hash(market)
    _settings(protocol)
    if signals.get("protocol_sha256", _digest(protocol)) != _digest(protocol):
        raise ValueError("Protocol differs from the frozen signals")
    if market.get("signals_sha256") != signal_hash:
        raise ValueError("Market receipt belongs to different frozen signals")
    if market.get("action_review_sha256"):
        raise ValueError("Attach a new review to the original market receipt, preserving earlier derived receipts")
    return {"signals_sha256": signal_hash, "market_sha256": _digest(market),
            "protocol_sha256": _digest(protocol)}


def _required(signals, market, protocol):
    required = set()
    for row in plan_windows(signals.get("events", []), market.get("calendar", [])):
        if row["status"] != "planned" or row["signal_status"] != "signal":
            continue
        for symbol in set(_symbols(protocol)) | {"IWM", row["symbol"]}:
            required.add((symbol, row["entry_at"][:10], row["exit_at"][:10]))
    return sorted(required)


def review_template(signals, market, protocol):
    """Return an unresolved template for each distinct stock/benchmark interval."""
    binding = _bound(signals, market, protocol)
    return {"schema_version": "procurement-action-review-v1", **binding,
            "created_at": utc_now(), "records": [
                {"symbol": symbol, "entry_date": entry, "exit_date": exit_day,
                 "status": "unreviewed", "reviewed": False, "reviewer": None,
                 "coverage_complete": False, "coverage_statement": None,
                 "evidence": [], "dividends": [], "other_actions": []}
                for symbol, entry, exit_day in _required(signals, market, protocol)],
            "instructions": (
                "Review the complete interval independently, including splits, mergers, symbol changes and dividends. "
                "A provider's empty corporate-action response is not proof of no action. Evidence items require "
                "source_url, capture_path inside the experiment root, capture_sha256, and locator. Use none only "
                "with reviewed=True, reviewer, complete coverage, and evidence. For cash_dividend list each "
                "ex_date and amount_per_share; only entry_date < ex_date <= exit_date is owned by this trade. "
                "Non-cash/unsupported actions or incomplete evidence remain unknown. Do not edit frozen signals, "
                "protocol, or the original market manifest.")}


def _evidence(root, rows):
    if not isinstance(rows, list) or not rows:
        raise ValueError("Resolved corporate-action review requires captured evidence")
    root = Path(root).resolve()
    checked = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Evidence must identify a source and hashed local capture")
        url = urlsplit(row.get("source_url", ""))
        if url.scheme not in {"https", "http"} or not url.hostname or not row.get("locator"):
            raise ValueError("Evidence needs a source URL and a specific source locator")
        path = (root / row.get("capture_path", "")).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError("Evidence capture must be a file inside the experiment root")
        if hashlib.sha256(path.read_bytes()).hexdigest() != row.get("capture_sha256"):
            raise ValueError("Corporate-action evidence capture hash mismatch")
        checked.append({**row, "capture_path": path.relative_to(root).as_posix()})
    return checked


def _assessment(root, row):
    status = row.get("status")
    if status not in {"none", "cash_dividend", "unknown", "unsupported_action", "unreviewed"}:
        raise ValueError("Unsupported action-review status")
    if status in {"unknown", "unsupported_action", "unreviewed"}:
        return {"status": "unknown", "coverage_complete": False, "evidence": [],
                "review_status": status, "reason": row.get("reason", "independent_action_review_unresolved")}
    if (row.get("reviewed") is not True or not isinstance(row.get("reviewer"), str)
            or not row["reviewer"].strip() or row.get("coverage_complete") is not True
            or not isinstance(row.get("coverage_statement"), str) or not row["coverage_statement"].strip()):
        raise ValueError("Resolved action review needs an explicit reviewer and complete interval assessment")
    evidence = _evidence(root, row.get("evidence"))
    if row.get("other_actions") != []:
        raise ValueError("Non-cash corporate actions require an unresolved/unsupported assessment")
    dividends = row.get("dividends")
    if not isinstance(dividends, list) or (status == "none" and dividends) or (status == "cash_dividend" and not dividends):
        raise ValueError("Dividend details must agree with action status")
    total, seen = 0.0, set()
    for dividend in dividends:
        if not isinstance(dividend, dict):
            raise ValueError("Dividend detail must be an object")
        try:
            ex_date = date.fromisoformat(dividend["ex_date"]).isoformat()
        except (ValueError, TypeError, KeyError):
            raise ValueError("Dividend requires a valid ex-date") from None
        amount = dividend.get("amount_per_share")
        identity = (ex_date, dividend.get("identifier", "regular"))
        if (not row["entry_date"] < ex_date <= row["exit_date"]
                or not _number(amount, True) or identity in seen):
            raise ValueError("Invalid/duplicate dividend or ex-date outside the owned interval")
        seen.add(identity)
        total += amount
    if not _number(total):
        raise ValueError("Nonfinite total cash dividend")
    return {"status": status, "coverage_complete": True, "evidence": evidence,
            "cash_dividend_per_entry_share": total, "dividends": dividends,
            "reviewer": row["reviewer"], "coverage_statement": row["coverage_statement"]}


def attach_review(root, signals, market, review, protocol, *, out):
    """Write a derived receipt; missing reviews remain explicit unknown outcomes."""
    binding = _bound(signals, market, protocol)
    if review.get("schema_version") != "procurement-action-review-v1" or any(
            review.get(key) != value for key, value in binding.items()):
        raise ValueError("Action review is not bound to these signals, protocol and market receipt")
    required = _required(signals, market, protocol)
    supplied = {}
    for row in review.get("records", []):
        key = (row.get("symbol"), row.get("entry_date"), row.get("exit_date"))
        if key not in required or key in supplied:
            raise ValueError("Duplicate or out-of-scope action-review interval")
        supplied[key] = row
    actions = {}
    for key in required:
        row = supplied.get(key)
        actions["|".join(key)] = (_assessment(root, row) if row is not None else
            {"status": "unknown", "coverage_complete": False, "evidence": [], "reason": "action_review_missing"})
    result = deepcopy(market)
    result.update(actions=actions, parent_market_sha256=binding["market_sha256"],
                  action_review_sha256=_digest(review), action_review=deepcopy(review),
                  action_review_attached_at=utc_now(),
                  action_review_summary={"required_intervals": len(required),
                      "resolved_intervals": sum(a["coverage_complete"] for a in actions.values()),
                      "unresolved_intervals": sum(not a["coverage_complete"] for a in actions.values())})
    # A derived receipt has its own identity as well as both input identities.
    result.pop("sha256", None)
    result["sha256"] = _digest(result)
    write_new_json(Path(out), result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("template", "attach-review"))
    parser.add_argument("--root", type=Path, default=Path("data/procurement-v1"))
    parser.add_argument("--signals", type=Path, required=True)
    parser.add_argument("--market", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--review", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    signals, market, protocol = (read_json(p) for p in (args.signals, args.market, args.protocol))
    if args.command == "template":
        result = review_template(signals, market, protocol)
        write_new_json(args.out, result)
    else:
        if args.review is None:
            parser.error("attach-review requires --review")
        result = attach_review(args.root, signals, market, read_json(args.review), protocol, out=args.out)
    print(json.dumps({"command": args.command, "output": str(args.out.resolve()),
                      "records": len(result.get("records", [])),
                      "action_review_summary": result.get("action_review_summary"), "network_requests": 0}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
