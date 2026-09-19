"""Opt-in bounded source recorder; no scheduler, model inference or trade signals."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
from pathlib import Path

from .procurement_sources import collect
from .store import canonical_json, read_json, utc_now, write_new_json


def _hash(value):
    return hashlib.sha256(canonical_json(value)).hexdigest()


def _time(value):
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("Recorder clocks require timezone-aware UTC timestamps")
    return result.astimezone(timezone.utc)


def capture_once(root: Path, protocol: dict, decision_report: dict, *, live=False) -> dict:
    """One fresh snapshot, only after the explicit research gate; never start a loop.

    The event cap counts a separate reviewed eligible-event ledger. Recording
    bytes never makes an event eligible. Each source receives a startup baseline.
    """
    decision = decision_report.get("decisions", {}).get("jev", {}).get("decision")
    if (decision not in {"promising_association_only", "promising_requires_prospective_confirmation"}
            or decision_report.get("jev_contribution_verified") is not True):
        raise ValueError("Recorder requires a promising Jev decision and explicit jev_contribution_verified=True")
    sources = protocol.get("sources", [])
    if (type(live) is not bool or protocol.get("orders_enabled") is not False
            or not sources or len(sources) > 10 or len({s["id"] for s in sources}) != len(sources)):
        raise ValueError("Expected the frozen source roster and simulation-only protocol")
    cap, days, interval = (protocol.get(k, default) for k, default in
        (("prospective_max_events", 30), ("prospective_max_days", 90), ("prospective_poll_seconds", 300)))
    if (type(cap) is not int or not 1 <= cap <= 30 or type(days) is not int or not 1 <= days <= 90
            or type(interval) is not int or interval != 300):
        raise ValueError("Recorder limits are <=30 eligible events, <=90 days and a 300-second poll interval")
    base = Path(root).resolve() / "prospective"
    config = {"protocol_sha256": _hash(protocol), "decision_report_sha256": _hash(decision_report),
              "sources": sources, "max_eligible_events": cap, "max_days": days, "poll_seconds": interval}
    if not live:
        return {"status": "dry_run", "configuration": config, "recorder_only": True, "alpha_proven": False}
    base.mkdir(parents=True, exist_ok=True)
    lock = base / "capture.lock"
    with lock.open("x", encoding="utf-8") as handle:
        handle.write(utc_now())
    try:
        checkpoints = sorted((base / "checkpoints").glob("*.json"))
        previous, previous_hash, candidate_ids = None, None, set()
        for index, path in enumerate(checkpoints):
            saved = read_json(path)
            if (saved.get("sequence") != index or saved.get("previous_sha256") != previous_hash
                    or saved.get("configuration_sha256") != _hash(config)):
                raise ValueError("Prospective checkpoint chain is incomplete or changed")
            candidate_ids.update(c["candidate_id"] for c in saved.get("candidates", []))
            previous, previous_hash = saved, _hash(saved)
        registered = base / "configuration.json"
        if registered.exists():
            if read_json(registered) != config:
                raise ValueError("Prospective configuration changed after registration")
        else:
            if previous:
                raise ValueError("Prospective configuration is missing")
            write_new_json(registered, config)
        now = _time(utc_now())
        started = previous["started_at"] if previous else now.isoformat()
        count = previous["candidate_count"] if previous else 0
        eligible, eligibility_hashes = set(), {}
        for path in sorted((base / "eligible-events").glob("*.json")):
            row = read_json(path)
            eligibility_hashes[path.name] = _hash(row)
            if (row.get("schema_version") != "procurement-prospective-eligibility-v1" or row.get("reviewed") is not True
                    or row.get("candidate_id") not in candidate_ids or row.get("status") not in {"eligible", "ineligible", "unknown"}
                    or not isinstance(row.get("event_id"), str) or not row["event_id"].strip()):
                raise ValueError("Eligibility ledger requires reviewed observations linked to recorded candidate IDs")
            if row["status"] == "eligible": eligible.add(row["event_id"])
        if previous and any(eligibility_hashes.get(name) != sha for name, sha in previous.get("eligibility_ledger_hashes", {}).items()):
            raise ValueError("Previously observed eligibility ledger records were removed or changed")
        if (now - _time(started)).total_seconds() >= days * 86400 or len(eligible) >= cap:
            return {"status": "bounded_stop", "candidate_count": count, "eligible_event_count": len(eligible), "recorder_only": True, "alpha_proven": False}
        elapsed = (now - _time(previous["poll_started_at"])).total_seconds() if previous else None
        if elapsed is not None and elapsed < interval:
            return {"status": "not_due", "next_poll_at": (_time(previous["poll_started_at"]) + timedelta(seconds=interval)).isoformat()}
        sequence = len(checkpoints)
        snapshot = base / "snapshots" / f"{sequence:06d}"
        window = {**protocol, "historical_start": (now.date() - timedelta(days=30)).isoformat(),
                  "historical_end": (now.date() + timedelta(days=30)).isoformat(), "collection_mode": "census"}
        write_new_json(snapshot / "collection-protocol.json", window)
        failures, candidates = [], []
        try:
            manifest = collect(snapshot, window, live=True)
        except Exception as exc:
            manifest = {"documents": [], "source_results": []}
            failures.append({"reason": "collector_failed", "exception_type": type(exc).__name__})
        baseline = set(previous["baseline_sources"] if previous else [])
        known = set(previous["known_source_versions"] if previous else [])
        known_urls = set(previous.get("known_source_urls", []) if previous else [])
        initialized = set()
        coverage = {s["id"]: False for s in sources}
        coverage.update({r["source_id"]: r.get("complete") is True for r in manifest.get("source_results", [])})
        prior_coverage = previous.get("source_inventory_coverage", {}) if previous else {}
        failures.extend({"source_id": r.get("source_id"), "reason": "source_incomplete", "errors": r.get("errors", [])}
                        for r in manifest.get("source_results", []) if r.get("complete") is not True)
        returned = {r.get("source_id") for r in manifest.get("source_results", [])}
        failures.extend({"source_id": s["id"], "reason": "source_result_missing"} for s in sources if s["id"] not in returned)
        for doc in manifest.get("documents", []):
            if doc.get("source_id") not in {s["id"] for s in sources}:
                failures.append({"reason": "unexpected_source"})
                continue
            if not doc.get("sha256") or not doc.get("blob_path") or doc.get("extraction_status") == "capture_failed":
                failures.append({"document_id": doc.get("document_id"), "reason": "document_capture_failed"})
                continue
            path = (snapshot / doc["blob_path"]).resolve()
            try:
                if not path.is_relative_to(snapshot.resolve()) or hashlib.sha256(path.read_bytes()).hexdigest() != doc["sha256"]:
                    raise ValueError("Source byte digest mismatch")
                captured = _time(doc["captured_at"])
            except (OSError, ValueError, KeyError, TypeError):
                failures.append({"document_id": doc.get("document_id"), "reason": "invalid_source_bytes_or_clock"})
                continue
            version = _hash([doc["source_id"], doc["sha256"]])
            url_key = _hash([doc["source_id"], doc["url"]])
            unseen = version not in known
            new_url = url_key not in known_urls
            known.add(version)
            known_urls.add(url_key)
            text_path = (snapshot / doc.get("text_path", "")).resolve()
            try:
                substantive = (doc.get("extraction_status", "").startswith("extracted_")
                    and text_path.is_relative_to(snapshot.resolve())
                    and len(text_path.read_text(encoding="utf-8").strip()) >= 50)
            except (OSError, UnicodeError):
                substantive = False
            if not substantive:
                failures.append({"document_id": doc.get("document_id"), "reason": "substantive_text_unavailable"})
                continue
            initialized.add(doc["source_id"])
            if unseen and doc["source_id"] in baseline:
                candidates.append({"candidate_id": version, "document_id": doc["document_id"],
                    "source_id": doc["source_id"], "url": doc["url"], "sha256": doc["sha256"],
                    "snapshot_relative_path": snapshot.relative_to(base).as_posix(),
                    "blob_path": doc["blob_path"], "first_seen_at": captured.isoformat(),
                    "inventory_coverage_complete": coverage[doc["source_id"]],
                    "may_preexist_observation": (new_url or not coverage[doc["source_id"]]
                        or not prior_coverage.get(doc["source_id"], False) or elapsed is None or elapsed > interval),
                    "observation_kind": "newly_observed_url" if new_url else "changed_bytes_at_observed_url",
                    "publication_time": None, "status": "unclassified_new_or_changed_bytes",
                    "eligible_trading_event": None, "ready_at": None})
                count += 1
        baseline.update(initialized)
        record = {"schema_version": "procurement-prospective-checkpoint-v1", "sequence": sequence,
            "previous_sha256": previous_hash, "configuration_sha256": _hash(config), "started_at": started,
            "poll_started_at": now.isoformat(), "poll_finished_at": utc_now(),
            "poll_gap_seconds": elapsed, "unobserved_delay_seconds": max(0, elapsed - interval) if elapsed is not None else None,
            "baseline_sources": sorted(baseline), "known_source_versions": sorted(known),
            "known_source_urls": sorted(known_urls), "source_inventory_coverage": coverage,
            "baseline_policy": "First substantive successful capture initializes the observed set; this does not establish exhaustive inventory or first publication",
            "candidate_count": count, "eligible_event_count": len(eligible), "candidates": candidates, "failures": failures,
            "eligibility_ledger_hashes": eligibility_hashes,
            "snapshot_manifest": manifest.get("manifest_path"), "recorder_only": True,
            "eligibility_classification_pending": True, "alpha_proven": False,
            "status": "captured"}
        write_new_json(base / "checkpoints" / f"{sequence:06d}.json", record)
        return record
    finally:
        lock.unlink()
