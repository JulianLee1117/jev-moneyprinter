"""Freeze source-only development signals before opening price outcomes."""
from __future__ import annotations

from pathlib import Path
import copy

from .experiment import digest, run_one
from .jev import JevRequestError
from .store import Store, utc_now, write_new_json


def signal_from_response(response: dict, threshold: float = .70) -> dict:
    if not 0 <= threshold <= 1:
        raise ValueError("Threshold must be a probability")
    answer = response["answers"]["domestic_producer_channel"]
    probability = answer["probabilities"]["positive"]
    side = "long" if answer["choice"] == "positive" and probability >= threshold else "cash"
    return {"status": "completed", "side": side, "positive_probability": probability,
            "chosen_channel": answer["choice"], "calibrated": False}


def predict_daily(store: Store, manifest: dict, protocol: dict, out: Path,
                  *, live: bool = False, progress=None, resume: dict | None = None) -> dict:
    if out.exists():
        raise ValueError("Output exists; preserve frozen predictions")
    if digest(manifest) != protocol["prediction_manifest_sha256"]:
        raise ValueError("Prediction manifest differs from registered protocol")
    expected = {r["document_id"]: r["publication_date"] for r in protocol["selected"]}
    entries = manifest["requests"]
    if len(entries) != len(expected) or {r["document_id"] for r in entries} != set(expected):
        raise ValueError("Prediction cohort changed or has duplicates")
    for entry in entries:
        if entry["publication_date"] != expected[entry["document_id"]] or digest(entry["request"]) != entry["request_sha256"]:
            raise ValueError("Frozen request or publication date changed")
    if protocol["arms"] != ["jev", "baseline", "always_long", "cash"]:
        raise ValueError("Unsupported registered arm set")
    threshold = protocol["model_policy"]["positive_probability_threshold"]
    if threshold != .70 or protocol["model_policy"]["confidence_statistic_used"] is not False:
        raise ValueError("This runner requires the frozen option-probability rule")
    budget = protocol["model_policy"]["provider_phase_budget_usd"]
    prior = {}
    if resume is not None:
        if (resume.get("protocol_sha256") != digest(protocol)
                or resume.get("prediction_manifest_sha256") != digest(manifest)
                or resume.get("prices_opened_by_runner") is not False):
            raise ValueError("Resume must preserve the same price-blind frozen inputs")
        prior = {row["document_id"]: row for row in resume["signals"]}
        if len(prior) != len(resume["signals"]) or set(prior) != set(expected):
            raise ValueError("Resume cohort differs from registered cohort")
        for doc, row in prior.items():
            if row["publication_date"] != expected[doc] or set(row["arms"]) != set(protocol["arms"]):
                raise ValueError("Resume dates or arms differ from registered cohort")
            for arm in ("jev", "baseline"):
                old = row["arms"][arm]
                if old["status"] not in {"completed", "failed", "skipped_after_failure", "dry_run"}:
                    raise ValueError("Unsupported resume signal status")
                if old["status"] == "completed" and old.get("side") not in {"long", "cash"}:
                    raise ValueError("Invalid completed resume signal")
    signals, runs, halted = [], [], False
    for index, entry in enumerate(entries):
        doc = entry["document_id"]
        row = {"document_id": doc, "publication_date": entry["publication_date"], "arms": {
            "always_long": {"status": "completed", "side": "long"},
            "cash": {"status": "completed", "side": "cash"}}}
        for arm in ["jev", "baseline"]:
            old = prior.get(doc, {}).get("arms", {}).get(arm)
            if live and old and old["status"] in {"completed", "failed"}:
                row["arms"][arm] = copy.deepcopy(old)
                runs.extend(copy.deepcopy([r for r in resume.get("runs", [])
                                           if r["document_id"] == doc and r["arm"] == arm]))
                if progress:
                    progress({"document": index + 1, "of": len(entries), "document_id": doc,
                              "arm": arm, "status": old["status"], "side": old.get("side"), "preserved": True})
                continue
            status = "skipped_after_failure" if halted else "dry_run"
            row["arms"][arm] = {"status": status, "side": None}
            if live and not halted:
                try:
                    run = run_one(store, entry["request"], arm=arm, transport="curl", phase_budget_usd=budget)
                    write_new_json(out / "responses" / f"{doc}-{arm}.json", run)
                    row["arms"][arm] = signal_from_response(run["response"], threshold)
                    runs.append({"document_id": doc, "arm": arm, "request_sha256": run["request_hash"],
                                 "reported_cost_usd": run["response"].get("usage", {}).get("cost"),
                                 "cached": run["cached"], "latency_ms": run["latency_ms"]})
                except (JevRequestError, ValueError, OSError) as exc:
                    row["arms"][arm] = {"status": "failed", "side": None, "error_type": type(exc).__name__,
                                         "diagnostic_sha256": getattr(exc, "response_blob_sha256", None),
                                         "validation_reason": getattr(exc, "validation_reason", None)}
                    # A safely archived invalid completion is a measured model
                    # failure, not a reason to skip independent future inputs.
                    # Transport/account/budget failures still stop new calls.
                    halted = not (getattr(exc, "response_blob_sha256", None)
                                  and getattr(exc, "validation_reason", None))
            if progress:
                progress({"document": index + 1, "of": len(entries), "document_id": doc, "arm": arm,
                          "status": row["arms"][arm]["status"], "side": row["arms"][arm]["side"]})
        signals.append(row)
    failed = any(row["arms"][arm]["status"] == "failed" for row in signals for arm in ("jev", "baseline"))
    report = {"schema_version": "daily-signals-v1", "frozen_at": utc_now(), "protocol_sha256": digest(protocol),
              "prediction_manifest_sha256": digest(manifest), "prices_opened_by_runner": False,
              "status": "incomplete" if halted else ("complete_with_failures" if failed else "complete") if live else "dry_run",
              "resume_signals_sha256": digest(resume) if resume is not None else None,
              "signals": signals, "runs": runs,
              "interpretation": "Source-only development channel labels; no alpha or executable trade claim."}
    write_new_json(out / "signals.json", report)
    return report
