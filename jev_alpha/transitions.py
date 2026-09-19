"""Price-blind, source-bound tests of legal changes against two different priors.

A preliminary estimate is not the operative pre-event deposit rate. Reviewed
table transcriptions are inputs, not model extractions or legal conclusions.
The model checks semantics in parallel; decimal arithmetic stays deterministic.
"""
from __future__ import annotations

import copy
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path

from .experiment import digest, run_one
from .jev import JevRequestError, build_request, estimate_request, validate_response
from .store import Store, utc_now, write_new_json


def _percent(value: object) -> Decimal:
    if not isinstance(value, str):
        raise ValueError("Rates must be decimal strings in percent units")
    try:
        number = Decimal(value)
    except InvalidOperation:
        raise ValueError("Invalid rate") from None
    if not number.is_finite() or number < 0 or number > 10000:
        raise ValueError("Rate outside finite nonnegative percent range")
    return number


def numeric_comparisons(rows: list[dict]) -> list[dict]:
    """Compare explicit same-entity, same-case, same-type source observations.

    'operative' describes the supplied final or provisional deposit instruction,
    not proof no intervening instruction changed it. Missing priors stay unknown.
    """
    results, seen = [], set()
    for row in rows:
        identifier = row["candidate_id"]
        if not identifier or identifier in seen:
            raise ValueError("Unique candidate IDs required")
        seen.add(identifier)
        current = row["current"]
        value = _percent(current["percent"])
        result = {"candidate_id": identifier, "current_percent": str(value),
                  "delta_unit": "percentage_points", "comparisons": {},
                  "operative_history_complete": False}
        for reference in ("preliminary", "earlier_operative_deposit"):
            prior = row.get(reference)
            if prior is None:
                result["comparisons"][reference] = {"status": "unknown"}
                continue
            if any(not current.get(key) or current[key] != prior.get(key)
                   for key in ("case_id", "entity_id", "rate_type")):
                raise ValueError("Rate comparison crosses case, entity or rate type")
            if reference == "preliminary" and (not current.get("review_period") or
                    current["review_period"] != prior.get("review_period")):
                raise ValueError("Preliminary and final estimates must share review period")
            previous = _percent(prior["percent"])
            difference = value - previous
            result["comparisons"][reference] = {
                "status": "source_table_comparison_only", "prior_percent": str(previous),
                "delta_percentage_points": str(difference),
                "direction": "increase" if difference > 0 else "decrease" if difference < 0 else "unchanged"}
        results.append(result)
    return results


def guard_transition(manifest: dict, response: dict) -> dict:
    """Check raw interpretation against arithmetic without using expected labels.

    This development review gate neither validates the input transcription nor
    authorizes a trade. The model's selected answer and probabilities stay intact.
    """
    request = manifest["request"]
    if digest(request) != manifest["request_sha256"]:
        raise ValueError("Changed frozen request")
    raw = validate_response(response, request)
    state = request["state"].get("evidence", request["state"])
    comparisons = numeric_comparisons(state["rate_candidates"])
    directions = [r["comparisons"]["preliminary"].get("direction") for r in comparisons]
    if not directions or None in directions:
        expected_direction = "unknown"
    elif "increase" in directions and "decrease" in directions:
        expected_direction = "mixed"
    elif "increase" in directions:
        expected_direction = "increase"
    elif "decrease" in directions:
        expected_direction = "decrease"
    else:
        expected_direction = "unchanged"
    findings = []
    answers = raw["answers"]
    chosen = answers.get("preliminary_margin_direction", {}).get("choice")
    if chosen is not None and chosen != expected_direction:
        findings.append({"code": "model_numeric_direction_disagreement",
                         "model_choice": chosen, "deterministic_direction": expected_direction,
                         "action": "Review; preserve raw distribution. Do not use this label as a signal."})
    if answers.get("summary_detail_consistency", {}).get("choice") == "conflicting":
        findings.append({"code": "source_contradiction_flagged", "action": "Resolve original source conflict."})
    blockers = []
    if state.get("timing", {}).get("earliest_public_verified") is not True:
        blockers.append("earliest_public_time_unverified")
    if state.get("operative_history_complete") is not True:
        blockers.append("operative_history_incomplete")
    if state.get("market_expectations") == "unknown":
        blockers.append("market_expectations_unknown")
    for issuer in state.get("dated_issuer_exposures", []):
        if issuer.get("magnitude", {}).get("earnings_sensitivity") is None:
            blockers.append(f"{issuer['symbol']}_event_earnings_unquantified")
    return {"schema_version": "transition-guard-v1", "request_sha256": digest(request),
            "response_sha256": digest(response), "expected_labels_used": False,
            "status": "needs_review" if findings or blockers else "consistency_only",
            "trade_eligible": False, "alpha_proven": False,
            "deterministic_preliminary_direction": expected_direction,
            "interpretation_findings": findings, "economic_timing_blockers": blockers,
            "raw_model_response": raw}


def _expected_labels(expected: dict, request: dict) -> None:
    if not isinstance(expected, dict) or not expected:
        raise ValueError("At least one frozen development label is required")
    for name, value in expected.items():
        question = request["questions"].get(name)
        if (not question or question.get("type") != "choice" or
                not isinstance(value, str) or value not in question["criteria"]):
            raise ValueError("Expected label outside a supplied choice question")


def prepare_transition(spec: dict, packets: dict[str, dict], pack: dict) -> dict:
    """Freeze a targeted development case. No network, labels or returns in state."""
    if spec.get("schema_version") != "transition-spec-v1":
        raise ValueError("Unsupported transition specification")
    cutoff = date.fromisoformat(spec["event_channel_date"])
    documents, source_lookup, ids = [], {}, set()
    for selection in spec["documents"]:
        doc_id = selection["document_id"]
        if doc_id in ids:
            raise ValueError("Duplicate source document")
        ids.add(doc_id)
        packet = packets[doc_id]
        manifest = packet["episode_manifest"]
        if manifest["document_id"] != doc_id:
            raise ValueError("Document identity mismatch")
        availability = copy.deepcopy(manifest["content_availability"])
        published = availability["candidate_publication_date"]
        published_date = date.fromisoformat(published)
        roles = selection.get("roles", [selection.get("role")])
        if (not isinstance(roles, list) or not roles or len(roles) != len(set(roles)) or
                not set(roles) <= {"current", "preliminary", "earlier_operative_deposit"}):
            raise ValueError("Unknown temporal role")
        if "current" in roles and len(roles) != 1:
            raise ValueError("Current source cannot also be a prior")
        timestamp = availability.get("candidate_timestamp")
        candidate_stamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00")) if timestamp else None
        if candidate_stamp is not None and candidate_stamp.utcoffset() is None:
            raise ValueError("Candidate channel timestamps require an explicit timezone")
        candidate_day = candidate_stamp.date() if candidate_stamp else None
        if candidate_day is not None and candidate_day > cutoff:
            raise ValueError("Source candidate channel is after event channel date")
        if "current" not in roles and published_date >= cutoff:
            raise ValueError("Prior source not available strictly before event channel date")
        if "current" in roles and published_date > cutoff and candidate_day is None:
            raise ValueError("Later-published current source requires an earlier candidate channel timestamp")
        availability["content_version_matches_timestamp"] = availability.get("content_version_matches_timestamp") is True
        availability["earliest_public_verified"] = availability.get("earliest_public_verified") is True
        availability["timing_validation_scope"] = "Calendar-date candidate only; neither exact content availability nor intraday cutoff is established by this check."
        if selection["source_sha256"] != manifest["source_sha256"]:
            raise ValueError("Source changed since transcription")
        source = {p["passage_id"]: p for p in packet["current_source_passages_with_ids"]}
        if len(source) != len(packet["current_source_passages_with_ids"]):
            raise ValueError("Duplicate source passage IDs")
        wanted = selection["passage_ids"]
        if not wanted or len(wanted) != len(set(wanted)) or not set(wanted) <= source.keys():
            raise ValueError("Selected evidence IDs missing, repeated or empty")
        passages = [copy.deepcopy(p) for p in packet["current_source_passages_with_ids"] if p["passage_id"] in wanted]
        source_lookup[doc_id] = {p["passage_id"] for p in passages}
        documents.append({"document_id": doc_id, "roles": list(roles), "publication_date": published,
                          "source_sha256": manifest["source_sha256"], "source_url": manifest["source_url"],
                          "content_availability": availability,
                          "passages": passages})
    if sum("current" in d["roles"] for d in documents) != 1:
        raise ValueError("Exactly one current document required")
    for row in spec["rate_candidates"]:
        for role, observation in (("current", row["current"]),
                                   ("preliminary", row.get("preliminary")),
                                   ("earlier_operative_deposit", row.get("earlier_operative_deposit"))):
            if observation is None:
                continue
            doc_id = observation["document_id"]
            anchors = observation["passage_ids"]
            if not anchors or not set(anchors) <= source_lookup.get(doc_id, set()):
                raise ValueError("Rate citation absent from supplied evidence")
            if not any(d["document_id"] == doc_id and role in d["roles"] for d in documents):
                raise ValueError("Rate cited against wrong temporal source role")
            if role == "earlier_operative_deposit" and observation.get("legal_effect") not in {"final_deposit", "provisional_deposit"}:
                raise ValueError("Earlier operative observation requires an explicit final or provisional deposit effect")
    # A date-only issuer source is admissible the next day, never at midnight of
    # its release date. Precision and incomplete historical coverage stay explicit.
    exposure = copy.deepcopy(spec.get("issuer_exposures", []))
    for item in exposure:
        if date.fromisoformat(item["source_available_date"]) >= cutoff or not item.get("source_url"):
            raise ValueError("Issuer evidence is missing provenance or not strictly prior")
    state = {"episode_id": spec["episode_id"], "documents": documents,
             "rate_candidates": copy.deepcopy(spec["rate_candidates"]),
             "dated_issuer_exposures": exposure,
             "timing": copy.deepcopy(spec["timing"]),
             "event_channel_date": spec["event_channel_date"],
             "market_expectations": "unknown", "operative_history_complete": False,
             "selection": "Targeted agent-reviewed development case chosen before prices; manually selected evidence, not model discovery."}
    # Lossless scoped ID compression avoids repeating the same 16-character
    # hash hundreds of times. Reconstruct with document.passage_id_prefix.
    id_maps = {}
    for document in state["documents"]:
        prefix = document["source_sha256"][:16] + ":"
        ids_for_doc = [p["passage_id"] for p in document["passages"]]
        if all(pid.startswith(prefix) for pid in ids_for_doc):
            document["passage_id_prefix"] = prefix
            mapping = {pid: pid[len(prefix):] for pid in ids_for_doc}
            id_maps[document["document_id"]] = mapping
            for passage in document["passages"]:
                passage["passage_id"] = mapping[passage["passage_id"]]
    for row in state["rate_candidates"]:
        for role in ("current", "preliminary", "earlier_operative_deposit"):
            observation = row.get(role)
            if observation and observation["document_id"] in id_maps:
                observation["passage_ids"] = [id_maps[observation["document_id"]][pid] for pid in observation["passage_ids"]]
    request = build_request(state, pack)
    estimate = estimate_request(request)
    if estimate["conservative_input_tokens"] > 28000:
        raise ValueError("Transition request exceeds conservative context guard")
    expected = spec["development_expected"]
    _expected_labels(expected, request)
    comparisons = numeric_comparisons(spec["rate_candidates"])
    return {"schema_version": "transition-experiment-v1", "created_at": utc_now(),
            "spec_sha256": digest(spec), "pack_sha256": digest(pack),
            "model": request["model"],
            "episode_id": spec["episode_id"], "request": request, "request_sha256": digest(request),
            "deterministic_rate_comparisons": comparisons,
            "deterministic_rate_comparisons_sha256": digest(comparisons),
            "expected": copy.deepcopy(expected), "expected_sha256": digest(expected), "label_origin": "agent_reviewed",
            "split": "development", "prices_used": False,
            "estimate": estimate, "trade_eligible": False}


def run_transition(store: Store, manifest: dict, out: Path, *, live: bool = False,
                   baseline: bool = False, budget_usd: float = 1) -> dict:
    if out.exists():
        raise ValueError("Output already exists; preserve observations")
    request = manifest["request"]
    if (manifest.get("schema_version") != "transition-experiment-v1" or
            digest(request) != manifest["request_sha256"] or manifest.get("model") != request["model"]):
        raise ValueError("Changed or invalid frozen request")
    estimate = estimate_request(request)  # Also revalidates the pinned model and question schema.
    if estimate["conservative_input_tokens"] > 28000:
        raise ValueError("Transition request exceeds conservative context guard")
    _expected_labels(manifest["expected"], request)
    if digest(manifest["expected"]) != manifest.get("expected_sha256"):
        raise ValueError("Changed frozen development labels")
    state = request["state"]
    evidence = state.get("evidence", state)
    comparisons = numeric_comparisons(evidence["rate_candidates"])
    if (comparisons != manifest["deterministic_rate_comparisons"] or
            digest(comparisons) != manifest.get("deterministic_rate_comparisons_sha256")):
        raise ValueError("Changed or inconsistent deterministic comparisons")
    if evidence["episode_id"] != manifest["episode_id"]:
        raise ValueError("Changed episode identity")
    arms = ["jev"] + (["baseline"] if baseline else [])
    label_count = len(manifest["expected"])
    report = {"schema_version": "transition-result-v1", "created_at": utc_now(),
              "manifest_sha256": digest(manifest), "episode_id": manifest["episode_id"],
              "prices_used": False, "trade_eligible": False, "alpha_proven": False,
              "label_origin": "agent_reviewed", "split": "development",
              "deterministic_rate_comparisons": comparisons,
              "requested_arms": arms, "requested_label_count": label_count * len(arms),
              "runs": [], "status": "dry_run" if not live else "complete"}
    halted, unexpected = False, None
    for arm in arms:
        labels = [{"question_id": name, "expected": expected, "chosen": None,
                   "matched": False, "answered": False}
                  for name, expected in manifest["expected"].items()]
        entry = {"arm": arm, "status": "skipped_after_failure" if halted else "dry_run",
                 "labels": labels, "matched": 0, "answered": 0, "total": label_count}
        if live and not halted:
            try:
                run = run_one(store, request, arm=arm, transport="curl", phase_budget_usd=budget_usd,
                              max_cost_usd=0.02 if arm == "jev" else 0.1)
                entry["run"] = run  # Preserve returned evidence even if local scoring fails.
                entry["guard"] = guard_transition(manifest, run["response"])
                for label in labels:
                    chosen = run["response"]["answers"][label["question_id"]]["choice"]
                    label.update(chosen=chosen, answered=True, matched=chosen == label["expected"])
                entry.update(status="completed", matched=sum(c["matched"] for c in labels), answered=len(labels))
            except Exception as exc:
                entry.update(status="failed", error_type=type(exc).__name__,
                             diagnostic_sha256=getattr(exc, "response_blob_sha256", None))
                report["status"] = "incomplete"
                halted = True
                if not isinstance(exc, (JevRequestError, ValueError, OSError)):
                    unexpected = exc
        report["runs"].append(entry)
    report["answered_label_count"] = sum(r["answered"] for r in report["runs"])
    report["matched_label_count"] = sum(r["matched"] for r in report["runs"])
    report["label_accuracy_including_missing"] = report["matched_label_count"] / report["requested_label_count"] if live else None
    write_new_json(out, report)
    if unexpected is not None:
        raise unexpected  # Programming failures remain visible after preserving the report.
    return report
