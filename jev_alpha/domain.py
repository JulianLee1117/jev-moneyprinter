"""Deterministic interpretation consistency checks, never a trading gate.

Missing evidence is not evidence of absence. The guard preserves the model's raw
answer and every probability; it adds review findings rather than relabeling an
output after seeing it. Review metadata is an explicit attestation, not proof
that the review was correct or historical hindsight has been eliminated.
"""

from __future__ import annotations

from datetime import date, datetime
from urllib.parse import urlsplit

from .jev import JevValidationError, validate_response


def _timestamp(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return result if result.utcoffset() is not None else None
    except ValueError:
        return None


def _reviewed(value: object) -> bool:
    return (isinstance(value, dict) and value.get("status") == "complete"
            and isinstance(value.get("reviewer"), str) and bool(value["reviewer"].strip())
            and _timestamp(value.get("reviewed_at")) is not None)


def _cited(value: object) -> bool:
    if not isinstance(value, list) or not value:
        return False
    for item in value:
        if (not isinstance(item, dict) or not isinstance(item.get("excerpt"), str)
                or not item["excerpt"].strip() or not isinstance(item.get("source_url"), str)):
            return False
        try:
            url = urlsplit(item["source_url"])
            if url.scheme not in {"http", "https"} or not url.hostname:
                return False
        except ValueError:
            return False
    return True


def guard_panel(request: dict, response: dict) -> dict:
    """Preserve a validated raw panel and flag unsupported selected labels.

    For reviewed exposure metadata, an entry identifies ``issuer_id`` or
    ``ticker``; has ``as_of`` (date), ``available_at`` (aware timestamp), cited
    ``evidence`` and ``review`` {status: complete, reviewer, reviewed_at}.
    ``available_at`` must precede the supplied event timestamp and ``as_of`` may
    not be later than its event date. Populating an exposure list alone cannot
    establish reviewed or historically available coverage.

    Prior comparisons require completeness=complete, nonempty documents and a
    completed review in prior_public_case_state_with_evidence. Current-source
    review uses episode_manifest.review. This small guard cannot verify the
    attestations, source recall, market expectations or trading profitability.
    """
    raw = validate_response(response, request)
    state = request["state"]
    if isinstance(state, dict) and "evidence" in state:
        state = state["evidence"]
    if not isinstance(state, dict):
        raise JevValidationError("Panel consistency guard requires an object evidence state.")
    manifest = state.get("episode_manifest")
    manifest = manifest if isinstance(manifest, dict) else {}
    answers = raw["answers"]
    blockers, contradictions, unresolved = [], [], []

    def block(code: str, field: str, reason: str, **details: object) -> None:
        finding = {"code": code, "field": field, "reason": reason, **details}
        blockers.append(finding)
        if field not in unresolved:
            unresolved.append(field)

    def conflict(name: str, code: str, reason: str) -> None:
        answer = answers.get(name, {})
        if answer.get("type") == "choice":
            contradictions.append({"code": code, "question_id": name,
                "chosen_label": answer["choice"], "reason": reason,
                "resolution": "Review evidence and interpretation; raw output was not changed."})

    passages = state.get("current_source_passages_with_ids")
    if not isinstance(passages, list) or not passages:
        block("current_source_missing", "current_source_passages_with_ids",
              "No supplied current-source passages support the interpretation.")
    if not _reviewed(manifest.get("review")):
        block("source_review_incomplete", "episode_manifest.review",
              "A completed, attributed review of the supplied source evidence is missing.")

    prior_questions = {"comparison_validity", "novelty_in_supplied_history",
                       "product_scope_change", "exporter_scope_change", "country_scope_change"}
    if prior_questions & answers.keys():
        prior = state.get("prior_public_case_state_with_evidence")
        prior = prior if isinstance(prior, dict) else {}
        prior_ready = (prior.get("completeness") == "complete"
                       and isinstance(prior.get("documents"), list) and bool(prior["documents"])
                       and _reviewed(prior.get("review")))
        if not prior_ready:
            block("prior_state_not_reviewed_complete", "prior_public_case_state_with_evidence",
                  "Prior coverage is missing, incomplete or not reviewed; supplied-history novelty is unresolved.")
            if answers.get("comparison_validity", {}).get("choice") in {"matched", "partial"}:
                conflict("comparison_validity", "unsupported_prior_comparison",
                         "A matched/partial comparison is asserted without reviewed complete prior evidence.")
            for name in sorted(prior_questions - {"comparison_validity"}):
                if answers.get(name, {}).get("choice") not in (None, "unknown"):
                    conflict(name, "unsupported_prior_change_interpretation",
                             "The selected change/confirmation label requires prior coverage that is unresolved.")

    suffixes = ("_exposure_support", "_offsetting_channel", "_producer_channel")
    issuer_questions = {}
    for name in answers:
        for suffix in suffixes:
            if name.endswith(suffix):
                issuer_questions.setdefault(name[:-len(suffix)].upper(), []).append(name)
                break
    exposures = state.get("dated_candidate_issuer_exposures")
    exposures = exposures if isinstance(exposures, list) else []
    availability = manifest.get("content_availability")
    availability = availability if isinstance(availability, dict) else {}
    event_time = _timestamp(availability.get("candidate_timestamp"))
    reviewed_issuers = []
    for issuer, names in sorted(issuer_questions.items()):
        records = [record for record in exposures if isinstance(record, dict)
                   and isinstance(record.get("issuer_id", record.get("ticker")), str)
                   and record.get("issuer_id", record.get("ticker")).upper() == issuer]
        field = "dated_candidate_issuer_exposures"
        if not records:
            block("issuer_exposure_missing", field,
                  "No issuer-specific exposure records were supplied; this does not establish no exposure or no offset.",
                  issuer_id=issuer)
            for name in names:
                if answers[name].get("choice") not in (None, "unknown"):
                    conflict(name, "unsupported_exposure_interpretation",
                             "Missing issuer evidence requires unknown; absence labels also require supporting coverage.")
            continue
        reviewed = all(_reviewed(record.get("review")) and _cited(record.get("evidence"))
                       for record in records)
        if not reviewed:
            block("issuer_exposure_unreviewed", field,
                  "Populated exposure records lack completed attributed reviews or source citations.", issuer_id=issuer)
        date_supported = event_time is not None
        for record in records:
            try:
                as_of = date.fromisoformat(record.get("as_of", ""))
            except (ValueError, TypeError):
                as_of = None
            available_at = _timestamp(record.get("available_at"))
            if (event_time is None or as_of is None or as_of > event_time.date()
                    or available_at is None or available_at > event_time):
                date_supported = False
        if not date_supported:
            block("issuer_exposure_time_unresolved", field,
                  "Exposure dates or public availability are missing, after the event, or cannot be compared to an event timestamp.",
                  issuer_id=issuer)
        if reviewed and date_supported:
            reviewed_issuers.append(issuer)

    if "implementation_status" in answers:
        facts = state.get("validated_rates_and_dates")
        if not isinstance(facts, dict) or not _reviewed(facts.get("review")):
            block("operative_dates_unreviewed", "validated_rates_and_dates",
                  "Operative timing lacks an attributed review of validated dates; model confidence cannot supply it.")
    if availability.get("content_version_matches_timestamp") is not True:
        block("source_version_time_unverified", "episode_manifest.content_availability",
              "The supplied content version is not verified against its historical availability timestamp.")

    return {"schema_version": "panel-consistency-v1",
            "scope": "interpretation_consistency_only",
            "document_id": manifest.get("document_id"),
            "source_sha256": manifest.get("source_sha256"),
            "status": "needs_review" if blockers or contradictions else "consistent_with_reviewed_inputs",
            "trade_eligible": False, "alpha_proven": False,
            "raw_model_response": raw, "raw_probabilities_modified": False,
            "blockers": blockers, "contradictions": contradictions,
            "unresolved_evidence": unresolved, "reviewed_issuer_ids": reviewed_issuers,
            "market_surprise_established": False,
            "historical_hindsight_eliminated": False,
            "interpretation": (
                "Consistency findings only. Review attestations are not independently verified. "
                "No earlier-disclosure completeness, calibrated probability, market surprise, "
                "economic magnitude, executable trade, or profitable alpha is established."
            )}
