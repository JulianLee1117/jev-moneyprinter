"""Evidence gates for a price-data pilot, never a profitability prediction.

The input is a list of manually adjudicated dossiers. Titles, model confidence,
document counts and product relevance cannot make a dossier eligible. A human
must cite the material change, dated issuer exposure, economic magnitude and
timing. ``market_expectations`` remains separate: unknown expectations permit a
pilot of material changes, but cannot support a claim of market surprise.

Citation shape: {"source_url": "https://...", "excerpt": "supporting passage"}.
``rationale`` explains the adjudication rather than substituting for a citation.
An unsupported gate with a citation is a documented exclusion. Unknown, missing
or malformed data blocks readiness. Multiple issuers and documents belonging to
one episode contribute at most one candidate episode.
"""

from __future__ import annotations

from collections import Counter
from datetime import date, datetime
import re
from typing import Any
from urllib.parse import urlparse


SCHEMA_VERSION = "profit-audit-v1"
_STATUSES = {"supported", "unsupported", "unknown"}
_REQUIRED_GATES = (
    "material_change",
    "issuer_exposure",
    "economic_magnitude",
    "earliest_availability",
)
_TIMESTAMP = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})$"
)


def _text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str) or not _TIMESTAMP.fullmatch(value):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return parsed if parsed.utcoffset() is not None else None
    except ValueError:
        return None


def _date(value: Any) -> date | None:
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def _evidence_valid(value: Any) -> bool:
    if not isinstance(value, list) or not value:
        return False
    for citation in value:
        if not isinstance(citation, dict) or not _text(citation.get("excerpt")):
            return False
        url = citation.get("source_url")
        if not isinstance(url, str):
            return False
        try:
            parsed = urlparse(url)
            if parsed.scheme not in {"http", "https"} or not parsed.hostname:
                return False
        except ValueError:
            return False
    return True


def dossier_template(document: dict[str, Any]) -> dict[str, Any]:
    """Create an explicitly unadjudicated dossier from discovery metadata.

    Accept either Federal Register ``document_number`` or ``document_id``.
    Metadata is copied through an allowlist; discovery data never supplies an
    episode assignment, evidence, review, economic judgment or first-public time.
    Missing IDs remain null for a reviewer to resolve and block the audit.
    """
    if not isinstance(document, dict):
        raise TypeError("document must be an object")
    document_id = document.get("document_id") or document.get("document_number")
    if not _text(document_id):
        document_id = None

    def unknown() -> dict[str, Any]:
        return {"status": "unknown", "rationale": "", "evidence": []}

    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "document_id": document_id,
        "episode_id": None,
        "document": {
            key: document[key]
            for key in ("title", "html_url", "publication_date", "document_number")
            if key in document
        },
        "review": {"status": "pending", "reviewer": None, "reviewed_at": None},
        "material_change": unknown(),
        "issuer_exposure": {**unknown(), "issuer_ids": [], "as_of": None},
        "economic_magnitude": unknown(),
        "earliest_availability": {
            **unknown(),
            "timestamp": None,
            "earlier_sources_checked": False,
            "content_version_matches_timestamp": False,
        },
        "market_expectations": unknown(),
        "uncertainties": [
            "Discovery metadata is not an adjudicated material event.",
            "Publication or public-inspection time may follow an earlier disclosure.",
            "Product relevance does not establish economically meaningful exposure.",
            "Unknown market expectations cannot support a surprise claim.",
        ],
    }
    return result


def evaluate_audit(
    dossiers: list[dict[str, Any]], *, min_candidate_episodes: int = 3
) -> dict[str, Any]:
    """Assess whether this selected cohort is ready for a small price pilot.

    Three independent *assigned* episodes is only a default feasibility hurdle;
    neither grouping correctness nor statistical independence is proven by IDs.
    Every selected dossier must be resolved, and a documented exclusion is a
    resolution. Ready dossiers sharing an episode count once. The function
    returns structured blockers for malformed data, including a non-list root.
    Invalid configuration raises ValueError rather than silently relaxing a gate.
    """
    if type(min_candidate_episodes) is not int or min_candidate_episodes < 1:
        raise ValueError("min_candidate_episodes must be a positive integer")

    blockers: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    result: dict[str, Any] = {
        "schema_version": "profit-audit-result-v1",
        "readiness": "needs_review",
        "alpha_proven": False,
        "min_candidate_episodes": min_candidate_episodes,
        "interpretation": (
            "Readiness only for an exploratory price-data pilot. Assigned episodes "
            "are not proof of independence, sufficient sample size or profitable alpha."
        ),
        "market_surprise_claim_permitted": False,
        "counts": {
            "input_dossiers": len(dossiers) if isinstance(dossiers, list) else 0,
            "unique_documents": 0,
            "unique_episodes": 0,
            "ready_dossiers": 0,
            "excluded_dossiers": 0,
            "needs_review_dossiers": 0,
            "malformed_dossiers": 0,
            "ready_episodes": 0,
            "rejected_episodes": 0,
            "needs_review_episodes": 0,
            "expectations_unknown_dossiers": 0,
        },
        "eligible_episode_ids": [],
        "episode_results": [],
        "dossier_results": records,
        "blockers": blockers,
    }

    def add_blocker(index: int | None, document_id: Any, episode_id: Any,
                    field: str, reason: str) -> None:
        blockers.append({
            "dossier_index": index,
            "document_id": document_id if _text(document_id) else None,
            "episode_id": episode_id if _text(episode_id) else None,
            "field": field,
            "reason": reason,
        })

    if not isinstance(dossiers, list):
        add_blocker(None, None, None, "dossiers", "Expected a list of dossier objects.")
        return result
    if not dossiers:
        add_blocker(None, None, None, "dossiers", "No adjudicated candidates supplied.")
        return result

    document_ids = Counter(
        row["document_id"] for row in dossiers
        if isinstance(row, dict) and _text(row.get("document_id"))
    )
    result["counts"]["unique_documents"] = len(document_ids)

    for index, row in enumerate(dossiers):
        start = len(blockers)
        malformed = False
        exclusions: list[str] = []
        document_id = row.get("document_id") if isinstance(row, dict) else None
        episode_id = row.get("episode_id") if isinstance(row, dict) else None

        def block(field: str, reason: str, *, invalid: bool = False) -> None:
            nonlocal malformed
            malformed = malformed or invalid
            add_blocker(index, document_id, episode_id, field, reason)

        if not isinstance(row, dict):
            block("dossier", "Expected a dossier object.", invalid=True)
            row = {}
        if row.get("schema_version") != SCHEMA_VERSION:
            block("schema_version", "Missing or unsupported dossier schema.", invalid=True)
        for key, value in (("document_id", document_id), ("episode_id", episode_id)):
            if not _text(value) or value != value.strip():
                block(key, "A nonempty, stable, whitespace-trimmed identifier is required.", invalid=True)
        if _text(document_id) and document_ids[document_id] > 1:
            block("document_id", "Duplicate document ID; resolve the duplicate before counting.", invalid=True)
        review = row.get("review")
        if not isinstance(review, dict) or (
            review.get("status") != "complete"
            or not _text(review.get("reviewer"))
            or _timestamp(review.get("reviewed_at")) is None
        ):
            block("review", "A completed named review with a timezone-aware timestamp is required.")

        # Inspect structural errors in every gate before evaluating exclusions.
        statuses: dict[str, str] = {}
        for name in (*_REQUIRED_GATES, "market_expectations"):
            gate = row.get(name)
            if not isinstance(gate, dict):
                block(name, "Missing or malformed adjudication block.", invalid=True)
                continue
            status = gate.get("status")
            if not isinstance(status, str) or status not in _STATUSES:
                block(f"{name}.status", "Expected supported, unsupported or unknown.", invalid=True)
                continue
            statuses[name] = status
            if status != "unknown":
                if not _text(gate.get("rationale")):
                    block(f"{name}.rationale", "An explicit adjudication rationale is required.")
                if not _evidence_valid(gate.get("evidence")):
                    block(f"{name}.evidence", "Cite at least one source URL and supporting excerpt.")
            if status == "unsupported" and name in _REQUIRED_GATES:
                exclusions.append(name)

        expectations = statuses.get("market_expectations", "unknown")
        if expectations != "supported":
            result["counts"]["expectations_unknown_dossiers"] += 1

        # A sourced negative finding resolves a candidate without requiring all
        # unrelated downstream work. Structural errors and unsupported assertions
        # without evidence still prevent a documented exclusion.
        if not exclusions:
            for name in _REQUIRED_GATES:
                if statuses.get(name) == "unknown":
                    block(name, "Unknown; this required economic/timing gate is unresolved.")
            exposure = row.get("issuer_exposure")
            availability = row.get("earliest_availability")
            exposure_date = None
            available_at = None
            if isinstance(exposure, dict) and exposure.get("status") == "supported":
                issuer_ids = exposure.get("issuer_ids")
                if not isinstance(issuer_ids, list) or not issuer_ids or not all(
                    _text(issuer_id) for issuer_id in issuer_ids
                ):
                    block("issuer_exposure.issuer_ids", "At least one supported issuer ID is required.")
                exposure_date = _date(exposure.get("as_of"))
                if exposure_date is None:
                    block("issuer_exposure.as_of", "Exposure must be dated as YYYY-MM-DD.")
            if isinstance(availability, dict) and availability.get("status") == "supported":
                available_at = _timestamp(availability.get("timestamp"))
                if available_at is None:
                    block("earliest_availability.timestamp", "An exact ISO timestamp with timezone is required.")
                if availability.get("earlier_sources_checked") is not True:
                    block("earliest_availability.earlier_sources_checked", "Earlier public channels must be checked explicitly.")
                if availability.get("content_version_matches_timestamp") is not True:
                    block("earliest_availability.content_version_matches_timestamp", (
                        "Verify with cited evidence that the analyzed content version was available "
                        "at the proposed timestamp; later published or revised text cannot inherit "
                        "an earlier public-inspection time."
                    ))
            if exposure_date is not None and available_at is not None:
                if exposure_date > available_at.date():
                    block("issuer_exposure.as_of", "Exposure dated after the event would introduce hindsight.")

        status = "needs_review" if len(blockers) > start else (
            "reject" if exclusions else "ready_for_price_pilot"
        )
        records.append({
            "dossier_index": index,
            "document_id": document_id if _text(document_id) else None,
            "episode_id": episode_id if _text(episode_id) else None,
            "readiness": status,
            "exclusion_gates": exclusions if status == "reject" else [],
            "market_expectations": expectations,
            "blockers": blockers[start:].copy(),
        })
        count_key = {
            "ready_for_price_pilot": "ready_dossiers",
            "reject": "excluded_dossiers",
            "needs_review": "needs_review_dossiers",
        }[status]
        result["counts"][count_key] += 1
        result["counts"]["malformed_dossiers"] += int(malformed)

    episodes: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        if _text(record["episode_id"]):
            episodes.setdefault(record["episode_id"], []).append(record)
    result["counts"]["unique_episodes"] = len(episodes)
    for episode_id, members in sorted(episodes.items()):
        states = {member["readiness"] for member in members}
        status = "needs_review" if "needs_review" in states else (
            "ready_for_price_pilot" if "ready_for_price_pilot" in states else "reject"
        )
        episode = {
            "episode_id": episode_id,
            "readiness": status,
            "document_ids": sorted({member["document_id"] for member in members if member["document_id"]}),
            "eligible_document_ids": sorted({
                member["document_id"] for member in members
                if member["readiness"] == "ready_for_price_pilot"
            }),
        }
        result["episode_results"].append(episode)
        count_key = {
            "ready_for_price_pilot": "ready_episodes",
            "reject": "rejected_episodes",
            "needs_review": "needs_review_episodes",
        }[status]
        result["counts"][count_key] += 1
        if status == "ready_for_price_pilot":
            result["eligible_episode_ids"].append(episode_id)

    if result["counts"]["needs_review_dossiers"]:
        result["readiness"] = "needs_review"
    elif result["counts"]["ready_episodes"] >= min_candidate_episodes:
        result["readiness"] = "ready_for_price_pilot"
    elif result["counts"]["excluded_dossiers"] == len(dossiers):
        result["readiness"] = "reject"
    else:
        add_blocker(None, None, None, "candidate_episodes", (
            f"Only {result['counts']['ready_episodes']} resolved candidate episodes; "
            f"the exploratory pilot minimum is {min_candidate_episodes}. "
            "This threshold is not a statistical significance requirement."
        ))
    # This audit does not compare actual prices to expectations. Even cited
    # expectations are insufficient to claim a measured market surprise.
    return result
