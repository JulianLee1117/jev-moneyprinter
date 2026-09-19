"""Price-blind selection and evidence packets. Heuristics never certify alpha."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict, deque
from datetime import date, datetime
from html.parser import HTMLParser

from .store import Store, canonical_json, utc_now

CORE_TITLE = re.compile(r"corrosion[ -]resistant", re.I)
SCREENING_QUESTIONS = ["document_authority", "decision_finality", "comparison_validity",
                       "novelty_in_supplied_history", "prior_state_incomplete", "exception_limits_headline"]


def stage_hint(title: str) -> str:
    text = title.lower()
    for token, label in (("circumvention", "circumvention"), ("scope", "scope"),
                         ("sunset", "sunset"), ("five-year", "sunset"),
                         ("amended", "amended"), ("rescission", "rescission"),
                         ("final", "final"), ("preliminary", "preliminary"),
                         ("initiation", "initiation"), ("order", "order")):
        if token in text:
            return label
    return "other"


def census(documents: list[dict]) -> dict:
    matches = [d for d in documents if CORE_TITLE.search(d.get("title", ""))]
    return {"documents": len(documents), "core_title_matches": len(matches),
            "title_matches_by_year": dict(sorted(Counter(d.get("publication_date", "unknown")[:4] for d in matches).items())),
            "title_stage_hints": dict(sorted(Counter(stage_hint(d.get("title", "")) for d in matches).items())),
            "independent_event_count": None,
            "limitations": ["Title matching is not exhaustive: omnibus notices may contain relevant information.",
                            "Stage hints are unvalidated title heuristics.",
                            "Documents and country rows are not independent events."]}


def select_pilot(documents: list[dict], count: int = 20, seed: str = "core-pilot-v1") -> dict:
    if not 1 <= count <= 200:
        raise ValueError("Pilot count must be between 1 and 200")
    candidates = [d for d in documents if CORE_TITLE.search(d.get("title", ""))]
    if not candidates:
        raise ValueError("No CORE title candidates; import or discover records first")
    ids = [d["document_number"] for d in candidates]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate document IDs in selection universe")
    rank = lambda d: hashlib.sha256((seed + ":" + d["document_number"]).encode()).hexdigest()
    buckets = defaultdict(lambda: defaultdict(list))
    for doc in candidates:
        date.fromisoformat(doc["publication_date"])
        buckets[stage_hint(doc["title"])][doc["publication_date"][:4]].append(doc)
    stages = {}
    for stage, years in sorted(buckets.items()):
        # Cover recent and older cycles before filling the middle; chronological
        # oldest-first traversal can exhaust a small pilot before recent years.
        year_names = deque(sorted(years))
        ordered_years = []
        while year_names:
            ordered_years.append(year_names.pop())
            if year_names:
                ordered_years.append(year_names.popleft())
        queues = [deque(sorted(years[year], key=rank)) for year in ordered_years]
        ordered = []
        while any(queues):
            for queue in queues:
                if queue:
                    ordered.append(queue.popleft())
        stages[stage] = deque(ordered)
    selected = []
    while len(selected) < min(count, len(candidates)):
        for queue in stages.values():
            if queue and len(selected) < count:
                selected.append(queue.popleft())
    selected.sort(key=lambda d: (d["publication_date"], d["document_number"]))
    return {"schema_version": "selection-v2", "created_at": utc_now(), "family": "CORE",
            "seed": seed, "requested_count": count, "selected_count": len(selected),
            "universe_count": len(candidates),
            "universe_sha256": hashlib.sha256(canonical_json(sorted(candidates, key=lambda d: d["document_number"]))).hexdigest(),
            "selection_method": "Price-blind stage round robin; within stage, year round robin alternating newest/oldest; SHA256 seed/document ordering within year. Title heuristics only.",
            "price_data_used": False,
            "interpretation": "A rejection-test sample, not a representative return sample or frozen trading rule.",
            "selected": [{**d, "stage_hint": stage_hint(d["title"])} for d in selected]}


class _PlainText(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self.hidden = 0

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style"}:
            self.hidden += 1
        if tag in {"p", "div", "br", "tr", "h1", "h2", "h3", "li"}:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in {"script", "style"}:
            self.hidden = max(0, self.hidden - 1)
        if tag in {"p", "div", "tr", "h1", "h2", "h3", "li"}:
            self.parts.append("\n")
        elif tag in {"td", "th"}:
            self.parts.append(" | ")

    def handle_data(self, data):
        if not self.hidden:
            self.parts.append(data)


def text_passages(raw: bytes, digest: str) -> list[dict]:
    text = raw.decode("utf-8-sig")
    if re.search(r"<(?:html|body|p|div|table)\b", text, re.I):
        parser = _PlainText()
        parser.feed(text)
        text = "".join(parser.parts)
    if len(text.strip()) < 100 or "request access" in text[:500].lower():
        raise ValueError("Source text is missing or appears to be an access/error page")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines() if line.strip()]
    return [{"passage_id": f"{digest[:16]}:p{i:04d}", "text": line} for i, line in enumerate(lines, 1)]


def source_observations(store: Store, doc_id: str) -> dict:
    pi = store.latest_source(doc_id, "public_inspection")
    result = {"public_inspection_candidate": None, "earliest_disclosure_verified": False}
    if pi:
        payload = json.loads(pi["content"])
        if not isinstance(payload, dict) or payload.get("document_number") != doc_id:
            raise ValueError("Archived public-inspection metadata has a mismatched document identity")
        filed_at = payload.get("filed_at")
        try:
            stamp = datetime.fromisoformat(filed_at.replace("Z", "+00:00"))
            if stamp.utcoffset() is None:
                raise ValueError
        except (AttributeError, TypeError, ValueError):
            raise ValueError("Archived public-inspection metadata has no valid timezone-aware filing time") from None
        result["public_inspection_candidate"] = {"filed_at": payload.get("filed_at"),
            "source_url": pi["source_url"], "source_sha256": pi["sha256"],
            "retrieved_at": pi["observed_at"], "original_metadata_captured_at": pi["original_captured_at"],
            "caveat": "One public channel; earlier Commerce/USITC/issuer disclosures require review."}
    return result


def evidence_packet(store: Store, doc_id: str, *, prior_ids: list[str] | None = None,
                    passage_ids: list[str] | None = None) -> dict:
    document = store.get_document(doc_id)
    if not document:
        raise ValueError("Document is not in the archive")
    current_date = date.fromisoformat(document["publication_date"])
    observations = source_observations(store, doc_id)
    pi_candidate = observations["public_inspection_candidate"]

    def archived(doc: str):
        source = store.latest_source(doc, "public_inspection_text") or store.latest_source(doc, "published_text")
        if not source:
            raise ValueError(f"No archived text for {doc}; collect it first")
        return source, text_passages(source["content"], source["sha256"])

    source, passages = archived(doc_id)
    content_availability = {"source_kind": source["kind"], "candidate_timestamp": None,
        "candidate_publication_date": document["publication_date"],
        "content_version_matches_timestamp": False,
        "caveat": "Text retrieved now; matching this content version to historical availability requires review."}
    if source["kind"] == "public_inspection_text" and pi_candidate:
        content_availability["candidate_timestamp"] = pi_candidate["filed_at"]
        current_date = datetime.fromisoformat(pi_candidate["filed_at"].replace("Z", "+00:00")).date()
    original_count = len(passages)
    if passage_ids is not None:
        wanted = set(passage_ids)
        if not wanted or not wanted <= {p["passage_id"] for p in passages}:
            raise ValueError("Passage selection contains unknown IDs or is empty")
        passages = [p for p in passages if p["passage_id"] in wanted]
    prior = []
    for prior_id in prior_ids or []:
        prior_document = store.get_document(prior_id)
        if not prior_document or date.fromisoformat(prior_document["publication_date"]) >= current_date:
            raise ValueError("Prior documents must have a publication date strictly before the current channel's availability date")
        previous_source, previous_passages = archived(prior_id)
        prior.append({"document": prior_document, "source_sha256": previous_source["sha256"], "passages": previous_passages})
    return {"current_source_passages_with_ids": passages,
            "prior_public_case_state_with_evidence": {"completeness": "unknown", "documents": prior},
            "validated_rates_and_dates": {"rates": [], "publication_date": document["publication_date"], "timing": observations},
            "dated_candidate_issuer_exposures": [], "earlier_public_source_evidence": [],
            "episode_manifest": {"document_id": doc_id, "title": document["title"], "episode_id": None,
                "source_url": source["source_url"], "source_sha256": source["sha256"],
                "archived_at": source["observed_at"], "original_passage_count": original_count,
                "content_availability": content_availability,
                "selected_passage_count": len(passages), "passage_selection": "all" if passage_ids is None else "explicit_manual_selection",
                "review_status": "unreviewed", "market_expectations": "unknown", "historical_hindsight_not_eliminated": True}}
