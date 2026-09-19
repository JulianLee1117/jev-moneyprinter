"""Bounded, read-only Federal Register collection with preserved source bytes."""

from __future__ import annotations

import json
import time
from datetime import date, timedelta
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen

from .store import Store, validate_document_id

BASE = "https://www.federalregister.gov/api/v1"
ALLOWED_HOSTS = {"www.federalregister.gov", "federalregister.gov", "public-inspection.federalregister.gov", "www.govinfo.gov", "www.gpo.gov"}


class SourceError(RuntimeError):
    pass


class FederalRegister:
    def __init__(self, store: Store, *, timeout: float = 20, min_interval: float = 0.4):
        self.store = store
        self.timeout = timeout
        self.min_interval = min_interval
        self._last_request = 0.0

    def fetch(self, url: str, *, kind: str, doc_id: str | None = None, optional: bool = False) -> tuple[bytes, int] | None:
        parsed = urlparse(url)
        if parsed.scheme != "https" or parsed.hostname not in ALLOWED_HOSTS or parsed.username or parsed.password:
            raise SourceError("Source URL must be an HTTPS Federal Register or GPO resource")
        remaining = self.min_interval - (time.monotonic() - self._last_request)
        if remaining > 0:
            time.sleep(remaining)
        self._last_request = time.monotonic()
        request = Request(url, headers={"User-Agent": "jev-alpha-research/0.1 (bounded public-record research)", "Accept": "application/json,text/plain,text/html,*/*"})
        try:
            with urlopen(request, timeout=self.timeout) as response:
                content = response.read(10_000_001)
                status = response.status
        except HTTPError as exc:
            content = exc.read(200_000)
            self.store.observe(url, content, kind=kind, status=exc.code, document_id=doc_id)
            if optional and exc.code == 404:
                return None
            raise SourceError(f"Public source returned HTTP {exc.code}") from None
        except (URLError, TimeoutError, OSError):
            raise SourceError("Public source request failed or timed out; partial archive is preserved") from None
        if len(content) > 10_000_000:
            raise SourceError("Source exceeds 10 MB size limit")
        observation = self.store.observe(url, content, kind=kind, status=status, document_id=doc_id)
        return content, observation

    def fetch_json(self, url: str, *, kind: str, doc_id: str | None = None, optional: bool = False):
        result = self.fetch(url, kind=kind, doc_id=doc_id, optional=optional)
        if result is None:
            return None
        raw, observation = result
        try:
            payload = json.loads(raw)
        except (ValueError, UnicodeError):
            raise SourceError("Source returned invalid JSON; raw response archived") from None
        if not isinstance(payload, dict):
            raise SourceError("Unexpected public-source JSON schema")
        return payload, observation

    def discover(self, term: str, start: str, end: str) -> dict:
        first, last = date.fromisoformat(start), date.fromisoformat(end)
        if first > last:
            raise ValueError("Start date must precede end date")
        collected: set[str] = set()

        def partition(lo: date, hi: date):
            args = {"conditions[term]": term, "conditions[publication_date][gte]": lo.isoformat(),
                    "conditions[publication_date][lte]": hi.isoformat(), "per_page": 1000, "order": "oldest", "page": 1}
            payload, obs = self.fetch_json(f"{BASE}/documents.json?{urlencode(args)}", kind="discovery")
            count = payload.get("count")
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise SourceError("Missing discovery count")
            if count > 2000:
                if lo == hi:
                    raise SourceError("Daily search exceeds API pagination limit; narrow the query")
                mid = lo + (hi - lo) // 2
                partition(lo, mid)
                partition(mid + timedelta(days=1), hi)
                return
            seen: set[str] = set()
            for page in range(1, max(1, (count + 999) // 1000) + 1):
                if page > 1:
                    args["page"] = page
                    payload, obs = self.fetch_json(f"{BASE}/documents.json?{urlencode(args)}", kind="discovery")
                records = payload.get("results")
                if not isinstance(records, list):
                    raise SourceError("Missing discovery results")
                for record in records:
                    doc_id = validate_document_id(record.get("document_number"))
                    self.store.put_document(record, obs)
                    seen.add(doc_id)
            if len(seen) != count:
                raise SourceError("Search count changed or results were truncated; rerun a narrower date window")
            collected.update(seen)

        partition(first, last)
        return {"documents_retrieved": len(collected), "document_ids": sorted(collected), "term": term,
                "start": start, "end": end, "interpretation": "Documents, not independent economic events."}

    def collect(self, doc_id: str) -> dict:
        validate_document_id(doc_id)
        metadata, obs = self.fetch_json(f"{BASE}/documents/{doc_id}.json", kind="detail", doc_id=doc_id)
        if metadata.get("document_number") != doc_id:
            raise SourceError("Detail document ID mismatch")
        self.store.put_document(metadata, obs)
        result = {"document_id": doc_id, "detail": "archived", "warnings": []}
        for key, kind in (("raw_text_url", "published_text"),):
            if metadata.get(key):
                try:
                    self.fetch(metadata[key], kind=kind, doc_id=doc_id)
                    result[kind] = "archived"
                except SourceError as exc:
                    result["warnings"].append(str(exc))
        try:
            public = self.fetch_json(f"{BASE}/public-inspection-documents/{doc_id}.json", kind="public_inspection", doc_id=doc_id, optional=True)
            if public:
                pi, _ = public
                if pi.get("document_number") != doc_id:
                    raise SourceError("Public-inspection document ID mismatch")
                result["public_inspection_filed_at"] = pi.get("filed_at")
                if pi.get("raw_text_url"):
                    self.fetch(pi["raw_text_url"], kind="public_inspection_text", doc_id=doc_id)
                    result["public_inspection_text"] = "archived"
            else:
                result["warnings"].append("Public-inspection metadata unavailable (404)")
        except SourceError as exc:
            result["warnings"].append(str(exc))
        result["earliest_public_disclosure_verified"] = False
        return result
