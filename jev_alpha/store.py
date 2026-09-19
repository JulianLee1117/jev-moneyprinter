"""Local source archive: immutable bytes and append-only observations."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


DOCUMENT_ID = re.compile(r"\d{4}-\d{4,6}\Z")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def validate_document_id(value: str) -> str:
    if not isinstance(value, str) or not DOCUMENT_ID.fullmatch(value):
        raise ValueError("Expected a Federal Register document number, e.g. 2026-14026")
    return value


def canonical_json(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def read_json(path: str | Path) -> dict:
    def reject_constant(value: str):
        raise ValueError(f"Invalid JSON numeric constant: {value}")
    return json.loads(Path(path).read_text(encoding="utf-8-sig"), parse_constant=reject_constant)


def write_new_json(path: str | Path, value: object) -> None:
    """Never silently replace a frozen selection or a human annotation."""
    target = Path(path)
    payload = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as handle:
        handle.write(payload)


class Store:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.blobs = self.root / "blobs"
        self.blobs.mkdir(exist_ok=True)
        self.db = sqlite3.connect(self.root / "research.sqlite3")
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA foreign_keys = ON")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS observations (
                id INTEGER PRIMARY KEY, source_url TEXT NOT NULL,
                observed_at TEXT NOT NULL, original_captured_at TEXT,
                status INTEGER NOT NULL, sha256 TEXT NOT NULL,
                kind TEXT NOT NULL, document_id TEXT
            );
            CREATE TABLE IF NOT EXISTS document_versions (
                id INTEGER PRIMARY KEY, document_id TEXT NOT NULL,
                observation_id INTEGER NOT NULL REFERENCES observations(id),
                metadata_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS documents (
                document_id TEXT PRIMARY KEY, metadata_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS model_runs (
                request_sha256 TEXT PRIMARY KEY,
                created_at TEXT NOT NULL, response_sha256 TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS model_attempts (
                request_sha256 TEXT PRIMARY KEY, started_at TEXT NOT NULL,
                finished_at TEXT, reserved_usd REAL NOT NULL,
                accounted_usd REAL NOT NULL, status TEXT NOT NULL
            );
        """)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.db.close()

    def put_blob(self, content: bytes) -> str:
        digest = hashlib.sha256(content).hexdigest()
        target = self.blobs / digest
        try:
            with target.open("xb") as handle:
                handle.write(content)
        except FileExistsError:
            if target.read_bytes() != content:
                raise ValueError("Archive content hash mismatch")
        return digest

    def observe(self, url: str, content: bytes, *, kind: str, status: int = 200,
                document_id: str | None = None, original_captured_at: str | None = None) -> int:
        if document_id:
            validate_document_id(document_id)
        digest = self.put_blob(content)
        with self.db:
            cursor = self.db.execute(
                "INSERT INTO observations(source_url,observed_at,original_captured_at,status,sha256,kind,document_id) VALUES(?,?,?,?,?,?,?)",
                (url, utc_now(), original_captured_at, status, digest, kind, document_id))
        return cursor.lastrowid

    def put_document(self, metadata: dict, observation_id: int) -> None:
        doc_id = validate_document_id(metadata.get("document_number"))
        previous = self.get_document(doc_id) or {}
        merged = {**previous, **metadata}
        with self.db:
            self.db.execute("INSERT INTO document_versions(document_id,observation_id,metadata_json) VALUES(?,?,?)",
                            (doc_id, observation_id, canonical_json(metadata).decode()))
            self.db.execute("INSERT INTO documents VALUES(?,?) ON CONFLICT(document_id) DO UPDATE SET metadata_json=excluded.metadata_json",
                            (doc_id, canonical_json(merged).decode()))

    def get_document(self, doc_id: str) -> dict | None:
        row = self.db.execute("SELECT metadata_json FROM documents WHERE document_id=?", (doc_id,)).fetchone()
        return json.loads(row[0]) if row else None

    def documents(self) -> list[dict]:
        return [json.loads(row[0]) for row in self.db.execute("SELECT metadata_json FROM documents ORDER BY document_id")]

    def latest_source(self, doc_id: str, kind: str) -> dict | None:
        row = self.db.execute("SELECT * FROM observations WHERE document_id=? AND kind=? AND status=200 ORDER BY id DESC LIMIT 1",
                              (doc_id, kind)).fetchone()
        if not row:
            return None
        result = dict(row)
        result["content"] = (self.blobs / result["sha256"]).read_bytes()
        return result

    def cache_response(self, request: dict, response: dict) -> str:
        req_hash = self.put_blob(canonical_json(request))
        response_hash = self.put_blob(canonical_json(response))
        with self.db:
            self.db.execute("INSERT OR IGNORE INTO model_runs VALUES(?,?,?)", (req_hash, utc_now(), response_hash))
        return req_hash

    def cached_response(self, request: dict) -> dict | None:
        req_hash = hashlib.sha256(canonical_json(request)).hexdigest()
        row = self.db.execute("SELECT response_sha256 FROM model_runs WHERE request_sha256=?", (req_hash,)).fetchone()
        return json.loads((self.blobs / row[0]).read_bytes()) if row else None

    def reserve_attempt(self, request: dict, estimate_usd: float, phase_limit: float = 20,
                        *, retry_reason: str | None = None) -> str:
        import math
        if not math.isfinite(estimate_usd) or estimate_usd < 0 or not math.isfinite(phase_limit) or phase_limit <= 0:
            raise ValueError("Invalid model budget")
        digest = self.put_blob(canonical_json(request))
        with self.db:
            self.db.execute("BEGIN IMMEDIATE")
            existing = self.db.execute("SELECT status FROM model_attempts WHERE request_sha256=?", (digest,)).fetchone()
            if existing:
                if not isinstance(retry_reason, str) or not retry_reason.strip():
                    raise ValueError("This request already has a recorded attempt; inspect its result/billing before retrying")
                if existing[0] not in {"outcome_uncertain", "invalid_response"}:
                    raise ValueError("Only a finished failed attempt can receive a deliberate diagnostic retry")
                # New accounting row retains the original uncertain reservation.
                # Its immutable envelope links to the exact original request.
                digest = self.put_blob(canonical_json({"request": request, "retry_of": digest,
                    "retry_reason": retry_reason.strip(), "created_at": utc_now()}))
            elif retry_reason is not None:
                raise ValueError("Diagnostic retry requires an existing failed attempt")
            spent = self.db.execute("SELECT COALESCE(SUM(accounted_usd),0) FROM model_attempts").fetchone()[0]
            if spent + estimate_usd > phase_limit:
                raise ValueError("Estimated cumulative model budget exceeded")
            self.db.execute("INSERT INTO model_attempts VALUES(?,?,NULL,?,?,?)",
                            (digest, utc_now(), estimate_usd, estimate_usd, "started"))
        return digest

    def finish_attempt(self, digest: str, response: dict | None, *, status: str | None = None) -> None:
        if status not in {None, "completed", "outcome_uncertain", "invalid_response"}:
            raise ValueError("Invalid attempt status")
        reported = (response or {}).get("usage", {}).get("cost")
        with self.db:
            # Unknown/failed billing keeps its conservative reservation. A reported
            # actual cost can exceed the estimate; it is preserved, never capped.
            self.db.execute("UPDATE model_attempts SET finished_at=?, status=?, accounted_usd=CASE WHEN ? IS NULL THEN accounted_usd ELSE ? END WHERE request_sha256=?",
                            (utc_now(), status or ("completed" if response else "outcome_uncertain"), reported, reported, digest))
