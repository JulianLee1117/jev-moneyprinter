"""Bounded, source-native operating-change collection, with no model or price calls.

Live Scry access is explicitly limited to the registered technical evaluation.
Credentials travel in curl stdin, responses are archived, and an independent
Store ledger reserves the maximum permitted exposure before every request.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import subprocess
import unicodedata
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .credentials import read_key
from .store import Store, canonical_json, utc_now, write_new_json
from .transport import _config_quote

BASE = "https://api.scry.io"
SCHEMA_URL = BASE + "/v1/scry/schema"
QUERY_URL = BASE + "/v1/scry/query"
RERANK_URL = BASE + "/v1/scry/rerank"
MAX_RESPONSE_BYTES = 12_000_000
STATUS_MARKER = b"\n__SCRY_HTTP_STATUS__:"
SOURCES = ("hackernews", "reddit")
KNOWN_LIMITS = [
    "Reddit live coverage is partial and not comparable with historical backfills.",
    "Creation time does not establish when the current text became available.",
    "Hacker News date-only observation metadata does not establish intraday availability.",
    "Missing Hacker News story roots prevent complete thread resolution.",
    "Product-mention samples do not estimate customer counts or vendor revenue.",
]
SELECTION_SEMANTICS = {
    "version": "indexed-native-sample-v2",
    "alias_match": "Case-insensitive whole alphanumeric tokens, using Scry hasAnyTokens; no substring or action-word filter.",
    "correction": "The failed v1 query used unindexed substring matching. Before any records were obtained, this was corrected to indexed whole-token product matching; mongo does not match mongolia.",
    "hackernews_state": "Source relation is already newest-wins FINAL; no additional state sort.",
    "reddit_state": "Indexed candidate IDs, all states of those IDs in the frozen stratum, latest-state fold, then recheck product tokens.",
}


class OperatingSourceError(RuntimeError):
    """Redacted source failure; no automatic retry is permitted."""

    def __init__(self, message, *, manifest=None):
        super().__init__(message)
        self.manifest = manifest or {}


def _finite_positive(value, label):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{label} must be positive and finite")
    return float(value)


def _utc(value, *, date_only=False, require_aware=False):
    if value is None or value == "":
        return None
    if type(value) in (int, float):
        result = datetime.fromtimestamp(value, timezone.utc)
    elif isinstance(value, str):
        text = value.strip()
        if re.fullmatch(r"\d{10}(?:\.\d+)?", text):
            result = datetime.fromtimestamp(float(text), timezone.utc)
        else:
            result = datetime.fromisoformat(text.replace("Z", "+00:00"))
            if result.tzinfo is None:
                # Scry DateTime fields are UTC; user protocol timestamps must be aware.
                if require_aware or ("T" in text and not date_only):
                    raise ValueError("Timestamp has no timezone")
                result = result.replace(tzinfo=timezone.utc)
    else:
        raise ValueError("Invalid timestamp")
    if date_only:
        result = result.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
    return result.astimezone(timezone.utc).isoformat()


def _joined(*values):
    return "\n\n".join(value for value in values if isinstance(value, str) and value)


def normalize_operating_record(source, row, *, captured_at, vendor_ids=()):
    """Retain full text and conservative availability; never infer author identity."""
    if source not in SOURCES or not isinstance(row, dict):
        raise ValueError("Unsupported operating source record")
    captured = _utc(captured_at, require_aware=True)
    if not captured:
        raise ValueError("An exact capture timestamp is required")
    if source == "hackernews":
        native = str(row.get("hn_id", row.get("id", "")))
        if not re.fullmatch(r"[1-9]\d*", native):
            raise ValueError("Invalid Hacker News native ID")
        text = _joined(row.get("title"), row.get("payload", row.get("text")))
        published = _utc(row.get("original_timestamp", row.get("time")))
        root = row.get("story_hn_id")
        parent = row.get("parent_hn_id", row.get("parent"))
        if root in (None, "", 0, "0") and (row.get("type") == "story" or (row.get("title") and not parent)):
            root = native
        root = str(root) if root not in (None, "", 0, "0") else None
        parent = str(parent) if parent not in (None, "", 0, "0") else None
        url = "https://news.ycombinator.com/item?id=" + native
        linked = row.get("uri", row.get("url"))
        author = row.get("original_author", row.get("by"))
        observed = row.get("state_observed_at")
        precision = "second" if observed else "unknown"
        if observed:
            state = _utc(observed)
        elif row.get("observed_on"):
            state = _utc(row["observed_on"], date_only=True)
            precision = "day_upper_bound"
        else:
            state = None
        resolution = "root" if root else ("parent_only" if parent else "unknown")
    else:
        native = str(row.get("id", row.get("native_id", "")))
        native = native if native.startswith("t3_") else "t3_" + native
        if not re.fullmatch(r"t3_[a-z0-9]+", native):
            raise ValueError("Invalid Reddit post native ID")
        subreddit = row.get("subreddit", "")
        if not isinstance(subreddit, str) or not re.fullmatch(r"[A-Za-z0-9_]+", subreddit):
            raise ValueError("Invalid subreddit")
        text = _joined(row.get("title"), row.get("selftext"))
        published = _utc(row.get("created_utc"))
        root, parent, resolution = native, None, "root"
        url = f"https://www.reddit.com/r/{subreddit}/comments/{native[3:]}/"
        linked, author = row.get("url"), row.get("author")
        state = _utc(row.get("state_observed_at"))
        retrieval = _utc(row.get("retrieved_on"))
        precision = ("second" if retrieval == state else "hour_upper_bound") if state else "unknown"
        if not state and row.get("observed_on"):
            state = _utc(row["observed_on"], date_only=True)
            precision = "day_upper_bound"
    if not text or not published:
        raise ValueError("Full source text and publication timestamp are required")
    record = {
        "record_id": source + ":" + native, "source": source, "native_id": native,
        "text": text, "url": url, "published_at": published,
        "state_observed_at": state, "state_observed_at_precision": precision,
        "captured_at": captured, "content_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "thread_id": source + ":" + root if root else None,
        "parent_id": source + ":" + parent if parent else None,
        "thread_resolution": resolution, "vendor_ids": sorted(set(vendor_ids)),
    }
    if isinstance(linked, str) and linked.startswith(("https://", "http://")):
        record["linked_url"] = linked
    if isinstance(author, str) and author and author not in {"[deleted]", "[removed]"}:
        record["author_hash"] = hashlib.sha256((source + ":" + author).encode("utf-8")).hexdigest()
    return record


def selection_hash(seed, record_id):
    return hashlib.sha256((seed + "|" + record_id).encode("utf-8")).hexdigest()


def _body_key(text):
    return " ".join(unicodedata.normalize("NFKC", text).split())


def merge_operating_records(records):
    """Merge native IDs and exact normalized bodies, preserving all original refs."""
    result, identities, bodies = [], {}, {}
    for record in sorted(records, key=lambda item: item["record_id"]):
        body = _body_key(record["text"])
        previous = identities.get(record["record_id"], bodies.get(body))
        ref = {key: record.get(key) for key in (
            "record_id", "url", "published_at", "state_observed_at", "state_observed_at_precision",
            "captured_at", "content_sha256", "thread_id", "parent_id", "thread_resolution")}
        if previous is None:
            previous = dict(record, source_references=[ref], duplicate_record_ids=[])
            result.append(previous)
        else:
            previous["vendor_ids"] = sorted(set(previous["vendor_ids"] + record["vendor_ids"]))
            if ref not in previous["source_references"]:
                previous["source_references"].append(ref)
            if record["record_id"] != previous["record_id"]:
                previous["duplicate_record_ids"] = sorted(set(previous["duplicate_record_ids"] + [record["record_id"]]))
        identities[record["record_id"]] = bodies[body] = previous
    return result


def validate_protocol(protocol):
    if not isinstance(protocol, dict) or protocol.get("use_case") != "technical_evaluation_only":
        raise ValueError("Explicit use_case='technical_evaluation_only' is required")
    if protocol.get("source_mode", "scry") not in {"scry", "hackernews_direct"}:
        raise ValueError("Unsupported operating source mode")
    vendors = protocol.get("vendors", [])
    if not vendors or len(vendors) > 8:
        raise ValueError("The pilot supports one to eight frozen vendors")
    ids = set()
    for vendor in vendors:
        vid, aliases = vendor.get("vendor_id"), vendor.get("aliases")
        if not isinstance(vid, str) or not re.fullmatch(r"[a-z0-9_]+", vid) or vid in ids:
            raise ValueError("Vendor IDs must be unique safe identifiers")
        ids.add(vid)
        if (not isinstance(aliases, list) or not aliases or len(aliases) > 20
                or any(not isinstance(a, str) or not a.strip() or len(a) > 100 for a in aliases)):
            raise ValueError("Explicit bounded vendor aliases are required")
    window = protocol.get("window", {})
    start, end = _utc(window.get("start"), require_aware=True), _utc(window.get("end_exclusive"), require_aware=True)
    if not start or not end or start >= end:
        raise ValueError("A valid frozen UTC window is required")
    for field, maximum in (("max_records", 1000), ("max_per_vendor", 125)):
        value = protocol.get(field)
        if type(value) is not int or not 1 <= value <= maximum:
            raise ValueError(f"{field} exceeds the bounded pilot")
    quotas = protocol.get("source_quotas", {})
    if set(quotas) != set(SOURCES) or any(type(v) is not int or not 0 <= v <= 125 for v in quotas.values()):
        raise ValueError("Explicit Hacker News and Reddit quotas are required")
    if sum(quotas.values()) > protocol["max_per_vendor"] or len(vendors) * sum(quotas.values()) > protocol["max_records"]:
        raise ValueError("Source quotas exceed the frozen record limits")
    if not isinstance(protocol.get("selection_seed"), str) or not protocol["selection_seed"]:
        raise ValueError("A frozen selection seed is required")
    subreddits = protocol.get("subreddits", [])
    if not subreddits or len(subreddits) > 30 or any(not isinstance(s, str) or not re.fullmatch(r"[A-Za-z0-9_]+", s) for s in subreddits):
        raise ValueError("A bounded explicit subreddit list is required")
    return start, end


def _literal(value):
    return "'" + value.replace("\\", "\\\\").replace("'", "\\'") + "'"


def build_source_query(protocol, vendor, source, schema):
    """Schema-gated SQL returns only the hashed bounded sample, never a bulk dump."""
    validate_protocol(protocol)
    relation = {"hackernews": "hackernews.items", "reddit": "reddit.posts"}[source]
    surfaces = schema.get("surfaces", [])
    surface = next((s for s in surfaces if s.get("relation") == relation), None)
    if not surface:
        raise ValueError(f"Schema does not expose {relation}")
    columns = {c["name"]: c.get("type", "") for c in surface.get("columns", [])}
    required = ({"hn_id", "title", "payload", "original_timestamp", "observed_on", "search_text_lc"}
                if source == "hackernews" else {"id", "subreddit", "title", "selftext", "created_utc", "state_observed_at", "search_text_lc"})
    if not required <= set(columns):
        raise ValueError(f"Schema is missing required {relation} columns")
    wanted = (["hn_id", "uri", "title", "payload", "original_author", "original_timestamp", "parent_hn_id", "story_hn_id", "first_observed_on", "observed_on", "is_deleted", "dead"]
              if source == "hackernews" else ["id", "subreddit", "author", "created_utc", "title", "selftext", "url", "retrieved_on", "observed_on", "state_observed_at", "edited"])
    projection = ", ".join(name for name in wanted if name in columns)
    id_column, timestamp = ("hn_id", "original_timestamp") if source == "hackernews" else ("id", "created_utc")
    native = "toString(hn_id)" if source == "hackernews" else "concat('t3_', toString(id))"
    time_expr = (timestamp if "DateTime" in columns[timestamp] else
                 f"toDateTime({timestamp}, 'UTC')" if re.search(r"Int|Float|Decimal", columns[timestamp])
                 else f"parseDateTimeBestEffortOrNull(toString({timestamp}), 'UTC')")
    start, end = protocol["window"]["start"], protocol["window"]["end_exclusive"]
    clauses = [f"{time_expr} >= parseDateTimeBestEffort({_literal(start)})",
               f"{time_expr} < parseDateTimeBestEffort({_literal(end)})"]
    # The registered aliases are single tokens. Indexed whole-token matching is
    # explicit in the report; it must not silently become substring matching.
    if any(not re.fullmatch(r"[a-z0-9]+", alias.lower()) for alias in vendor["aliases"]):
        raise ValueError("Indexed collection requires single alphanumeric alias tokens")
    aliases = ", ".join(_literal(alias.lower()) for alias in vendor["aliases"])
    product_predicate = f"hasAnyTokens(search_text_lc, [{aliases}])"
    if source == "reddit":
        clauses.append("lowerUTF8(subreddit) IN (" + ", ".join(_literal(s.lower()) for s in protocol["subreddits"]) + ")")
    order = f"lower(hex(SHA256(concat({_literal(protocol['selection_seed'] + '|' + source + ':')}, {native}))))"
    ending = f" ORDER BY {order}, {id_column} LIMIT {protocol['source_quotas'][source]}"
    stratum = " AND ".join(clauses)
    if source == "hackernews":
        if "FINAL" not in surface.get("description", ""):
            raise ValueError("Hacker News schema does not attest newest-wins FINAL semantics")
        return f"SELECT {projection} FROM {relation} WHERE {stratum} AND {product_predicate}" + ending
    # Do not limit candidate IDs before the current-state fold: removed mentions
    # would otherwise either be resurrected or consume slots in a biased sample.
    candidate_ids = f"SELECT DISTINCT id FROM {relation} WHERE {stratum} AND {product_predicate}"
    latest = "state_observed_at DESC" + (", retrieved_on DESC" if "retrieved_on" in columns else "")
    inner = f"SELECT {projection}, search_text_lc FROM {relation} WHERE {stratum} AND id IN ({candidate_ids})"
    inner += f" ORDER BY {latest} LIMIT 1 BY id"
    return f"SELECT {projection} FROM ({inner}) WHERE {product_predicate}" + ending


def _redact(raw, key):
    for secret in {key.encode("utf-8"), json.dumps(key)[1:-1].encode("utf-8")}:
        raw = raw.replace(secret, b"[REDACTED]")
    return raw


class ScryClient:
    """Single-attempt client; the billing Store is independent of model spending."""

    def __init__(self, store, protocol, *, max_scry_usd=5.0, key=None):
        validate_protocol(protocol)
        self.store, self.protocol = store, protocol
        self.limit = min(5.0, _finite_positive(max_scry_usd, "Scry limit"),
                         _finite_positive(protocol.get("budget", {}).get("scry_usd", 5.0), "Protocol Scry budget"))
        self.key = key if key is not None else read_key("SCRY_API_KEY")
        if not isinstance(self.key, str) or not self.key or any(not 33 <= ord(c) <= 126 for c in self.key):
            raise ValueError("A valid SCRY_API_KEY is required")
        self.ledger = Store(store.root / "scry-billing")

    def close(self):
        self.ledger.db.close()

    def spending(self):
        rows = self.ledger.db.execute("SELECT status, reserved_usd, accounted_usd FROM model_attempts").fetchall()
        return {"limit_usd": self.limit, "accounted_usd": sum(r["accounted_usd"] for r in rows),
                "attempt_count": len(rows), "uncertain_attempts": sum(r["status"] != "completed" for r in rows),
                "ledger_path": str(self.ledger.root)}

    def request(self, method, endpoint, payload=None, *, exposure_usd=None):
        allowed = {("GET", SCHEMA_URL), ("POST", QUERY_URL), ("POST", RERANK_URL)}
        if (method, endpoint) not in allowed:
            raise ValueError("Unsupported Scry endpoint")
        if method == "GET" and payload is not None:
            raise ValueError("Schema GET has no request body")
        if endpoint == QUERY_URL and (not isinstance(payload, str) or not payload.startswith("SELECT ")):
            raise ValueError("Only bounded SELECT query strings are supported")
        if endpoint == QUERY_URL:
            limit = re.search(r" LIMIT ([1-9]\d*)$", payload)
            if not limit or int(limit[1]) > 125 or ";" in payload:
                raise ValueError("Scry SQL requires one bounded SELECT with LIMIT at most 125")
        if endpoint == RERANK_URL and not isinstance(payload, dict):
            raise ValueError("Rerank requires a JSON object")
        if endpoint == RERANK_URL:
            documents = payload.get("documents")
            if not isinstance(documents, list) or not 2 <= len(documents) <= 1000:
                raise ValueError("Rerank requires 2 to 1000 explicitly supplied documents")
        ceiling = exposure_usd if exposure_usd is not None else self.protocol.get("budget", {}).get("initial_query_ceiling_usd", 0.10)
        ceiling = _finite_positive(ceiling, "Query exposure")
        nanos = math.floor(ceiling * 1_000_000_000)
        if nanos < 1 or ceiling > self.limit:
            raise ValueError("Query exposure is outside the source budget")
        reservation = nanos / 1_000_000_000
        request = {"provider": "scry", "method": method, "endpoint": endpoint, "payload": payload,
                   "max_exposure_nanodollars": nanos,
                   "protocol_sha256": hashlib.sha256(canonical_json({k: v for k, v in self.protocol.items() if k != "fixtures"})).hexdigest()}
        cached = self.ledger.cached_response(request)
        if cached is not None:
            return dict(cached, cache_hit=True)
        executable = shutil.which("curl.exe") or shutil.which("curl")
        if executable is None:
            raise OperatingSourceError("curl is unavailable; no request was attempted")
        digest = self.ledger.reserve_attempt(request, reservation, self.limit)
        content_type = "text/plain" if endpoint == QUERY_URL else "application/json"
        config = ["url = " + _config_quote(endpoint), "request = " + _config_quote(method),
                  'proto = "=https"', "no-location", "max-redirs = 0", "retry = 0", "silent",
                  "max-time = 60", "connect-timeout = 10", f"max-filesize = {MAX_RESPONSE_BYTES}",
                  'user-agent = "jev-alpha-research/0.1"', 'header = "Accept: application/json"',
                  "header = " + _config_quote("Content-Type: " + content_type),
                  "header = " + _config_quote("Authorization: Bearer " + self.key),
                  'header = "x-scry-max-seconds: 5"', 'header = "x-scry-budget: 50000000"',
                  "header = " + _config_quote(f"x-scry-max-exposure: {nanos}"),
                  "write-out = " + _config_quote(STATUS_MARKER.decode() + "%{http_code}")]
        if payload is not None:
            body = payload if isinstance(payload, str) else canonical_json(payload).decode("utf-8")
            config.append("data-binary = " + _config_quote(body))
        manifest = {"request_sha256": digest, "request": request, "reservation_usd": reservation,
                    "cache_hit": False, "captured_at": None, "http_status": None}
        try:
            result = subprocess.run([executable, "-q", "--config", "-"],
                input=("\n".join(config) + "\n").encode("utf-8"), stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL, timeout=65, check=False, shell=False,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            raw = result.stdout
            if not isinstance(raw, bytes) or len(raw) > MAX_RESPONSE_BYTES + len(STATUS_MARKER) + 3:
                raise OperatingSourceError("Invalid or oversized Scry transport output")
            body, marker, status = raw.rpartition(STATUS_MARKER)
            if not marker or not re.fullmatch(rb"[1-5]\d\d", status):
                raise OperatingSourceError("Scry transport returned no valid HTTP status")
            status = int(status)
            redacted = _redact(body, self.key)
            manifest.update(captured_at=utc_now(), http_status=status,
                            response_redacted=redacted != body,
                            response_sha256=hashlib.sha256(redacted).hexdigest())
            manifest["observation_id"] = self.store.observe(endpoint, redacted, kind="scry-response", status=status)
            envelope = json.loads(redacted, parse_constant=lambda x: (_ for _ in ()).throw(ValueError("Nonfinite JSON")))
            if not isinstance(envelope, dict):
                raise OperatingSourceError("Scry response must be a JSON object")
            manifest["envelope"] = envelope
            cost = envelope.get("spend_nanodollars")
            if cost is not None and (type(cost) not in (int, float) or not math.isfinite(cost) or cost < 0):
                raise OperatingSourceError("Scry returned invalid billing metadata")
            if result.returncode or not 200 <= status < 300:
                raise OperatingSourceError(f"Scry returned HTTP {status} or an incomplete transfer; no retry was attempted")
            self.ledger.finish_attempt(digest, {"usage": {"cost": cost / 1_000_000_000}} if cost is not None else None,
                                       status="completed" if cost is not None else "outcome_uncertain")
            manifest["accounted_usd"] = cost / 1_000_000_000 if cost is not None else reservation
            manifest["billing_outcome"] = "reported" if cost is not None else "uncertain_reserved"
            self.ledger.cache_response(request, manifest)
            return manifest
        except (OperatingSourceError, OSError, subprocess.SubprocessError, ValueError, UnicodeError) as error:
            self.ledger.finish_attempt(digest, None, status="outcome_uncertain")
            manifest.update(billing_outcome="uncertain_reserved", accounted_usd=reservation)
            message = str(error) if isinstance(error, OperatingSourceError) else "Scry request failed; no automatic retry was attempted"
            raise OperatingSourceError(message, manifest=manifest) from None


def _rows(envelope):
    rows, columns = envelope.get("rows", []), envelope.get("columns", [])
    names = [c["name"] if isinstance(c, dict) else c for c in columns]
    result = []
    for row in rows:
        if isinstance(row, dict):
            result.append(row)
        elif isinstance(row, list) and len(row) == len(names):
            result.append(dict(zip(names, row)))
        else:
            raise ValueError("Invalid Scry row/column envelope")
    return result


def _qualifies(record, row, vendor, protocol):
    start, end = _utc(protocol["window"]["start"]), _utc(protocol["window"]["end_exclusive"])
    if not start <= record["published_at"] < end:
        return False
    if record["source"] == "reddit" and row.get("subreddit", "").lower() not in {s.lower() for s in protocol["subreddits"]}:
        return False
    # ClickHouse's words tokenizer recognizes alphanumeric tokens; underscore
    # and punctuation delimit tokens. All registered aliases are ASCII words.
    tokens = set(re.findall(r"[^\W_]+", record["text"].lower(), flags=re.UNICODE))
    return any(alias.lower() in tokens for alias in vendor["aliases"])


def _direct_hn(store, ids):
    """Only explicitly supplied item IDs; no discovery or recursive expansion."""
    if not isinstance(ids, list) or len(ids) > 1000 or any(type(i) not in (int, str) or not re.fullmatch(r"[1-9]\d*", str(i)) for i in ids):
        raise ValueError("Direct HN requires at most 1000 explicit native IDs")
    rows, manifests = [], []
    for native in sorted(set(map(str, ids)), key=int):
        url = f"https://hacker-news.firebaseio.com/v0/item/{native}.json"
        request = urllib.request.Request(url, headers={"User-Agent": "jev-alpha-research/0.1", "Accept": "application/json"})
        with urllib.request.urlopen(request, timeout=30) as response:
            raw = response.read(MAX_RESPONSE_BYTES + 1)
        if len(raw) > MAX_RESPONSE_BYTES:
            raise OperatingSourceError("Direct HN item exceeds the source size limit")
        captured = utc_now()
        obs = store.observe(url, raw, kind="hackernews-direct")
        row = json.loads(raw)
        manifests.append({"url": url, "observation_id": obs, "captured_at": captured,
                          "response_sha256": hashlib.sha256(raw).hexdigest()})
        if isinstance(row, dict) and str(row.get("id")) == native:
            rows.append((row, captured))
    return rows, manifests


def collect_operating_sources(store, protocol, out: Path, *, live=False, max_scry_usd=5.0):
    """Collect a frozen bounded cohort and write one immutable JSON report.

    Offline fixtures: protocol['fixtures'] contains 'schema', 'hackernews' and
    'reddit' row lists, plus an exact 'captured_at'. Without fixtures, offline mode
    is a dry run. Live mode never uses fixtures and never retries failed calls.
    """
    validate_protocol(protocol)
    out = Path(out)
    if out.exists():
        raise FileExistsError("Source report already exists; refusing to overwrite")
    report = {"schema_version": "operating-source-collection-v1", "experiment_id": protocol.get("experiment_id"),
              "created_at": utc_now(), "live": live, "source_mode": protocol.get("source_mode", "scry"),
              "protocol_sha256": hashlib.sha256(canonical_json({k: v for k, v in protocol.items() if k != "fixtures"})).hexdigest(),
              "records": [], "queries": [], "coverage": [], "known_limits": KNOWN_LIMITS,
              "selection_semantics": SELECTION_SEMANTICS,
              "errors": [], "status": "dry_run", "historical_alpha_eligible": False,
              "billing": {"accounted_usd": 0.0, "attempt_count": 0}}
    report["coverage"] = [{"vendor_id": vendor["vendor_id"], "source": source,
                           "quota": protocol["source_quotas"][source], "selected_before_merge": 0,
                           "unfilled": protocol["source_quotas"][source], "no_crossfill": True,
                           "status": "not_attempted"}
                          for vendor in protocol["vendors"] for source in SOURCES]
    fixtures = protocol.get("fixtures") if not live else None
    if not live and fixtures is None:
        write_new_json(out, report)
        return report
    client, selected = None, []
    try:
        direct_rows = None
        if fixtures is not None:
            schema = fixtures.get("schema", {})
            captured = _utc(fixtures["captured_at"], require_aware=True)
            report["schema"] = {"fixture": True, "envelope": schema}
        elif protocol.get("source_mode") == "hackernews_direct":
            direct_rows, report["direct_requests"] = _direct_hn(store, protocol.get("direct_hn_ids"))
            schema = {}
            report["known_limits"] = KNOWN_LIMITS + ["Direct HN coverage is restricted to explicitly supplied IDs; it is not a source census."]
        else:
            client = ScryClient(store, protocol, max_scry_usd=max_scry_usd)
            report["schema"] = client.request("GET", SCHEMA_URL, exposure_usd=0.000000001)
            schema = report["schema"]["envelope"]
        for vendor in protocol["vendors"]:
            for source in SOURCES:
                quota = protocol["source_quotas"][source]
                coverage = next(c for c in report["coverage"] if c["vendor_id"] == vendor["vendor_id"] and c["source"] == source)
                if not quota or (direct_rows is not None and source == "reddit"):
                    coverage["status"] = "not_requested" if not quota else "unavailable_in_direct_mode"
                    continue
                if direct_rows is not None:
                    rows = direct_rows
                    coverage["status"] = "supplied_ids_only"
                else:
                    sql = build_source_query(protocol, vendor, source, schema)
                    if fixtures is not None:
                        query = {"fixture": True, "request": {"payload": sql}, "vendor_id": vendor["vendor_id"],
                                 "source": source, "envelope": fixtures.get("envelopes", {}).get(source, {})}
                        rows = [(row, captured) for row in fixtures.get(source, [])]
                    else:
                        query = client.request("POST", QUERY_URL, sql)
                        query.update(vendor_id=vendor["vendor_id"], source=source)
                        rows = [(row, query["captured_at"]) for row in _rows(query["envelope"])]
                        if len(rows) > quota:
                            raise OperatingSourceError("Scry exceeded the server-side row limit", manifest=query)
                    report["queries"].append(query)
                    envelope = query["envelope"]
                    coverage.update({key: envelope.get(key) for key in ("completeness", "coverage", "deadline_partial", "truncated", "rows_before_limit_at_least", "row_count")})
                    coverage["status"] = "fixture" if fixtures is not None else ("complete_query" if envelope.get("completeness") == "complete" and not envelope.get("deadline_partial") and not envelope.get("truncated") else "incomplete_or_unknown")
                candidates = {}
                for row, row_capture in rows:
                    try:
                        record = normalize_operating_record(source, row, captured_at=row_capture, vendor_ids=[vendor["vendor_id"]])
                        old = candidates.get(record["record_id"])
                        if old is None or (record["state_observed_at"] or "") > (old[0]["state_observed_at"] or ""):
                            candidates[record["record_id"]] = (record, row)
                    except (ValueError, TypeError, OverflowError):
                        coverage["invalid_rows"] = coverage.get("invalid_rows", 0) + 1
                matching = [record for record, row in candidates.values() if _qualifies(record, row, vendor, protocol)]
                chosen = sorted(matching, key=lambda r: (selection_hash(protocol["selection_seed"], r["record_id"]), r["record_id"]))[:quota]
                selected.extend(chosen)
                coverage.update(selected_before_merge=len(chosen), unfilled=quota - len(chosen))
        report["status"] = "collected"
    except (OperatingSourceError, ValueError, OSError) as error:
        report["status"] = "stopped_source_error"
        report["errors"].append({"message": str(error) if isinstance(error, (OperatingSourceError, ValueError)) else "Source I/O failed",
                                 "manifest": error.manifest if isinstance(error, OperatingSourceError) else {}})
    finally:
        if client is not None:
            report["billing"] = client.spending()
            client.close()
    report["records"] = merge_operating_records(selected)
    report["selected_before_merge"] = len(selected)
    report["record_count"] = len(report["records"])
    write_new_json(out, report)
    return report
