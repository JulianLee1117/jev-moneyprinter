"""Bounded original Hacker News parent-chain resolution; no corpus or text edits."""
from __future__ import annotations

import copy
import hashlib
import json
import re
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .store import canonical_json, read_json, utc_now, write_new_json

MAX_BODY_BYTES = 1_000_000
MAX_HOPS = 20
KIND = "operating-hn-parent"


def _native(value):
    if not isinstance(value, str):
        return None
    match = re.fullmatch(r"hackernews:([1-9]\d*)", value)
    return match[1] if match else None


def _url(native):
    return f"https://hacker-news.firebaseio.com/v0/item/{native}.json"


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _fetch_hn_item(native):
    """One original API GET; errors are data, never implicit retries."""
    request = urllib.request.Request(_url(native), headers={
        "User-Agent": "jev-alpha-research/0.1", "Accept": "application/json"})
    try:
        with urllib.request.build_opener(_NoRedirect()).open(request, timeout=8) as response:
            status, raw = response.status, response.read(MAX_BODY_BYTES + 1)
    except urllib.error.HTTPError as error:
        status, raw = error.code, error.read(MAX_BODY_BYTES + 1)
    except (OSError, urllib.error.URLError, TimeoutError):
        return {"status": 0, "body": b"", "captured_at": utc_now(), "error": "transport_failure_no_retry"}
    if len(raw) > MAX_BODY_BYTES:
        return {"status": status, "body": b"", "captured_at": utc_now(), "error": "response_size_limit"}
    return {"status": status, "body": raw, "captured_at": utc_now()}


def _validate_item(native, fetched):
    metadata = {key: fetched.get(key) for key in ("observation_id", "captured_at", "status", "response_sha256", "cache_hit")}
    metadata.update(source="official_hn_api", native_id=native, url=_url(native))
    if fetched.get("error") or fetched["status"] != 200:
        return {"error": fetched.get("error", "http_error"), "provenance": metadata}
    try:
        item = json.loads(fetched["body"])
    except (ValueError, UnicodeError):
        return {"error": "invalid_json", "provenance": metadata}
    if not isinstance(item, dict):
        return {"error": "item_missing", "provenance": metadata}
    if type(item.get("id")) is not int or str(item["id"]) != native:
        return {"error": "native_id_mismatch", "provenance": metadata}
    kind, parent = item.get("type"), item.get("parent")
    if kind in {"story", "job", "poll"} and parent is None:
        return {"root": native, "provenance": dict(metadata, item_type=kind)}
    if kind in {"comment", "pollopt"} and type(parent) is int and parent > 0 and parent != item["id"]:
        return {"parent": str(parent), "provenance": dict(metadata, item_type=kind, parent_id="hackernews:" + str(parent))}
    return {"error": "invalid_item_type_or_parent", "provenance": metadata}


def _references(capture):
    for record in capture["records"]:
        yield record
        for reference in record.get("source_references", []):
            yield reference


def _trace(native, nodes):
    chain, seen, current = [], set(), native
    for _ in range(MAX_HOPS):
        if current in seen:
            return {"status": "cycle", "chain": chain}
        seen.add(current)
        node = nodes.get(current)
        if node is None:
            return {"status": "needs_item", "needed": current, "chain": chain}
        chain.append(node["provenance"])
        if node.get("error"):
            return {"status": node["error"], "chain": chain}
        if node.get("root"):
            return {"status": "resolved", "root": node["root"], "chain": chain}
        current = node["parent"]
    return {"status": "hop_limit", "chain": chain}


def resolve_operating_threads(store, capture, out: Path, *, live=False, max_requests=500, workers=4):
    """Resolve every missing native HN reference, preserving all selected evidence.

    Existing capture edges and immutable cached API observations are used before
    network requests. Only thread annotations change. A refusal or rate limit
    stops further API calls; unresolved references remain explicitly unresolved.
    """
    if type(max_requests) is not int or not 0 <= max_requests <= 500:
        raise ValueError("HN resolution permits at most 500 requests")
    if type(workers) is not int or not 1 <= workers <= 4:
        raise ValueError("HN resolution permits one to four workers")
    if not isinstance(capture, dict):
        capture = read_json(capture)
    if not isinstance(capture.get("records"), list) or len(capture["records"]) > 1000:
        raise ValueError("A bounded operating source capture is required")
    out = Path(out)
    if out.exists():
        raise FileExistsError("Thread capture already exists; refusing to overwrite")
    result = copy.deepcopy(capture)
    report = {"schema_version": "operating-thread-resolution-v1", "started_at": utc_now(),
              "input_sha256": hashlib.sha256(canonical_json(capture)).hexdigest(),
              "live": live, "max_requests": max_requests, "max_hops": MAX_HOPS,
              "workers": workers, "requests": 0, "cache_hits": 0, "lookups": [],
              "status": "completed", "limits": [
                  "Parent relationships are observed now and do not prove historical text or thread membership.",
                  "Previously captured roots are reused as source assertions; this is not a full revalidation of existing roots.",
                  "Missing/deleted items, conflicting types, cycles and exhausted limits stay unresolved."]}
    nodes, targets = {}, set()
    for ref in _references(capture):
        native, parent, root = _native(ref.get("record_id")), _native(ref.get("parent_id")), _native(ref.get("thread_id"))
        if not native:
            continue
        if not root:
            targets.add(native)
        provenance = {"source": "original_capture", "record_id": ref["record_id"],
                      "captured_at": ref.get("captured_at"), "parent_id": ref.get("parent_id"),
                      "thread_id": ref.get("thread_id")}
        proposed = ({"root": root, "provenance": provenance} if root else
                    {"parent": parent, "provenance": provenance} if parent else None)
        if proposed is not None:
            previous = nodes.get(native)
            if previous is None or proposed.get("root"):
                nodes[native] = proposed
    checked_cache, stop_network = set(), False
    with ThreadPoolExecutor(max_workers=workers) as executor:
        while True:
            traces = {native: _trace(native, nodes) for native in sorted(targets, key=int)}
            needed = sorted({trace["needed"] for trace in traces.values() if trace["status"] == "needs_item"}, key=int)
            if not needed:
                break
            found_cache = False
            for native in needed:
                if native in checked_cache:
                    continue
                checked_cache.add(native)
                row = store.db.execute("SELECT * FROM observations WHERE source_url=? AND kind=? ORDER BY id DESC LIMIT 1", (_url(native), KIND)).fetchone()
                if row:
                    fetched = {"status": row["status"], "body": (store.blobs / row["sha256"]).read_bytes(),
                               "captured_at": row["original_captured_at"] or row["observed_at"],
                               "observation_id": row["id"], "response_sha256": row["sha256"], "cache_hit": True}
                    nodes[native] = _validate_item(native, fetched)
                    report["lookups"].append(nodes[native]["provenance"])
                    report["cache_hits"] += 1
                    found_cache = True
            if found_cache:
                continue
            if not live or stop_network or report["requests"] >= max_requests:
                report["status"] = "offline_unresolved" if not live else "source_refusal" if stop_network else "request_limit"
                break
            batch = needed[:min(workers, max_requests - report["requests"])]
            # Network runs concurrently; SQLite archival remains in this thread.
            for native, fetched in zip(batch, executor.map(_fetch_hn_item, batch)):
                report["requests"] += 1
                fetched["observation_id"] = store.observe(_url(native), fetched["body"], kind=KIND,
                    status=fetched["status"], original_captured_at=fetched["captured_at"])
                fetched["response_sha256"] = hashlib.sha256(fetched["body"]).hexdigest()
                fetched["cache_hit"] = False
                nodes[native] = _validate_item(native, fetched)
                report["lookups"].append(dict(nodes[native]["provenance"], error=nodes[native].get("error")))
                stop_network = stop_network or fetched["status"] in {401, 403, 429}
    traces = {native: _trace(native, nodes) for native in sorted(targets, key=int)}
    for ref in _references(result):
        native = _native(ref.get("record_id"))
        if native not in targets or ref.get("thread_id"):
            continue
        trace = traces[native]
        annotation = {"status": trace["status"], "original_thread_id": ref.get("thread_id"),
                      "original_thread_resolution": ref.get("thread_resolution"), "chain": trace["chain"]}
        if trace["status"] == "resolved":
            ref["thread_id"] = "hackernews:" + trace["root"]
            ref["thread_resolution"] = "root"
            annotation["root_id"] = ref["thread_id"]
        else:
            annotation["unresolved_reason"] = report["status"] if trace["status"] == "needs_item" else trace["status"]
        ref["root_provenance"] = annotation
    report.update(finished_at=utc_now(), missing_native_roots_before=len(targets),
                  resolved_native_roots=sum(t["status"] == "resolved" for t in traces.values()),
                  unresolved_native_roots=sum(t["status"] != "resolved" for t in traces.values()),
                  unresolved=[{"record_id": "hackernews:" + native, "reason": trace["status"] if trace["status"] != "needs_item" else report["status"]}
                              for native, trace in traces.items() if trace["status"] != "resolved"])
    result["thread_resolution_run"] = report
    write_new_json(out, result)
    return result
