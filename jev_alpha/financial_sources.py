"""Assemble chronological, hash-checked financial source references offline.

The output is an evidence inventory, not a model prompt or frozen signal plan.
It deliberately cannot authorize price collection. Original source text stays
in the local archive; missing sources remain visible for every census event.
"""
from __future__ import annotations

import argparse
import hashlib
from datetime import date
from pathlib import Path

from .experiment import digest
from .store import read_json, utc_now, write_new_json


def file_ref(root: Path, path: str, expected: str | None = None, *, allow_empty: bool = False) -> dict:
    """Verify bytes without accepting paths outside the explicit archive root."""
    root = root.resolve()
    target = (root / path).resolve()
    if not target.is_relative_to(root):
        raise ValueError("Source path escapes the archive root")
    raw = target.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if (not raw and not allow_empty) or (expected is not None and actual != expected):
        raise ValueError(f"Missing content or source hash mismatch: {path}")
    return {"path": target.relative_to(root).as_posix(), "sha256": actual}


def verify_linked_files(root: Path, value: object) -> set[str]:
    """Check recorded archive references, including preserved failed responses."""
    verified = set()
    if isinstance(value, dict):
        for key, child in value.items():
            hash_key = key[:-5] + "_sha256" if key.endswith("_path") else ("sha256" if key == "path" else None)
            if hash_key in value and isinstance(child, str) and isinstance(value[hash_key], str):
                verified.add(file_ref(root, child, value[hash_key], allow_empty=True)["path"])
            verified.update(verify_linked_files(root, child))
    elif isinstance(value, list):
        for child in value:
            verified.update(verify_linked_files(root, child))
    return verified


def unique(rows: list, key: str) -> dict:
    result = {}
    for row in rows:
        value = row[key]
        if not isinstance(value, str) or not value or value in result:
            raise ValueError(f"Missing or duplicate {key}")
        result[value] = row
    return result


def iso_date(value: str) -> str:
    if date.fromisoformat(value).isoformat() != value:
        raise ValueError("Expected an ISO calendar date")
    return value


def document(root: Path, row: dict, issuer: str) -> dict:
    micron = issuer == "Micron"
    return {
        "event_id": row["event_id"], "issuer": issuer,
        "release_date": iso_date(row["publication_date_displayed"] if micron else row["source_release_date"]),
        "source_url": row["requested_url"] if micron else row["source_url"],
        "document_kind": "prepared_remarks_without_qa" if micron else "earnings_conference_transcript",
        "raw": file_ref(root, row["body_path"] if micron else row["raw_path"],
                        row["body_sha256"] if micron else row["raw_sha256"]),
        "text": file_ref(root, row["full_text_path"] if micron else row["text_path"],
                         row["full_text_sha256"] if micron else row["text_sha256"]),
        "pages": file_ref(root, row["pages_path"], row["pages_sha256"]),
        "page_count": row["page_count"],
        "captured_at": row["capture_finished_at"] if micron else row["first_captured_at"],
        "exact_historical_availability_verified": (
            row.get("historical_exact_version_available_at_release_verified") is True if micron
            else row.get("exact_historical_body_availability_verified") is True),
        "post_call_amendment_pages": sorted({r["page_number"] for r in row.get("explicit_after_call_amendments", [])}),
    }


def compile_packets(*, root: Path, census_path: str, micron_path: str,
                    tsmc_path: str, exposure_path: str, interval_paths: list[str],
                    supplemental_paths: list[str] | None = None) -> dict:
    supplemental_paths = supplemental_paths or []
    paths = [census_path, micron_path, tsmc_path, exposure_path, *interval_paths, *supplemental_paths]
    artifacts = {p: file_ref(root, p) for p in paths}
    census, micron, tsmc, exposure = [read_json(root / p) for p in paths[:4]]
    events = census["events"]
    event_index = unique(events, "event_id")
    if census["event_count"] != len(events) or events != sorted(events, key=lambda r: r["release_date"]):
        raise ValueError("The complete census must retain its chronological order")
    # Capture tools stored byte hashes; the exposure audit explicitly stored a
    # canonical JSON hash. Do not conflate those two representations.
    census_hash = artifacts[census_path]["sha256"]
    if (micron["census_sha256"] != census_hash
            or tsmc["frozen_event_census_sha256"] != census_hash
            or exposure["census_canonical_sha256"] != digest(census)):
        raise ValueError("A source manifest refers to a different event census")
    docs = unique([document(root, r, "Micron") for r in micron["sources"]]
                  + [document(root, r, "TSMC") for r in tsmc["documents"]], "event_id")
    for eid, row in docs.items():
        if eid in event_index and any(row[k] != event_index[eid][k] for k in ("issuer", "release_date")):
            raise ValueError("Source identity/date differs from frozen census")

    supplemental = {}
    supplemental_files = set()
    expected_dates = {eid: row["release_date"] for eid, row in event_index.items()}
    expected_dates.update({"tsmc-2024-q3": "2024-10-17", "micron-fy2025-q1": "2024-12-18"})
    for path in supplemental_paths:
        manifest = read_json(root / path)
        supplemental_files.update(verify_linked_files(root, manifest))
        for index, row in enumerate(manifest["slides"]):
            sid, eid = row["source_id"], row["event_id"]
            if sid in supplemental or eid not in expected_dates or row["displayed_release_date"] != expected_dates[eid]:
                raise ValueError("Supplemental source identity/date mismatch or duplicate")
            supplemental[sid] = {"event_id": eid, "source_url": row["url"],
                                 "manifest": {**artifacts[path], "json_pointer": f"/slides/{index}"},
                                 "raw": file_ref(root, row["body_path"], row["body_sha256"]),
                                 "pages": file_ref(root, row["pages_path"], row["pages_sha256"]),
                                 "page_count": row["page_count"],
                                 "exact_historical_availability_verified": row.get("historical_exact_version_availability_verified") is True,
                                 "visual_table_verification_complete": row.get("table_layout_or_image_only_content_visually_verified") is True}

    intervals = {}
    interval_files = set()
    for path in interval_paths:
        interval_manifest = read_json(root / path)
        interval_files.update(verify_linked_files(root, interval_manifest))
        for index, row in enumerate(interval_manifest["intervals"]):
            eid = row["event_id"]
            if eid not in event_index or eid in intervals:
                raise ValueError("Interval census has an unexpected or duplicate event")
            intervals[eid] = (row, {**artifacts[path], "json_pointer": f"/intervals/{index}"})

    mappings = unique(exposure["event_mapping"], "event_id")
    if set(mappings) != set(event_index):
        raise ValueError("Supplier mapping must retain exactly the full event census")
    versions = unique(exposure["annual_exposure_versions"], "exposure_version_id")
    supplements = unique(exposure["relationship_supplements"], "source_id")
    supplier_sources = unique(exposure["sources"], "source_id")
    used_supplier_sources = {}
    # Fixed predecessor identities avoid silently using an older quarter if the
    # required predecessor's source file is missing.
    previous = {"TSMC": "tsmc-2024-q3", "Micron": "micron-fy2025-q1"}
    prior_dates = {"TSMC": "2024-10-17", "Micron": "2024-12-18"}
    packets = []
    for event in events:
        eid, issuer = event["event_id"], event["issuer"]
        release = iso_date(event["release_date"])
        prior_id, prior_date = previous[issuer], prior_dates[issuer]
        current, prior = docs.get(eid), docs.get(prior_id)
        if prior_date >= release:
            raise ValueError("Predecessor must strictly precede the event")
        if prior and (prior["issuer"] != issuer or prior["release_date"] != prior_date):
            raise ValueError("Predecessor source identity/date mismatch")
        gaps = []
        if current is None:
            gaps.append("current_document_missing")
        if prior is None:
            gaps.append("required_predecessor_document_missing")
        if issuer == "Micron":
            gaps.append("prepared_remarks_do_not_include_earnings_qa")
        interval_ref = None
        if eid not in intervals:
            gaps.append("intervening_disclosure_census_missing")
        else:
            interval, interval_ref = intervals[eid]
            if (interval["interval_after_date"] != prior_date
                    or interval["interval_before_date"] != release):
                raise ValueError("Intervening interval does not match source pair")
            if interval.get("complete_public_prior_state_reconciliation") is not True:
                gaps.append("intervening_guidance_reconciliation_incomplete")
            # Keep the complete private interval record addressable, including
            # its conference, presentation and boundary-day content gaps.
        mapping = mappings[eid]
        if mapping["issuer"] != issuer or mapping["release_date"] != release:
            raise ValueError("Supplier mapping identity/date mismatch")
        version = versions[mapping["annual_exposure_version_id"]]
        if iso_date(version["usable_from_date_conservative"]) > release:
            raise ValueError("Future supplier filing would leak into an earlier event")
        superseded = version.get("superseded_from_date_in_this_annual_pair")
        if superseded is not None and release >= iso_date(superseded):
            raise ValueError("Supplier filing was superseded before this event")
        selected_supplier_ids = sorted(set([version["source_id"], *mapping["dated_relationship_source_ids"]]))
        for sid in selected_supplier_ids:
            if sid != version["source_id"]:
                if sid not in supplements or iso_date(supplements[sid]["usable_from_date_conservative"]) > release:
                    raise ValueError("Future or undated supplier relationship source")
            source = supplier_sources[sid]
            used_supplier_sources[sid] = {
                "source_url": source["requested_url"],
                "raw": file_ref(root, source["body_path"], source["body_sha256"]),
                "text": file_ref(root, source["text_path"], source["text_sha256"]),
            }
        if not current or not prior or not all(d["exact_historical_availability_verified"] for d in (current, prior)):
            gaps.append("exact_historical_document_availability_unverified")
        if mapping.get("exact_historical_version_availability_verified") is not True:
            gaps.append("exact_historical_supplier_source_availability_unverified")
        supplement_ids = [sid for sid, row in supplemental.items() if row["event_id"] in (eid, prior_id)]
        if any(not supplemental[sid]["exact_historical_availability_verified"] for sid in supplement_ids):
            gaps.append("exact_historical_supplemental_availability_unverified")
        if any(not supplemental[sid]["visual_table_verification_complete"] for sid in supplement_ids):
            gaps.append("supplemental_tables_and_images_need_visual_review")
        packets.append({
            "event_id": eid, "issuer": issuer, "release_date": release,
            "current_document_id": eid if current else None,
            "prior_document_id": prior_id if prior else None,
            "required_prior_document_id": prior_id,
            "current_supplemental_document_ids": [sid for sid in supplement_ids if supplemental[sid]["event_id"] == eid],
            "prior_supplemental_document_ids": [sid for sid in supplement_ids if supplemental[sid]["event_id"] == prior_id],
            "intervening_interval": interval_ref,
            "supplier_document_ids": selected_supplier_ids,
            "qualitative_supplier_relationship_supported": mapping.get("qualitative_historical_supplier_relationship_supported") is True,
            "quantitative_customer_revenue_weight_available": mapping.get("customer_specific_revenue_weight_pct") is not None,
            "gaps": gaps,
            "economic_conditions_evaluated": False,
            "signals_frozen": False, "ready_for_price_collection": False,
        })
        previous[issuer], prior_dates[issuer] = eid, release
    return {
        "schema_version": "read-through-source-packets-v1", "created_at": utc_now(),
        "artifacts": artifacts, "documents": list(docs.values()),
        "supplemental_documents": supplemental,
        "supplier_documents": used_supplier_sources, "packets": packets,
        "verified_interval_file_count": len(interval_files),
        "verified_supplemental_file_count": len(supplemental_files),
        "packets_sha256": digest(packets), "event_count": len(packets),
        "prices_used": False, "models_called": False, "ready_for_price_collection": False,
        "purpose": "Verified source references for all events; not prompts, economic labels or signals.",
        "source_text_policy": "Read original full text via verified references; no keyword-based event or passage exclusion.",
        "readiness_limit": "Supplemental sources need explicit reconciliation; a download or zero keyword hits cannot close a content gap.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    for arg in ("census", "micron", "tsmc", "exposure"):
        parser.add_argument(f"--{arg}", required=True)
    parser.add_argument("--interval", action="append", default=[])
    parser.add_argument("--supplement", action="append", default=[])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = compile_packets(root=args.root, census_path=args.census, micron_path=args.micron,
                             tsmc_path=args.tsmc, exposure_path=args.exposure, interval_paths=args.interval,
                             supplemental_paths=args.supplement)
    write_new_json(args.out, result)
    print(f"Assembled {result['event_count']} events; price collection remains disabled.")


if __name__ == "__main__":
    main()
