"""Frozen operating-evidence cohorts and independently reviewed feasibility gates.

Source reports are not verified customer transactions. This module has no trade
signal or order interface, and never converts interpretation scores into returns.
"""
from __future__ import annotations

from datetime import datetime, timezone
import copy
import hashlib
import math
import re
import unicodedata
from urllib.parse import urlsplit

from .experiment import digest
from .store import utc_now


def _time(value: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError("An aware source timestamp is required")
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("Naive source timestamps cannot establish chronology")
    return result.astimezone(timezone.utc)


def _body_key(text: str) -> str:
    return hashlib.sha256(re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip().encode()).hexdigest()


def _hn_chain_links(ref: dict, source: str) -> set[tuple[str, str]]:
    """Retain anchored, validated ancestry even when its final root is missing."""
    def native(value):
        match = re.fullmatch(r"hackernews:([1-9]\d*)", value) if isinstance(value, str) else None
        return match[1] if match else None

    current = native(ref.get("record_id"))
    provenance = ref.get("root_provenance")
    if source != "hackernews" or not current or not isinstance(provenance, dict):
        return set()
    chain = provenance.get("chain")
    if not isinstance(chain, list):
        return set()
    links, seen = set(), set()
    for step in chain:
        if not isinstance(step, dict) or current in seen or step.get("error"):
            break
        seen.add(current)
        if step.get("source") == "official_hn_api":
            if (step.get("native_id") != current or type(step.get("status")) is not int
                    or step["status"] != 200 or type(step.get("observation_id")) is not int
                    or step["observation_id"] <= 0
                    or not isinstance(step.get("response_sha256"), str)
                    or not re.fullmatch(r"[0-9a-f]{64}", step["response_sha256"])
                    or step.get("url") != f"https://hacker-news.firebaseio.com/v0/item/{current}.json"):
                break
            if step.get("item_type") in {"story", "job", "poll"} and not step.get("parent_id"):
                break
            if step.get("item_type") not in {"comment", "pollopt"}:
                break
            parent = native(step.get("parent_id"))
        elif step.get("source") == "original_capture":
            if native(step.get("record_id")) != current:
                break
            root = native(step.get("thread_id"))
            if root:
                links.add((source, root))
                break
            parent = native(step.get("parent_id"))
        else:
            break
        if not parent or parent == current or parent in seen:
            break
        links.add((source, parent))
        current = parent
    return links


def _source_links(row: dict) -> tuple[set[tuple[str, str]], bool]:
    """Keep every identity/thread/parent from merged copies, not just the first."""
    references = row.get("source_references", [])
    if not isinstance(references, list) or any(not isinstance(r, dict) for r in references):
        raise ValueError("Invalid merged source references")
    links, unresolved = set(), False
    for ref in [row, *references]:
        rid = ref.get("record_id")
        if not isinstance(rid, str) or not rid:
            raise ValueError("Merged source reference requires a record identity")
        source = ref.get("source") or (rid.split(":", 1)[0] if ":" in rid else row.get("source"))
        if not isinstance(source, str) or not source:
            raise ValueError("Source namespace is required")
        for field in ("record_id", "thread_id", "parent_id"):
            identity = ref.get(field)
            if identity:
                value = str(identity)
                links.add((source, value.removeprefix(source + ":")))
        links.update(_hn_chain_links(ref, source))
        unresolved |= not ref.get("thread_id") or ref.get("thread_resolution") in {"unknown", "parent_only", "incomplete"}
    return links, unresolved


def freeze_operating_cohort(protocol: dict, capture: dict) -> dict:
    """Freeze source records and cluster splits before opening model answers.

An exact body copied between threads connects the threads transitively. Missing
root linkage is retained as a limit on independence, never silently certified.
"""
    if protocol.get("schema_version") != "operating-changes-protocol-v1":
        raise ValueError("Unsupported operating protocol")
    if capture.get("protocol_sha256") != digest(protocol):
        raise ValueError("Source capture does not match the registered protocol")
    records = capture.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("A nonempty source capture is required")
    if len(records) > protocol["max_records"]:
        raise ValueError("Source cohort exceeds the frozen record cap")
    vendor_ids = {v["vendor_id"] for v in protocol["vendors"]}
    start, end = (_time(protocol["window"][k]) for k in ("start", "end_exclusive"))
    fraction = protocol.get("development_fraction")
    if type(fraction) not in (int, float) or not math.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError("Invalid development split fraction")
    if not isinstance(protocol.get("selection_seed"), str) or not protocol["selection_seed"]:
        raise ValueError("A frozen split seed is required")
    by_id, parents, body_owner, link_owner, unresolved_ids = {}, {}, {}, {}, set()

    def find(key):
        while parents[key] != key:
            parents[key] = parents[parents[key]]
            key = parents[key]
        return key

    def union(left, right):
        a, b = find(left), find(right)
        parents[max(a, b)] = min(a, b)

    for row in records:
        rid = row.get("record_id")
        if not isinstance(rid, str) or not rid or rid in by_id:
            raise ValueError("Duplicate or invalid source record ID")
        text = row.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Source text must be preserved, not empty")
        if hashlib.sha256(text.encode()).hexdigest() != row.get("content_sha256"):
            raise ValueError("Source text digest mismatch")
        if not start <= _time(row["published_at"]) < end:
            raise ValueError("Source record lies outside the frozen publication window")
        if _time(row["captured_at"]) < _time(row["published_at"]):
            raise ValueError("Source capture cannot precede publication")
        if row.get("state_observed_at") is not None:
            _time(row["state_observed_at"])
        found = row.get("vendor_ids")
        if not isinstance(found, list) or not found or len(set(found)) != len(found) or not set(found) <= vendor_ids:
            raise ValueError("Unknown or duplicate vendor membership")
        source_url = urlsplit(row.get("url", ""))
        if source_url.scheme != "https" or not source_url.hostname or source_url.username or source_url.password:
            raise ValueError("Public source provenance URL is required")
        parents[rid] = rid
        by_id[rid] = copy.deepcopy(row)
        by_id[rid]["vendor_ids"] = sorted(found)
        if "source_references" in by_id[rid]:
            by_id[rid]["source_references"] = sorted(by_id[rid]["source_references"], key=digest)
        body = _body_key(text)
        if body in body_owner:
            union(rid, body_owner[body])
        body_owner[body] = rid
        links, unresolved = _source_links(row)
        if unresolved:
            unresolved_ids.add(rid)
        for link in sorted(links):
            if link in link_owner:
                union(rid, link_owner[link])
            link_owner[link] = rid
    components = {}
    for rid in by_id:
        components.setdefault(find(rid), []).append(rid)
    splits, cluster_ids = {}, {}
    threshold = int(protocol["development_fraction"] * 10_000)
    for members in components.values():
        group = digest(sorted(members))
        fraction = int(hashlib.sha256((protocol["selection_seed"] + ":split:" + group).encode()).hexdigest()[:16], 16) % 10_000
        split = "development" if fraction < threshold else "evaluation"
        for rid in members:
            splits[rid], cluster_ids[rid] = split, group
    ordered = [by_id[rid] for rid in sorted(by_id)]
    unresolved = sorted(unresolved_ids)
    return {
        "schema_version": "operating-cohort-v1", "created_at": utc_now(),
        "experiment_id": protocol["experiment_id"], "protocol_sha256": digest(protocol),
        "capture_sha256": digest(capture), "records": ordered, "splits": splits,
        "cluster_ids": cluster_ids, "cluster_count": len(components),
        "unresolved_thread_record_ids": unresolved,
        "source_status": capture.get("status"), "source_coverage": copy.deepcopy(capture.get("coverage", capture.get("queries", []))),
        "development_count": sum(s == "development" for s in splits.values()),
        "evaluation_count": sum(s == "evaluation" for s in splits.values()),
        "selection_frozen_before_models": True, "historical_alpha_eligible": False,
        "interpretation": "Split protects known threads and exact copied bodies. Semantic duplication and incomplete source roots still require review. These historical current-version records are not a predictive holdout."
    }


def review_sample(cohort: dict, selected_ids: set[str], size: int = 80, *, split: str | None = None,
                  seed: str = "operating-reject-audit-v1", unknown_ids: set[str] | None = None) -> dict:
    if split not in {None, "development", "evaluation"}:
        raise ValueError("Unsupported review split")
    ids = {r["record_id"] for r in cohort["records"] if split is None or cohort["splits"][r["record_id"]] == split}
    unknown_ids = unknown_ids or set()
    if (not isinstance(selected_ids, set) or not isinstance(unknown_ids, set) or not selected_ids | unknown_ids <= ids
            or type(size) is not int or size < 0 or not isinstance(seed, str) or not seed):
        raise ValueError("Invalid review population")
    retained = selected_ids | unknown_ids
    rejects = sorted(ids - retained, key=lambda rid: (hashlib.sha256((seed + ":" + rid).encode()).hexdigest(), rid))[:size]
    return {"schema_version": "operating-review-sample-v1", "cohort_sha256": digest(cohort),
            "split": split, "reject_seed": seed, "requested_reject_sample_size": size,
            "selected_record_ids": sorted(selected_ids), "rejected_sample_ids": rejects,
            "unknown_retained_record_ids": sorted(unknown_ids),
            "review_record_ids": sorted(retained | set(rejects)),
            "full_population_reviewed": False, "full_population_in_review_sample": len(retained) + len(rejects) == len(ids),
            "label_origin": "independent_agent_review", "human_gold": False}


def evaluate_operating_reviews(protocol: dict, cohort: dict, review: dict,
                               arm_selected: dict[str, set[tuple[str, str]]], costs: dict | None = None,
                               *, arm_status: dict | None = None) -> dict:
    """Count independently resolved episodes, never treating missing work as negative.

    ``review.split`` optionally restricts the population to development/evaluation.
    ``arm_status[arm]`` must contain cohort_sha256, status='complete',
    response_coverage_complete=True, evaluated_pairs=[[record_id,vendor_id],...],
    and unknown_pairs=[...]. Every scoped pair must have been evaluated. Omitted
    arm_status remains callable for diagnostic counts but cannot pass the gate.
    Independently qualified unknown control records block an incremental claim;
    they are disclosed separately, not credited as selected control successes.
    """
    if cohort.get("protocol_sha256") != digest(protocol) or review.get("cohort_sha256") != digest(cohort):
        raise ValueError("Review/protocol/cohort identity mismatch")
    if review.get("label_origin") not in {"independent_agent_review", "human_review"}:
        raise ValueError("Explicit independent review origin is required")
    split = review.get("split")
    if split not in {None, "development", "evaluation"}:
        raise ValueError("Unsupported review split")
    records = {r["record_id"]: r for r in cohort["records"]
               if split is None or cohort["splits"][r["record_id"]] == split}
    if not records:
        raise ValueError("Empty review population")
    population = {(r["record_id"], v) for r in records.values() for v in r["vendor_ids"]}
    needed = {"keyword", "semantic_retrieval", "baseline", "jev"}
    if not isinstance(arm_selected, dict) or set(arm_selected) - needed:
        raise ValueError("Unknown comparison arm")
    if any(not isinstance(ids, set) or not ids <= population for ids in arm_selected.values()):
        raise ValueError("Arm selection contains an unknown record/vendor pair")
    if arm_status is not None and (not isinstance(arm_status, dict) or set(arm_status) - needed):
        raise ValueError("Unknown comparison arm status")
    arm_status = arm_status or {}
    unknowns, completed = {}, {}

    def pairs(values):
        if not isinstance(values, (list, set, tuple)):
            raise ValueError("Arm coverage pairs must be explicit")
        items = [tuple(pair[k] for k in ("record_id", "vendor_id"))
                 if isinstance(pair, dict) and set(pair) == {"record_id", "vendor_id"} else pair for pair in values]
        if any(not isinstance(pair, (list, tuple)) or len(pair) != 2 or any(not isinstance(v, str) for v in pair) for pair in items):
            raise ValueError("Invalid arm coverage pair")
        result = {tuple(pair) for pair in items}
        if len(result) != len(values) or not result <= population:
            raise ValueError("Duplicate or out-of-population arm coverage pair")
        return result

    for arm in needed:
        state = arm_status.get(arm)
        if state is None:
            unknowns[arm], completed[arm] = set(), False
            continue
        if not isinstance(state, dict) or state.get("cohort_sha256") != digest(cohort):
            raise ValueError("Arm status cohort identity mismatch")
        evaluated = pairs(state.get("evaluated_pairs", []))
        unknowns[arm] = pairs(state.get("unknown_pairs", []))
        if (not unknowns[arm] <= evaluated or not arm_selected.get(arm, set()) <= evaluated
                or unknowns[arm] & arm_selected.get(arm, set())):
            raise ValueError("Retained arm pair lacks evaluation coverage")
        completed[arm] = (state.get("status") == "complete" and state.get("response_coverage_complete") is True
                          and evaluated == population and arm in arm_selected)
    seen, episodes, qualified_ids, reviewed_ids, resolved_ids, by_record_episode = set(), {}, set(), set(), set(), {}
    body_episode = {}
    for row in review.get("records", []):
        rid, vendor = row.get("record_id"), row.get("vendor_id")
        key = (rid, vendor)
        if rid not in records or vendor not in records[rid]["vendor_ids"] or key in seen:
            raise ValueError("Review contains duplicate or unregistered record/vendor")
        seen.add(key)
        reviewed_ids.add(key)
        if row.get("qualifies") is not None and type(row.get("qualifies")) is not bool:
            raise ValueError("Review decision must be true, false, or unknown")
        if type(row.get("qualifies")) is bool:
            resolved_ids.add(key)
        if row.get("qualifies") is not True:
            continue
        required = ("firsthand_report", "paid_product", "production_use", "completed_or_active_change", "business_link", "timing_supported")
        if not all(row.get(field) is True for field in required):
            raise ValueError("Qualifying review lacks the economic or timing evidence")
        if (not isinstance(row.get("episode_id"), str) or not row["episode_id"].strip()
                or not isinstance(row.get("organization"), str) or not row["organization"].strip()):
            raise ValueError("A qualifying episode needs an organization and deduplication identity")
        evidence = row.get("evidence")
        if not isinstance(evidence, list) or not evidence:
            raise ValueError("A qualifying episode requires exact supporting source spans")
        own_source = False
        for span in evidence:
            if not isinstance(span, dict):
                raise ValueError("Invalid supporting span")
            source = records.get(span.get("record_id"))
            quote = span.get("quote")
            if not source or not isinstance(quote, str) or not quote.strip() or quote not in source["text"]:
                raise ValueError("Review quote is absent from frozen source text")
            if (vendor not in source["vendor_ids"]
                    or cohort["cluster_ids"][source["record_id"]] != cohort["cluster_ids"][rid]):
                raise ValueError("Review cannot borrow unrelated customer or vendor evidence")
            own_source |= source["record_id"] == rid
        if not own_source:
            raise ValueError("Qualifying pair requires evidence in its own source record")
        body = _body_key(records[rid]["text"])
        if body in body_episode and body_episode[body] != row["episode_id"]:
            raise ValueError("Copied source body cannot create multiple economic episodes")
        body_episode[body] = row["episode_id"]
        episode = episodes.setdefault(row["episode_id"], {"vendor_ids": set(), "record_ids": set()})
        episode["vendor_ids"].add(vendor)
        episode["record_ids"].add(rid)
        qualified_ids.add(key)
        by_record_episode.setdefault(key, set()).add(row["episode_id"])
    arms = {}
    for arm, selected in arm_selected.items():
        covered_episodes = set().union(*(by_record_episode.get(rid, set()) for rid in selected)) if selected else set()
        unknown_episodes = set().union(*(by_record_episode.get(rid, set()) for rid in unknowns[arm])) if unknowns[arm] else set()
        reviewed_selected = selected & reviewed_ids
        resolved_selected = selected & resolved_ids
        correct_selected = selected & qualified_ids
        arms[arm] = {"selected_records": len(selected), "reviewed_selected_records": len(reviewed_selected),
                     "selected_unreviewed_records": len(selected - reviewed_ids),
                     "selected_unresolved_records": len(selected - resolved_ids),
                     "resolved_selected_records": len(resolved_selected),
                     "qualifying_selected_records": len(correct_selected),
                     "precision_on_reviewed_records": len(correct_selected) / len(resolved_selected) if resolved_selected else None,
                     "precision_denominator": "independently resolved selected record/vendor pairs; unknown labels excluded",
                     "qualifying_episode_ids": sorted(covered_episodes),
                     "qualifying_episodes": len(covered_episodes), "run_complete": completed[arm],
                     "unknown_retained_pairs": sorted([list(pair) for pair in unknowns[arm]]),
                     "qualifying_unknown_retained_episode_ids": sorted(unknown_episodes)}
    vendor_count = len(set().union(*(e["vendor_ids"] for e in episodes.values()))) if episodes else 0
    controls = set().union(*(set(arms[a]["qualifying_episode_ids"]) for a in needed - {"jev"} if a in arms))
    control_unknown_episodes = set().union(*(set(arms[a]["qualifying_unknown_retained_episode_ids"]) for a in needed - {"jev"} if a in arms))
    jev_episodes = set(arms.get("jev", {}).get("qualifying_episode_ids", []))
    incremental = sorted(jev_episodes - controls - control_unknown_episodes)
    selected_union = set().union(*arm_selected.values()) if arm_selected else set()
    unknown_union = set().union(*unknowns.values())
    selected_record_ids = {rid for rid, _ in selected_union | unknown_union}
    selected_review_population = {pair for pair in population if pair[0] in selected_record_ids}
    complete_selected_review = selected_review_population <= resolved_ids
    review_spec = protocol.get("review", {})
    sample_size = review_spec.get("reject_sample_size", 80)
    sample = review_sample(cohort, {rid for rid, _ in selected_union}, sample_size, split=split,
                           seed=review_spec.get("reject_seed", "operating-reject-audit-v1"),
                           unknown_ids={rid for rid, _ in unknown_union})
    reject_ids = set(sample["rejected_sample_ids"])
    reject_population = {pair for pair in population if pair[0] in reject_ids}
    complete_reject_review = reject_population <= resolved_ids
    gate = protocol["feasibility_gate"]
    density_pass = len(episodes) >= gate["minimum_distinct_qualifying_episodes"] and vendor_count >= gate["minimum_vendors"]
    all_arms = needed <= arms.keys()
    all_runs = all(completed.values())
    coverage = cohort.get("source_coverage")
    source_complete = (cohort.get("source_status") == "collected" and isinstance(coverage, list) and bool(coverage)
        and all(isinstance(c, dict) and (c.get("quota") == 0 and c.get("status") == "not_requested"
                or c.get("status") == "complete_query" and c.get("completeness") == "complete"
                and c.get("deadline_partial") is False and c.get("truncated") is False
                and c.get("invalid_rows", 0) == 0) for c in coverage))
    if protocol.get("source_quotas"):
        expected_strata = {(v["vendor_id"], s) for v in protocol["vendors"] for s in protocol["source_quotas"]}
        actual_strata = [(c.get("vendor_id"), c.get("source")) for c in coverage] if isinstance(coverage, list) else []
        source_complete = (source_complete and set(actual_strata) == expected_strata and len(actual_strata) == len(expected_strata)
            and all(type(c.get("quota")) is int and c["quota"] == protocol["source_quotas"][c["source"]] for c in coverage))
    # Costs are descriptive; lower API fees alone cannot prove total savings.
    complete = complete_selected_review and complete_reject_review and all_arms and all_runs and source_complete
    source_decision = ("incomplete" if not source_complete or resolved_ids != population
                       else "eligible_for_comparison" if density_pass else "stop_family")
    advantage = bool(incremental) and complete
    if not complete:
        decision = "incomplete"
    elif not density_pass:
        decision = "stop_family"
    elif advantage:
        decision = "eligible_for_financial_hypothesis_registration"
    else:
        decision = "no_demonstrated_jev_advantage"
    return {"schema_version": "operating-feasibility-review-v1", "created_at": utc_now(),
            "protocol_sha256": digest(protocol), "cohort_sha256": digest(cohort), "review_sha256": digest(review),
            "decision": decision, "qualifying_episodes": len(episodes), "qualifying_vendors": vendor_count,
            "source_feasibility_decision": source_decision,
            "density_gate_passed": density_pass, "all_controls_present": all_arms,
            "all_arm_runs_complete": all_runs, "source_capture_complete": source_complete,
            "selected_review_complete": complete_selected_review, "reject_review_complete": complete_reject_review,
            "required_rejected_sample_ids": sample["rejected_sample_ids"], "required_reject_sample_size": len(reject_ids),
            "jev_incremental_episode_ids": incremental if complete else [],
            "unconfirmed_incremental_episode_ids": incremental if not complete else [],
            "control_unknown_qualifying_episode_ids": sorted(control_unknown_episodes),
            "comparison_complete": complete, "split": split,
            "arms": arms, "costs": costs or {}, "total_review_time_measured": False,
            "reviewed_records": len({r for r, _v in reviewed_ids}), "reviewed_record_vendor_pairs": len(reviewed_ids),
            "cohort_records": len(records), "metric_unit": "record_vendor_pair",
            "label_origin": review["label_origin"], "human_gold": review["label_origin"] == "human_review",
            "unresolved_review_pairs": len(reviewed_ids - resolved_ids),
            "full_population_review_resolved": resolved_ids == population,
            "full_recall_established": complete and resolved_ids == population and bool(qualified_ids),
            "episodes": [{"episode_id": eid, "vendor_ids": sorted(e["vendor_ids"]), "record_ids": sorted(e["record_ids"])} for eid, e in sorted(episodes.items())],
            "alpha_proven": False, "trade_eligible": False,
            "interpretation": "An interpretation/density gate only. No measured return, representative customer-spend estimate, or calibrated trade probability. Agent labels remain fallible; partial reject review cannot establish full recall."}
