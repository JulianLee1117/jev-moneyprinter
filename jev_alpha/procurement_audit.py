"""Price-blind development selection and equal-evidence error accounting."""
from collections import Counter
import math
from pathlib import Path

from .store import utc_now, write_new_json
from .procurement import ARMS, _categories, digest, packet_request, rules_answers, signal_for
from .procurement_models import MODELS, build_chat_request, _valid_serving_model


def development_entries(inputs, limit=48):
    if type(limit) is not int or limit < 3:
        raise ValueError("At least three development packets are required")
    eligible = [e for e in inputs["records"] if e["split"] == "development" and e.get("model_required", True)]
    return sorted(eligible, key=lambda e: digest(["concurrency-v1", e["packet_id"]]))[:limit]


def _number(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def _entries(inputs):
    rows = {e["packet_id"]: e for e in inputs["records"]}
    if len(rows) != len(inputs["records"]):
        raise ValueError("Frozen packet IDs must be unique")
    return rows


def _labels(inputs, review, *, development_only=False):
    if (review.get("price_blind") is not True or review.get("inputs_sha256") != digest(inputs)
            or review.get("label_origin") != "independent_source_review"):
        raise ValueError("Review must be price-blind, input-bound and explicitly independently source-labeled")
    entries, labels, seen = _entries(inputs), {}, set()
    for row in review.get("records", []):
        pid = row.get("packet_id")
        if pid not in entries or pid in seen:
            raise ValueError("Review contains duplicate or foreign packets")
        seen.add(pid)
        if row.get("status") != "reviewed":
            continue
        entry = entries[pid]
        if development_only and entry["split"] != "development":
            raise ValueError("Holdout labels cannot participate in development profile selection")
        answers, evidence = row.get("source_label"), row.get("evidence")
        request = packet_request(entry)
        if (not isinstance(answers, dict) or set(answers) != set(request["questions"])
                or not evidence or not isinstance(evidence, (str, dict, list))
                or isinstance(evidence, str) and not evidence.strip()):
            raise ValueError("Completed source review requires the full categorical panel and source evidence")
        if any(value not in (None, "unresolved") and (not isinstance(value, str)
                or value not in request["questions"][name]["criteria"]) for name, value in answers.items()):
            raise ValueError("Source label contains an invalid category")
        if row.get("source_signal", "unresolved") not in {"eligible", "ineligible", "unresolved"}:
            raise ValueError("Invalid independent source-signal label")
        if row.get("source_signal") == "eligible" and (row.get("project_identity_verified") is not True
                or not row.get("project_id") or answers.get("recipient") in {None, "unresolved", "none", "unknown"}):
            raise ValueError("Eligible source labels require a verified project and identified recipient")
        if row.get("review_seconds") is not None and not _number(row["review_seconds"]):
            raise ValueError("Measured review time must be finite and nonnegative")
        labels[pid] = row
    return labels


def _bound_row(row, entry, arm):
    request = packet_request(entry)
    body = request if arm == "jev" else build_chat_request(request, MODELS[arm])
    if row.get("input_sha256") != digest(request["state"]) or row.get("request_hash") != digest(body):
        raise ValueError("Run evidence, questions or model differ from frozen inputs")
    if row.get("status") == "not_required" and entry.get("model_required") is not False:
        raise ValueError("Only an explicitly exempt frozen input can skip inference")
    if row.get("status") == "not_required" and any(
            field in entry and row.get(field) != entry[field]
            for field in ("skip_inference_reason", "deterministic_signal_status")):
        raise ValueError("Inference exemption reason/status differs from the frozen input")
    if row.get("status") == "completed":
        if not _valid_serving_model(row.get("model"), MODELS[arm]):
            raise ValueError("Completed response serving model differs from its arm")
        answers = _categories(row)
        if set(answers) != set(request["questions"]) or any(
                not isinstance(value, str) or value not in request["questions"][name]["criteria"] for name, value in answers.items()):
            raise ValueError("Completed response lacks the complete allowed categorical panel")


def _fresh(row):
    timing = row.get("timing_ms")
    return (row.get("status") == "completed" and row.get("latency_eligible") is True
            and row.get("cached") is False and isinstance(timing, dict)
            and all(_number(timing.get(k)) for k in ("queue", "inference", "validation", "end_to_end"))
            and sum(timing[k] for k in ("queue", "inference", "validation")) <= timing["end_to_end"] + 1)


def _accuracy(rows, labels):
    matched = adjudicated = exact = scored = unresolved = 0
    for row in rows:
        gold = labels[row["packet_id"]]["source_label"]
        targets = {k: v for k, v in gold.items() if v not in (None, "unresolved")}
        unresolved += len(gold) - len(targets)
        prediction = _categories(row) if row.get("status") == "completed" else {}
        adjudicated += len(targets)
        matched += sum(prediction.get(k) == v for k, v in targets.items())
        if targets:
            scored += 1
            exact += all(prediction.get(k) == v for k, v in targets.items())
    return {"adjudicated_categories": adjudicated, "matched_categories": matched,
            "category_accuracy": matched / adjudicated if adjudicated else None,
            "exact_match_scored_packets": scored, "exact_match_packets": exact,
            "exact_match_accuracy": exact / scored if scored else None, "unresolved_gold_categories": unresolved}


def freeze_profile(root: Path, inputs, benchmarks, review, protocol):
    """Freeze throughput selection after independently labeled development batches."""
    if inputs.get("protocol_sha256") != digest(protocol) or (root / "market").exists():
        raise ValueError("Profile requires registered inputs and must precede market-data collection")
    by_id, labels = _entries(inputs), _labels(inputs, review, development_only=True)
    profile = {"schema_version": "procurement-profile-v1", "frozen_at": utc_now(),
        "inputs_sha256": digest(inputs), "protocol_sha256": digest(protocol), "review_sha256": digest(review),
        "arms": {}, "market_arrival_claim": False, "alpha_proven": False,
        "selection": "Lowest batch wall milliseconds per fresh validated packet. Source accuracy is disclosed, not silently treated as a profitability gate."}
    shared_allocation = None
    expected = digest({**protocol, "development_only": True})
    for benchmark in benchmarks:
        plan, runs = benchmark["plan"], benchmark["runs"]
        arm = plan["arm"]
        if arm not in ARMS[1:] or arm in profile["arms"] or plan.get("development_only") is not True or plan.get("protocol_sha256") != expected:
            raise ValueError("One registered development-only benchmark per model arm is required")
        allocation = plan["groups"]
        if set(allocation) != {"1", "4", "8"} or set(runs) != set(allocation) or any(not group for group in allocation.values()):
            raise ValueError("All three nonempty concurrency groups are required")
        planned_ids = [p["packet_id"] for group in allocation.values() for p in group]
        if (len(set(planned_ids)) != len(planned_ids) or any(pid not in by_id or by_id[pid]["split"] != "development"
                or not by_id[pid].get("model_required", True) for pid in planned_ids)):
            raise ValueError("Benchmark groups must be disjoint required development packets")
        if set(planned_ids) != {e["packet_id"] for e in development_entries(inputs, len(planned_ids))}:
            raise ValueError("Benchmark packet selection differs from the frozen hash-selected development pool")
        hashes = {pid: digest(packet_request(by_id[pid])) for pid in planned_ids}
        ordered = sorted((by_id[pid] for pid in planned_ids), key=lambda e: (hashes[e["packet_id"]], e["packet_id"]))
        for index, workers in enumerate((1, 4, 8)):
            if [p["packet_id"] for p in allocation[str(workers)]] != [e["packet_id"] for e in ordered[index::3]]:
                raise ValueError("Benchmark groups differ from the declared disjoint hash allocation")
        if shared_allocation is not None and allocation != shared_allocation:
            raise ValueError("Every model must use the identical development allocation")
        shared_allocation = allocation
        choices, audit = [], {}
        for worker, run in runs.items():
            if (run.get("arm") != arm or run.get("model") != MODELS[arm] or run.get("workers") != int(worker)
                    or run.get("protocol_sha256") != expected):
                raise ValueError("Benchmark run differs from its declared model, protocol or concurrency")
            planned = allocation[worker]
            if [r["packet_id"] for r in run["records"]] != [p["packet_id"] for p in planned]:
                raise ValueError("Benchmark run must preserve complete ordered group coverage")
            for row, planned_row in zip(run["records"], planned):
                entry = by_id[row["packet_id"]]
                _bound_row(row, entry, arm)
                if planned_row.get("request_sha256") != hashes[entry["packet_id"]]:
                    raise ValueError("Benchmark plan question definitions changed")
                if row["packet_id"] not in labels:
                    raise ValueError("Review every assigned benchmark packet, including failed model calls")
            fresh = [r for r in run["records"] if _fresh(r)]
            wall = run.get("fresh_execution_wall_ms")
            selectable = (len(fresh) == len(planned) and _number(wall) and wall > 0
                          and wall >= max((r["timing_ms"]["end_to_end"] for r in fresh), default=0))
            audit[worker] = {**_accuracy(run["records"], labels), "assigned_packets": len(planned),
                "fresh_validated_packets": len(fresh), "status_counts": dict(Counter(r["status"] for r in run["records"])),
                "batch_wall_ms": wall if _number(wall) else None, "eligible_for_selection": selectable,
                "ms_per_fresh_packet": wall / len(fresh) if selectable else None,
                "max_submitted_group_size": run.get("max_submitted_group_size")}
            if selectable: choices.append((wall / len(fresh), int(worker)))
        speed, workers = min(choices) if choices else (None, 1)
        profile["arms"][arm] = {"model": MODELS[arm], "workers": workers,
            "selection_status": "development_measured" if choices else "no_valid_fresh_comparison_default_one",
            "development_ms_per_packet": speed, "benchmark_sha256": digest(benchmark), "selection_audit": audit,
            "batching": "One complete panel per focal amount; batches are offline, disjoint and not identical workload replays"}
    if set(profile["arms"]) != set(ARMS[1:]):
        raise ValueError("All three model configurations must be frozen together")
    profile["sha256"] = digest(profile)
    write_new_json(root / "profile.json", profile)
    return profile


def validate_profile(profile, inputs, protocol):
    """Consumers can reject edited profiles before holdout execution or freezing."""
    payload = {k: v for k, v in profile.items() if k != "sha256"}
    if (profile.get("schema_version") != "procurement-profile-v1" or profile.get("sha256") != digest(payload)
            or profile.get("inputs_sha256") != digest(inputs) or profile.get("protocol_sha256") != digest(protocol)
            or set(profile.get("arms", {})) != set(MODELS)):
        raise ValueError("Frozen profile is changed or bound to different inputs/protocol")
    for arm, row in profile["arms"].items():
        if row.get("model") != MODELS[arm] or type(row.get("workers")) is not int or row["workers"] not in {1, 4, 8}:
            raise ValueError("Frozen model or concurrency is invalid")
    return profile


def compare(inputs, runs, review, protocol=None):
    """Source labels measure errors; failed or missing predictions stay in denominators.

    Pass the registered protocol to admit its development-only benchmark variant.
    Without it, only runs bearing the input's exact base protocol hash are accepted.
    """
    entries, labels = _entries(inputs), _labels(inputs, review)
    base_hash = inputs.get("protocol_sha256")
    if not isinstance(base_hash, str):
        raise ValueError("Frozen inputs require a protocol binding")
    allowed = {base_hash}
    if protocol is not None:
        if digest(protocol) != base_hash: raise ValueError("Comparison protocol differs from inputs")
        allowed.add(digest({**protocol, "development_only": True}))
    by_arm = {a: {} for a in ARMS[1:]}
    for run in runs:
        arm = run.get("arm")
        if arm not in by_arm or run.get("model") != MODELS[arm] or run.get("protocol_sha256") not in allowed:
            raise ValueError("Comparison requires pinned model arms and registered protocols")
        for row in run["records"]:
            pid = row["packet_id"]
            if pid not in entries or pid in by_arm[arm]:
                raise ValueError("Comparison requires unique known results per packet/arm")
            if run["protocol_sha256"] != base_hash and entries[pid]["split"] != "development":
                raise ValueError("Development-only run contains holdout evidence")
            _bound_row(row, entries[pid], arm)
            by_arm[arm][pid] = row
    metrics = {}
    for arm in ARMS:
        counts, per_question = Counter(), {}
        predicted_projects, true_projects, adjudicated_predictions = set(), set(), set()
        unresolved_labels = 0
        for pid, gold in labels.items():
            entry = entries[pid]
            row = by_arm.get(arm, {}).get(pid, {})
            predicted = rules_answers(entry) if arm == "rules" else _categories(row) if row.get("status") == "completed" else {}
            counts["reviewed_packets"] += 1
            scoring = entry.get("model_required", True)
            counts["missing_reviewed_predictions"] += scoring and not predicted
            targets = {k: v for k, v in gold["source_label"].items() if v not in (None, "unresolved")}
            if scoring:
                for question, answer in targets.items():
                    q = per_question.setdefault(question, Counter())
                    q["reviewed"] += 1
                    q["matched"] += predicted.get(question) == answer
                    q["missing"] += question not in predicted
                if targets:
                    counts["exact_match_scored_packets"] += 1
                    counts["all_reviewed_categories_match"] += bool(predicted) and all(predicted.get(k) == v for k, v in targets.items())
            source_signal = gold.get("source_signal", "unresolved")
            base = (entry["source_id"], gold.get("project_id") or entry["project_id"])
            if source_signal == "eligible": true_projects.add((*base, gold["source_label"]["recipient"]))
            if source_signal == "unresolved": unresolved_labels += 1
            decision = signal_for(entry, predicted, inputs, gold)
            if decision["status"] == "signal":
                # A wrong-recipient signal must never borrow the gold recipient.
                project = (*base, predicted.get("recipient"))
                predicted_projects.add(project)
                if source_signal != "unresolved": adjudicated_predictions.add(project)
        rows = list(by_arm.get(arm, {}).values())
        status = Counter(r["status"] for r in rows)
        missing_ids = set(entries) - set(by_arm.get(arm, {})) if arm != "rules" else set()
        status["missing_result"] += len(missing_ids)
        fresh = [r for r in rows if _fresh(r)]
        metrics[arm] = {**dict(counts), "categories": {k: dict(v) for k, v in per_question.items()},
            "expected_packets": len(entries), "reported_packets": len(entries) if arm == "rules" else len(rows),
            "missing_required_results": sum(entries[pid].get("model_required", True) for pid in missing_ids),
            "frozen_inference_exemption_status_counts": dict(Counter(
                e.get("deterministic_signal_status", "unknown") for e in entries.values()
                if e.get("model_required") is False)),
            "frozen_inference_exemption_reason_counts": dict(Counter(
                e.get("skip_inference_reason", "frozen_input_does_not_require_inference") for e in entries.values()
                if e.get("model_required") is False)),
            "status_counts": dict(status), "cached_rows": sum(bool(r.get("cached")) for r in rows),
            "fresh_latency_rows": len(fresh), "fresh_end_to_end_ms": [r["timing_ms"]["end_to_end"] for r in fresh],
            "reviewed_qualifying_projects": len(true_projects), "recalled_qualifying_projects": len(true_projects & predicted_projects),
            "qualifying_project_recall": len(true_projects & predicted_projects) / len(true_projects) if true_projects else None,
            "false_positive_reviewed_projects": len(adjudicated_predictions - true_projects),
            "unresolved_source_signal_labels": unresolved_labels}
    times = [r["review_seconds"] for r in labels.values() if r.get("review_seconds") is not None]
    return {"schema_version": "procurement-quality-v1", "inputs_sha256": digest(inputs), "review_sha256": digest(review),
        "run_hashes": [digest(r) for r in runs], "arms": metrics,
        "reviewed_documents": len({entries[pid]["document_id"] for pid in labels}),
        "reviewed_packets": len(labels), "pending_review_packets": len(review.get("records", [])) - len(labels),
        "review_seconds": sum(times) if times and len(times) == len(labels) else None,
        "measured_review_seconds": sum(times) if times else None, "timed_review_packets": len(times),
        "unknown_review_time_packets": len(labels) - len(times), "alpha_proven": False,
        "useful_event_recall": "Denominator is independently reviewed eligible source/project/recipient identities, including inference-exempt, failed or missing model results. Exempt unknowns are not negative source labels. It is not exhaustive corpus recall.",
        "sampling_note": "Purposive candidates/disagreements/retained-row audit, not an unbiased corpus accuracy estimate or full-corpus semantic model comparison. Category accuracy excludes model-exempt rows; independent source review and project recall retain them. Mapping/extraction gaps remain unknown. Offline latency excludes caches and does not establish live arrival performance."}
