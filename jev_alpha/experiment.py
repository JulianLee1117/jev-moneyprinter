"""Frozen inputs, paid-run accounting and comparisons without return claims."""

from __future__ import annotations

import copy
import hashlib
import math
import re
import time
from pathlib import Path

from .credentials import openrouter_key
from .jev import JevClient, JevRequestError, build_request, estimate_request, validate_response
from .research import evidence_packet
from .store import Store, canonical_json, read_json, utc_now, write_new_json


def digest(value: object) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def compact_state(state: dict, *, byte_budget: int = 10_000) -> dict:
    """Frozen keyword/context baseline for equal-evidence model comparisons.

    This is not the Jev selector. All omissions are explicit; labels/answers are
    never input to retrieval. Use fanout to evaluate every passage separately.
    """
    if type(byte_budget) is not int or byte_budget < 1000:
        raise ValueError("Evidence byte budget must be an integer >=1000")
    result = copy.deepcopy(state)
    passages = result["current_source_passages_with_ids"]
    scores = {}
    patterns = [(r"cash.deposit|assessment|rescission|rescind", 5),
                (r"except|unless|however|not.apply|remain.unchanged", 4),
                (r"preliminary|final|determin|align|deadline|initiat", 2),
                (r"scope|circumvent|subsid|margin", 1)]
    for index, passage in enumerate(passages):
        score = sum(weight for pattern, weight in patterns if re.search(pattern, passage["text"], re.I))
        scores[index] = score + (8 if index < 16 else 0)
    ranked = sorted(range(len(passages)), key=lambda i: (-scores[i], i))
    selected = set()
    used = 0
    for index in ranked:
        neighborhood = sorted({index, max(0, index - 1), min(len(passages) - 1, index + 1)} - selected)
        size = sum(len(canonical_json(passages[i])) for i in neighborhood)
        if used + size <= byte_budget:
            selected.update(neighborhood)
            used += size
    if not selected:
        raise ValueError("No whole passage neighborhood fits the evidence budget; use fanout fragmentation")
    result["current_source_passages_with_ids"] = [p for i, p in enumerate(passages) if i in selected]
    result["episode_manifest"].update({"selected_passage_count": len(selected),
        "passage_selection": "keyword-neighborhood-v1", "retrieval_byte_budget": byte_budget,
        "omitted_passage_ids": [p["passage_id"] for i, p in enumerate(passages) if i not in selected],
        "retrieval_recall_validated": False})
    return result


def prepare_comparison(store: Store, selection: dict, question_pack: dict, labels: dict, out: Path) -> dict:
    """Prepare only labeled development documents, without using their answers."""
    if out.exists():
        raise ValueError("Experiment output directory exists; preserve frozen inputs")
    _validate_development_labels(labels, question_pack["questions"])
    selected = {d["document_number"] for d in selection["selected"]}
    doc_ids = sorted({row["document_id"] for row in labels["records"]})
    if not set(doc_ids) <= selected:
        raise ValueError("Labeled documents must belong to the frozen pilot")
    names = sorted({row["question_id"] for row in labels["records"]})
    requests = []
    for doc_id in doc_ids:
        state = compact_state(evidence_packet(store, doc_id))
        request = build_request(state, question_pack, names)
        estimate = estimate_request(request)
        if estimate["conservative_input_tokens"] > 28_000:
            raise ValueError("Prepared comparison input exceeds conservative context guard")
        requests.append({"document_id": doc_id, "input_hash": digest(request["state"]),
                         "request_hash": digest(request), "request": request, "estimate": estimate})
    manifest = {"schema_version": "comparison-v1", "created_at": utc_now(),
        "selection_sha256": digest(selection), "labels_sha256": digest(labels),
        "question_pack_sha256": digest(question_pack), "question_ids": names,
        "scope": "Agent-reviewed development interpretation sanity test; no held-out alpha evaluation.",
        "models": ["keyword-rules-v1", "typesafe/jev-1.13", "openai/gpt-5.4-mini"],
        "requests": requests, "labels": labels, "prices_used": False,
        "retrieval": "Identical keyword-neighborhood-v1 evidence for all model arms; no label excerpts/answers used to select passages."}
    write_new_json(out / "manifest.json", manifest)
    return {"manifest": str(out / "manifest.json"), "documents": len(requests),
            "questions_per_document": len(names), "labels": len(labels["records"]),
            "jev_estimated_cost_usd": sum(r["estimate"]["estimated_cost_usd"] for r in requests), "submitted": False}


def _validate_development_labels(labels: dict, questions: dict) -> None:
    from .metrics import evaluate_labels
    evaluate_labels(labels, [])
    for label in labels["records"]:
        if label["split"] != "development":
            raise ValueError("This runner is development-only; holdout requires a separate frozen protocol")
        question = questions.get(label["question_id"])
        if not question:
            raise ValueError("Label question is absent from the comparison")
        expected = label["expected"]
        if (question["type"] == "choice" and (not isinstance(expected, str) or expected not in question["criteria"])) or (question["type"] == "noul" and type(expected) is not bool):
            raise ValueError("Label value does not match its frozen question")


def support_diagnostics(manifest: dict) -> list[dict]:
    entries = {r["document_id"]: r for r in manifest["requests"]}
    rows = []
    for label in manifest["labels"]["records"]:
        entry = entries[label["document_id"]]
        state = entry["request"]["state"]["evidence"]
        present = {p["passage_id"] for p in state["current_source_passages_with_ids"]}
        anchors = {pid for citation in label["evidence"] for pid in citation.get("passage_ids", [])}
        rows.append({"document_id": label["document_id"], "question_id": label["question_id"],
            "cited_passage_count": len(anchors), "cited_passages_retained": len(anchors & present),
            "all_cited_passages_retained": bool(anchors) and anchors <= present,
            "interpretation": "Post-retrieval diagnostic only; not used to select input, tune or exclude scores."})
    return rows


def rule_predictions(entry: dict) -> list[dict]:
    request = entry["request"]
    evidence = request["state"]["evidence"]
    title = evidence["episode_manifest"]["title"].lower()
    fulltext = "\n".join(p["text"] for p in evidence["current_source_passages_with_ids"]).lower()
    result = []
    for name, question in request["questions"].items():
        row = {"document_id": entry["document_id"], "question_id": name, "model": "keyword-rules-v1",
               "input_hash": entry["input_hash"], "kind": question["type"],
               "probability_interpretation": "Uncalibrated deterministic rule output"}
        guess = None
        if name == "decision_finality":
            if "final" in title:
                guess = "final"
            elif "preliminary" in title:
                guess = "preliminary"
            elif any(word in title for word in ("initiat", "schedule", "alignment", "opportunity")):
                guess = "not_a_decision"
        elif name == "document_authority":
            if any(word in title for word in ("final", "preliminary", "results", "determination")):
                guess = "agency_decision"
            elif any(word in title for word in ("schedule", "alignment", "opportunity")):
                guess = "procedural_notice"
        elif name == "treatment_time_basis":
            assessment = "assessment" in fulltext
            deposit = "cash deposit" in fulltext or "cash-deposit" in fulltext
            # Names come from the frozen question pack; unmapped rules abstain.
            for option in question.get("criteria", {}):
                if assessment and deposit and option == "both":
                    guess = option
        row["abstain"] = guess is None
        if question["type"] == "choice":
            options = list(question["criteria"])
            row["probabilities"] = {option: (1.0 if option == guess else 0.0) if guess else 1/len(options) for option in options}
        elif question["type"] == "noul":
            row["noul"] = 0.5
        result.append(row)
    return result


def run_one(store: Store, request: dict, *, arm: str = "jev", phase_budget_usd: float = 1,
            max_cost_usd: float | None = None, transport: str = "urllib", retry_reason: str | None = None) -> dict:
    """Record one bounded attempt. Never blindly retry an uncertain paid call."""
    if arm == "jev":
        estimate = estimate_request(request)
        body = request
        limit = 0.01 if max_cost_usd is None else max_cost_usd
        if estimate["conservative_input_tokens"] > 28_000:
            raise ValueError("Request exceeds conservative Jev context guard")
        client_class = JevClient
    elif arm == "baseline":
        from .baseline import BaselineClient, build_baseline_request, estimate_baseline
        body = build_baseline_request(request)
        estimate = estimate_baseline(request)
        limit = 0.1 if max_cost_usd is None else max_cost_usd
        client_class = BaselineClient
    else:
        raise ValueError("Unknown model arm")
    if not math.isfinite(phase_budget_usd) or phase_budget_usd <= 0:
        raise ValueError("Budget must be finite and positive")
    cached = store.cached_response(body)
    if cached:
        return {"response": validate_response(cached, request), "cached": True,
                "request_hash": digest(body), "latency_ms": None, "incremental_cost_usd": 0}
    key = openrouter_key()
    if not key:
        raise JevRequestError("Configure OPENROUTER_API_KEY in local environment or ignored .env")
    if estimate["estimated_cost_usd"] > limit:
        raise ValueError("Request exceeds per-call estimated cost guard")
    if transport not in {"urllib", "curl"}:
        raise ValueError("Transport must be urllib or curl")
    if transport == "curl":
        from .transport import CurlJSONTransport
        client = client_class(api_key=key, max_estimated_cost_usd=limit, transport=CurlJSONTransport())
    else:
        client = client_class(api_key=key, max_estimated_cost_usd=limit)
    attempt = store.reserve_attempt(body, estimate["estimated_cost_usd"], phase_budget_usd, retry_reason=retry_reason)
    started = time.perf_counter()
    response = None
    try:
        response = client.submit(request)
        store.cache_response(body, response)
    except BaseException as error:
        diagnostic = getattr(error, "response_data", None)
        if response is None and isinstance(diagnostic, dict):
            error.response_blob_sha256 = store.put_blob(canonical_json(diagnostic))
            usage = diagnostic.get("usage")
            cost = usage.get("cost") if isinstance(usage, dict) else None
            known_cost = type(cost) in (int, float) and math.isfinite(cost) and cost >= 0
            store.finish_attempt(attempt, {"usage": {"cost": cost}} if known_cost else None,
                                 status="invalid_response")
        else:
            store.finish_attempt(attempt, response)
        raise
    elapsed = (time.perf_counter() - started) * 1000
    store.finish_attempt(attempt, response)
    return {"response": response, "cached": False, "request_hash": digest(body),
            "transport": transport,
            "latency_ms": elapsed, "incremental_cost_usd": response.get("usage", {}).get("cost"),
            "estimated_cost_usd": estimate["estimated_cost_usd"]}


def answer_predictions(entry: dict, run: dict, arm: str) -> list[dict]:
    response = run["response"]
    rows = []
    for name, answer in response["answers"].items():
        row = {"document_id": entry["document_id"], "question_id": name, "model": arm,
               "serving_model": response["model"], "input_hash": entry["input_hash"],
               "run_id": run["request_hash"], "kind": answer["type"]}
        if answer["type"] == "choice":
            row["probabilities"] = answer["probabilities"]
        elif answer["type"] == "noul":
            row["noul"] = answer["noul"]
        else:
            continue
        if run["latency_ms"] is not None:
            row["latency_ms"] = run["latency_ms"]
        # Model cost comparison includes original inference cost for cached data.
        cost = response.get("usage", {}).get("cost")
        if cost is not None:
            row["cost_usd"] = cost
        rows.append(row)
    return rows


def run_comparison(store: Store, manifest_path: Path, out: Path, *, live: bool = False,
                   phase_budget_usd: float = 1, transport: str = "urllib") -> dict:
    from .metrics import evaluate_labels
    manifest = read_json(manifest_path)
    if manifest.get("schema_version") != "comparison-v1" or digest(manifest["labels"]) != manifest["labels_sha256"]:
        raise ValueError("Invalid comparison manifest or changed labels")
    if out.exists():
        raise ValueError("Comparison output directory exists; preserve run results")
    for entry in manifest["requests"]:
        if digest(entry["request"]) != entry["request_hash"] or digest(entry["request"]["state"]) != entry["input_hash"]:
            raise ValueError("Frozen request hash mismatch")
    if not manifest["requests"]:
        raise ValueError("Comparison has no prepared requests")
    _validate_development_labels(manifest["labels"], manifest["requests"][0]["request"]["questions"])
    evaluate_labels(manifest["labels"], [row for entry in manifest["requests"] for row in rule_predictions(entry)])
    if live and not openrouter_key():
        raise JevRequestError("Configure local OpenRouter key before live comparison")
    predictions, runs, failures = [], [], []
    halted = None
    for entry in manifest["requests"]:
        predictions.extend(rule_predictions(entry))
        if not live:
            continue
        for arm, model in (("jev", "typesafe/jev-1.13"), ("baseline", "openai/gpt-5.4-mini")):
            try:
                if halted:
                    raise ValueError("Remaining requests skipped after provider failure: " + halted)
                run = run_one(store, entry["request"], arm=arm, phase_budget_usd=phase_budget_usd, transport=transport)
                write_new_json(out / "responses" / f"{entry['document_id']}-{arm}.json", run)
                predictions.extend(answer_predictions(entry, run, model))
                runs.append({"document_id": entry["document_id"], "arm": arm, **{k:v for k,v in run.items() if k != "response"}})
            except (JevRequestError, ValueError, OSError) as exc:
                failures.append({"document_id": entry["document_id"], "arm": arm, "error": str(exc),
                                 "diagnostic_sha256": getattr(exc, "response_blob_sha256", None)})
                # Failure contributes an abstention to every question, not a
                # disappearing observation that would inflate measured quality.
                for row in rule_predictions(entry):
                    predictions.append({**row, "model": model, "abstain": True, "input_available": False})
                if isinstance(exc, JevRequestError):
                    halted = str(exc)
    metrics = evaluate_labels(manifest["labels"], predictions)
    report = {"schema_version": "comparison-report-v1", "created_at": utc_now(),
        "manifest_sha256": digest(manifest), "live_requested": live,
        "metrics": metrics, "runs": runs, "failures": failures,
        "label_support_diagnostics": support_diagnostics(manifest),
        "equal_evidence_comparison_complete": live and len(runs) == 2 * len(manifest["requests"]) and not failures,
        "alpha_proven": False, "holdout_evaluated": False,
        "limitation": "Small purposive agent-reviewed development set, measuring retrieval plus interpretation; evidence may be omitted. Partial model runs are inconclusive for equal-evidence quality. No return data, causal attribution, or verified calibration."}
    write_new_json(out / "predictions.json", predictions)
    write_new_json(out / "report.json", report)
    return {"report": str(out / "report.json"), "live_requested": live,
            "completed_model_calls_or_cache_hits": len(runs), "failures": len(failures), "metrics": metrics,
            "alpha_proven": False}


def run_fanout(store: Store, plan: dict, out: Path, *, live: bool = False,
               phase_budget_usd: float = 1, transport: str = "urllib") -> dict:
    from .fanout import select_evidence
    if out.exists():
        raise ValueError("Fanout output directory exists; preserve run results")
    # Validate plan coverage before any paid call. Missing responses remain
    # selected in this preflight check and cannot create a false complete run.
    select_evidence(plan, {})
    responses, runs, failures = {}, [], []
    if live:
        if not openrouter_key():
            raise JevRequestError("Configure local OpenRouter key before live fanout")
        for chunk in plan["chunks"]:
            try:
                run = run_one(store, chunk["request"], phase_budget_usd=phase_budget_usd, transport=transport)
                responses[chunk["chunk_id"]] = run["response"]
                write_new_json(out / "responses" / f"{chunk['chunk_id']}.json", run)
                runs.append({"chunk_id": chunk["chunk_id"], **{k:v for k,v in run.items() if k != "response"}})
            except (JevRequestError, ValueError, OSError) as exc:
                failures.append({"chunk_id": chunk["chunk_id"], "error": str(exc),
                                 "diagnostic_sha256": getattr(exc, "response_blob_sha256", None)})
                break
    selected = select_evidence(plan, responses)
    original_cost = sum(response.get("usage", {}).get("cost", 0) for response in responses.values())
    report = {"schema_version": "fanout-run-v1", "created_at": utc_now(),
        "plan_sha256": digest(plan), "live_requested": live,
        "reported_original_inference_cost_usd": original_cost,
        "runs": runs, "failures": failures, "selection": selected, "alpha_proven": False}
    write_new_json(out / "report.json", report)
    write_new_json(out / "selection.json", selected)
    return {"report": str(out / "report.json"), "status": selected["status"],
        "chunks_completed": len(responses), "chunks_total": len(plan["chunks"]),
        "passages_selected": len(selected["selected_passage_ids"]),
        "original_passages": len(plan["original_passage_ids"]),
        "reported_incremental_cost_usd": sum(r.get("incremental_cost_usd") or 0 for r in runs),
        "reported_original_inference_cost_usd": original_cost,
        "failures": failures, "alpha_proven": False}
