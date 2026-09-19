"""Source-span coverage diagnostics, not estimates of alpha or exhaustive recall.

Freeze labels before examining retrieval outputs. Existing interpretation labels
can be reused as a development diagnostic, but do not exhaust all consequential
source text. A complete source copy is a coverage ceiling, not a model baseline.
Offsets are half-open Python Unicode code-point offsets, never byte offsets.

Typical use::

    frozen = freeze_support_labels(labels, full_source_states)
    arms = {
        "full_source": {"selections": [selection_from_state(s) for s in states]},
        "keyword": {"selections": [selection_from_state(compact_state(s)) for s in states]},
        "jev": {"selections": [selection_from_fanout(selection)],
                "inference_calls": [{"call_id": "request-sha256", "reported_cost_usd": .001}]},
    }
    report = evaluate_coverage(frozen, full_source_states, arms)

Absent method documents and unresolved labels remain in the label denominator.
Unknown costs remain unknown; cached responses should supply original inference
cost separately from incremental replay cost. No network or filesystem I/O.
"""

from __future__ import annotations

import hashlib
import json
import math


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _identity(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Nonempty {name} required")
    return value


def _sources(states: list[dict]) -> dict:
    sources = {}
    for state in states:
        manifest = state["episode_manifest"]
        doc = _identity(manifest.get("document_id"), "document_id")
        sha = _identity(manifest.get("source_sha256"), "source_sha256")
        if doc in sources:
            raise ValueError("Duplicate source document")
        if manifest.get("passage_selection") != "all":
            raise ValueError("Coverage denominators require full source packets")
        passages = {}
        for passage in state["current_source_passages_with_ids"]:
            pid = _identity(passage.get("passage_id"), "passage_id")
            if pid in passages or not isinstance(passage.get("text"), str):
                raise ValueError("Source passages require unique IDs and exact text")
            passages[pid] = passage["text"]
        if not passages:
            raise ValueError("Source packets cannot be empty")
        declared_count = manifest.get("original_passage_count")
        if declared_count is not None and (type(declared_count) is not int or declared_count != len(passages)):
            raise ValueError("Full source packet does not match its original passage count")
        sources[doc] = {"source_sha256": sha, "passages": passages}
    return sources


def freeze_support_labels(labels: dict, source_states: list[dict]) -> dict:
    """Convert pre-existing citations into frozen exact source-span requirements.

    Every listed passage is required in full: this deliberately conservative
    diagnostic does not invent smaller semantic anchors from model outputs.
    Missing documents, passage IDs or source versions become unresolved labels,
    rather than disappearing. Contradictory excerpts are rejected as corrupt
    labels. To scope a pilot, explicitly subset records *before* this call and
    retain the parent label-file hash in the surrounding experiment manifest.
    """
    sources = _sources(source_states)
    records = labels.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("At least one frozen label is required")
    frozen, identities = [], set()
    for row in records:
        doc = _identity(row.get("document_id"), "label document_id")
        question = _identity(row.get("question_id"), "question_id")
        identity = (doc, question)
        if identity in identities:
            raise ValueError("Duplicate document-question label")
        identities.add(identity)
        origin = _identity(row.get("label_origin"), "label_origin")
        split = _identity(row.get("split"), "split")
        evidence = row.get("evidence")
        if not isinstance(evidence, list):
            raise ValueError("Label evidence must be a list")
        spans, issues = {}, []
        source = sources.get(doc)
        if source is None:
            issues.append("source_document_missing")
        for citation in evidence:
            sha = citation.get("source_sha256")
            ids = citation.get("passage_ids")
            if not isinstance(ids, list) or not ids or any(not isinstance(p, str) or not p for p in ids):
                issues.append("citation_passage_ids_missing")
                continue
            if len(set(ids)) != len(ids):
                raise ValueError("Citation passage IDs must not repeat")
            if source is None:
                continue
            if sha != source["source_sha256"]:
                issues.append("citation_source_version_missing")
                continue
            if any(pid not in source["passages"] for pid in ids):
                issues.append("cited_passage_missing")
                continue
            excerpt = "\n".join(source["passages"][pid] for pid in ids)
            if citation.get("excerpt") != excerpt:
                raise ValueError("Label excerpt does not match exact cited source passages")
            for pid in ids:
                text = source["passages"][pid]
                if not text:
                    issues.append("empty_support_passage")
                else:
                    spans[pid] = {"passage_id": pid, "offset_start": 0,
                                  "offset_end": len(text), "text": text}
        if not spans:
            issues.append("no_resolved_support")
        frozen.append({"label_id": doc + ":" + question, "document_id": doc,
                       "question_id": question, "label_origin": origin, "split": split,
                       "category": "exception" if question == "exception_limits_headline" else "interpretation_support",
                       "source_sha256": source["source_sha256"] if source else None,
                       "spans": list(spans.values()), "issues": sorted(set(issues)),
                       "resolved": not issues})
    return {"schema_version": "source-support-labels-v1", "input_labels_sha256": _digest(labels),
            "source_snapshot_sha256": _digest(sources), "records": frozen,
            "interpretation": "Existing source-citation support only; not exhaustive materiality labels, independent gold, or alpha.",
            "offset_unit": "unicode_code_points"}


def selection_from_state(state: dict) -> dict:
    """Normalize a full or compact packet without silently treating IDs as text."""
    manifest = state["episode_manifest"]
    return {"document_id": manifest["document_id"], "source_sha256": manifest["source_sha256"],
            "status": "complete", "spans": [
                {"passage_id": p["passage_id"], "offset_start": 0,
                 "offset_end": len(p["text"]), "text": p["text"]}
                for p in state["current_source_passages_with_ids"]]}


def selection_from_fanout(selection: dict) -> dict:
    """Use exact retained fragments; retaining one fragment does not retain its parent."""
    if selection.get("schema_version") != "jev-passage-selection-v1":
        raise ValueError("Expected a Jev passage selection")
    return {"document_id": selection["document_id"], "source_sha256": selection["source_sha256"],
            "status": selection["status"], "spans": [
                {"passage_id": f["parent_passage_id"], "offset_start": f["offset_start"],
                 "offset_end": f["offset_end"], "text": f["text"]}
                for f in selection["selected_fragments"]]}


def _union(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    result = []
    for start, end in sorted(intervals):
        if result and start <= result[-1][1]:
            result[-1] = (result[-1][0], max(end, result[-1][1]))
        else:
            result.append((start, end))
    return result


def _retained(start: int, end: int, selected: list[tuple[int, int]]) -> int:
    return sum(max(0, min(end, b) - max(start, a)) for a, b in _union(selected))


def _costs(calls: list[dict]) -> dict:
    unique = {}
    for call in calls:
        call_id = _identity(call.get("call_id"), "inference call_id")
        cost = call.get("reported_cost_usd")
        if cost is not None and (type(cost) not in (int, float) or not math.isfinite(cost) or cost < 0):
            raise ValueError("Reported inference cost must be finite and nonnegative or unknown")
        if call_id in unique and unique[call_id] != cost:
            raise ValueError("Conflicting costs for the same inference call")
        unique[call_id] = cost
    known = math.fsum(cost for cost in unique.values() if cost is not None)
    missing = sum(cost is None for cost in unique.values())
    return {"unique_inference_calls": len(unique), "calls_with_unknown_cost": missing,
            "known_reported_cost_usd": known, "total_reported_cost_usd": None if missing else known}


def evaluate_coverage(frozen: dict, source_states: list[dict], arms: dict) -> dict:
    """Compare exact retained support, review volume and cost with all-label denominators.

    Every omitted required span is an unsafe-exclusion *candidate*, not proof the
    model made a wrong prediction. Passing all existing labels cannot establish
    exhaustive recall because unlabeled consequential text may still be omitted.
    Incomplete selector runs may retain fallback evidence; coverage counts that
    evidence but reports the run as incomplete, so fallback cannot masquerade as
    successful filtering. No threshold tuning or label generation occurs here.
    """
    sources = _sources(source_states)
    if frozen.get("schema_version") != "source-support-labels-v1" or frozen.get("source_snapshot_sha256") != _digest(sources):
        raise ValueError("Frozen labels must match the exact source snapshot")
    records = frozen["records"]
    if not records or not isinstance(arms, dict) or not arms:
        raise ValueError("Nonempty labels and retrieval arms required")
    output = {}
    source_chars = sum(len(t) for s in sources.values() for t in s["passages"].values())
    source_bytes = sum(len(t.encode()) for s in sources.values() for t in s["passages"].values())
    distinct_support = {}
    for label in records:
        for span in label["spans"]:
            key = (label["document_id"], span["passage_id"])
            distinct_support.setdefault(key, []).append((span["offset_start"], span["offset_end"]))
    support_chars = sum(b-a for spans in distinct_support.values() for a, b in _union(spans))
    for name, arm in arms.items():
        _identity(name, "arm name")
        selected, present, incomplete = {}, set(), []
        display_chars = 0
        for selection in arm.get("selections", []):
            doc = selection["document_id"]
            if doc in present:
                raise ValueError("Duplicate selected document in an arm")
            present.add(doc)
            source = sources.get(doc)
            if source is None or selection["source_sha256"] != source["source_sha256"]:
                raise ValueError("Selection source version is outside frozen sources")
            if selection.get("status") not in {"complete", "incomplete"}:
                raise ValueError("Selection must declare complete or incomplete execution")
            if selection["status"] == "incomplete":
                incomplete.append(doc)
            for span in selection["spans"]:
                pid, start, end = span["passage_id"], span["offset_start"], span["offset_end"]
                text = source["passages"].get(pid)
                if text is None or type(start) is not int or type(end) is not int or not 0 <= start <= end <= len(text):
                    raise ValueError("Selection contains an invalid source span")
                if span.get("text") != text[start:end]:
                    raise ValueError("Selection text does not match exact frozen source span")
                selected.setdefault((doc, pid), []).append((start, end))
                display_chars += end-start
        retained_chars = sum(b-a for spans in selected.values() for a, b in _union(spans))
        retained_bytes = sum(len(sources[doc]["passages"][pid][a:b].encode())
                             for (doc, pid), spans in selected.items() for a, b in _union(spans))
        rows = []
        for label in records:
            total, retained, omitted = 0, 0, []
            for span in label["spans"]:
                size = span["offset_end"]-span["offset_start"]
                kept = _retained(span["offset_start"], span["offset_end"],
                                 selected.get((label["document_id"], span["passage_id"]), []))
                total += size
                retained += kept
                if kept < size:
                    omitted.append({"passage_id": span["passage_id"], "required_characters": size,
                                    "retained_characters": kept})
            covered = label["resolved"] and total > 0 and retained == total
            rows.append({"label_id": label["label_id"], "document_id": label["document_id"],
                         "category": label["category"], "label_origin": label["label_origin"],
                         "split": label["split"], "label_resolved": label["resolved"],
                         "source_issues": label["issues"], "selection_document_missing": label["document_id"] not in present,
                         "required_support_characters": total, "retained_support_characters": retained,
                         "all_required_support_retained": covered, "omitted_support": omitted})
        covered_count = sum(r["all_required_support_retained"] for r in rows)
        distinct_retained = sum(_retained(a, b, selected.get(key, []))
                                for key, spans in distinct_support.items() for a, b in _union(spans))
        output[name] = {"label_count": len(records), "labels_fully_retained": covered_count,
                        "label_support_recall": covered_count / len(records),
                        "resolved_label_count": sum(r["resolved"] for r in records),
                        "unresolved_label_count": sum(not r["resolved"] for r in records),
                        "distinct_source_document_count": len(sources),
                        "labeled_document_count": len({r["document_id"] for r in records}),
                        "retrieved_document_count": len(present),
                        "distinct_support_passage_count": len(distinct_support),
                        "distinct_support_characters": support_chars,
                        "distinct_support_characters_retained": distinct_retained,
                        "distinct_support_character_recall": distinct_retained/support_chars if support_chars else None,
                        "review_text_characters": retained_chars, "display_text_characters_with_duplicates": display_chars,
                        "review_text_utf8_bytes": retained_bytes,
                        "source_text_characters": source_chars, "source_text_utf8_bytes": source_bytes,
                        "review_character_fraction": retained_chars/source_chars if source_chars else None,
                        "missing_selection_documents": sorted(set(sources)-present),
                        "incomplete_selector_documents": sorted(incomplete),
                        "unsafe_exclusion_candidates": [r["label_id"] for r in rows if not r["all_required_support_retained"]],
                        "exception_support_omitted_or_unresolved": [r["label_id"] for r in rows if r["category"] == "exception" and not r["all_required_support_retained"]],
                        "cost": _costs(arm.get("inference_calls", [])), "labels": rows}
    return {"schema_version": "source-support-coverage-v1", "frozen_labels_sha256": _digest(frozen),
            "source_snapshot_sha256": _digest(sources), "arms": output,
            "alpha_proven": False, "exhaustive_material_evidence_recall_measured": False,
            "limitations": ["Labels are existing citation support, not exhaustive consequential text.",
                            "Agent-reviewed development labels are not independent gold or a holdout.",
                            "Character recall covers resolvable cited spans; unresolved labels remain failures in label recall.",
                            "Full-source coverage is a ceiling, not a semantic model result.",
                            "Review volume is text volume, not measured human review time.",
                            "Different review volumes and incomplete runs prevent a matched-budget superiority claim.",
                            "Inference cost covers supplied unique calls; include failed-call costs or reservations separately.",
                            "No latency, price reaction, execution cost, portfolio return or alpha is measured."]}
