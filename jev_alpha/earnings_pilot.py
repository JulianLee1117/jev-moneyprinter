"""Bounded development feature run, completed and frozen before return collection.

Inputs are current copies of original releases, not proven historical versions.
The deterministic passage budget is a disclosed retrieval limitation, not a
claim of exhaustive issuer knowledge. No prices, fitting, or orders here.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import re

from .compact_decisions import (build_compact_request, run_compact,
                                validate_compact_response)
from .earnings_returns import semantic_score
from .experiment import digest, run_one
from .jev import build_request, estimate_request, validate_response
from .readthrough import valid_jev_identity
from .store import Store, canonical_json, read_json, utc_now, write_new_json


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def select_passages(passages, byte_budget=8000):
    """Whole original passages, fixed keyword neighborhoods, independent of returns."""
    patterns = [(r"outlook|guidance|expect|forecast|anticipat|reaffirm|rais|lower|revis", 5),
                (r"demand|order|backlog|margin|pricing|cost|cash|inventory|risk|disrupt", 3)]
    scores = [sum(w for p, w in patterns if re.search(p, row["text"], re.I))
              + (4 if i < 4 else 0) for i, row in enumerate(passages)]
    selected, used = set(), 0
    for i in sorted(range(len(passages)), key=lambda i: (-scores[i], i)):
        neighborhood = sorted({max(0, i - 1), i, min(len(passages) - 1, i + 1)} - selected)
        size = sum(len(canonical_json(passages[j])) for j in neighborhood)
        if used + size <= byte_budget:
            selected.update(neighborhood)
            used += size
    return ([p for i, p in enumerate(passages) if i in selected],
            {"method": "earnings-keyword-neighborhood-v1", "byte_budget": byte_budget,
             "selected_bytes": used, "original_passages": len(passages),
             "omitted_passage_ids": [p["passage_id"] for i, p in enumerate(passages) if i not in selected],
             "retrieval_recall_validated": False})


def prepare(root: Path, panel_path: Path, protocol_path: Path, addendum_path: Path):
    panel = read_json(panel_path)
    source_path = root / "sources/source-pairs.v2.json"
    pairs = read_json(source_path)["pairs"]
    cohort_path, windows_path = root / "sources/cohort.v1.json", root / "windows.v1.json"
    expected = {r["event_id"] for r in read_json(cohort_path)["selected"]}
    if len(pairs) != 12 or {p["event_id"] for p in pairs} != expected:
        raise ValueError("Preserve all twelve frozen source outcomes")
    entries = []
    for pair in pairs:
        entry = {"event_id": pair["event_id"], "symbol": pair["symbol"],
                 "status": "source_unavailable", "request": None}
        if pair.get("usable_paired_text"):
            documents = []
            for role in ("prior", "current"):
                docs = [d for d in pair["documents"] if d["role"] == role]
                if len(docs) != 1 or len(docs[0].get("exhibits", [])) != 1:
                    raise ValueError("Expected one original release packet per role")
                doc = docs[0]
                artifact = doc["exhibits"][0]
                path = root / "sources" / artifact["packet_path"]
                if file_hash(path) != artifact["packet_sha256"]:
                    raise ValueError("Source packet hash changed")
                packet = read_json(path)
                passages, selection = select_passages(packet["current_source_passages_with_ids"])
                if not passages:
                    raise ValueError("No whole original passage fits fixed budget")
                documents.append({"role": role, "source_url": doc["source_url"],
                    "filing_date": doc["filingDate"], "historical_content_version_verified": False,
                    "passages": passages, "selection": selection})
            # Large omission-ID arrays are audit metadata, not model context.
            evidence = {"symbol": pair["symbol"],
                "coverage": "Deterministically selected original passages; omitted text may contain relevant evidence. Missing comparisons must be unknown.",
                "documents": [{k: v for k, v in d.items() if k != "selection"} for d in documents]}
            request = build_request(evidence, panel)
            estimate = estimate_request(request)
            if estimate["conservative_input_tokens"] > 28000:
                raise ValueError("Prepared request exceeds frozen context guard")
            entry.update(status="prepared", request=request, request_sha256=digest(request),
                         estimate=estimate, document_selection=documents)
        entries.append(entry)
    manifest = {"schema_version": "earnings-features-input-v1", "frozen_at": utc_now(),
        "cohort_sha256": file_hash(cohort_path), "windows_sha256": file_hash(windows_path),
        "sources_sha256": file_hash(source_path), "panel_sha256": file_hash(panel_path),
        "protocol_sha256": file_hash(protocol_path), "addendum_sha256": file_hash(addendum_path),
        "records": entries, "prices_opened": False, "phase_budget_usd": 2,
        "question_ids": list(panel["questions"])}
    write_new_json(root / "feature-inputs.v1.json", manifest)
    return manifest


def run_features(root: Path):
    inputs = read_json(root / "feature-inputs.v1.json")
    if (root / "features.v1.json").exists() or (root / "market/collection-plan.v1.json").exists():
        raise ValueError("Final features or market collection already frozen")
    store, records = Store(root / "models"), []
    for entry in inputs["records"]:
        row = {"event_id": entry["event_id"], "symbol": entry["symbol"],
               "status": entry["status"], "arms": {},
               "scores": {"jev_native": None, "jev_category": None, "compact": None}}
        for arm in ("jev", "compact"):
            path = root / "feature-attempts" / f"{entry['event_id']}-{arm}.json"
            if entry["status"] != "prepared":
                result = {"status": "source_unavailable"}
            elif path.exists():
                result = read_json(path)
                if result.get("status") == "valid":
                    run = result["run"]
                    body = entry["request"] if arm == "jev" else build_compact_request(entry["request"])
                    if run.get("request_hash") != digest(body):
                        raise ValueError("Archived feature attempt does not match frozen request")
                    validator = validate_response if arm == "jev" else validate_compact_response
                    response = validator(run["response"], entry["request"])
                    if arm == "jev" and not valid_jev_identity(response.get("model")):
                        raise ValueError("Unexpected archived Jev serving identity")
                    answers = response["answers"]
                    if result["categorical_score"] != semantic_score(answers, inputs["question_ids"], False):
                        raise ValueError("Archived categorical score differs from response")
                    if arm == "jev" and result["native_score"] != semantic_score(answers, inputs["question_ids"], True):
                        raise ValueError("Archived native score differs from response")
                elif result.get("status") != "failed":
                    raise ValueError("Nonterminal archived feature attempt")
            else:
                try:
                    run = (run_one(store, entry["request"], arm="jev", phase_budget_usd=2,
                                   transport="curl") if arm == "jev"
                           else run_compact(store, entry["request"], phase_budget_usd=2))
                    if arm == "jev" and not valid_jev_identity(run["response"].get("model")):
                        raise ValueError("Unexpected Jev serving identity")
                    answers = run["response"]["answers"]
                    categorical = semantic_score(answers, inputs["question_ids"], False)
                    native = semantic_score(answers, inputs["question_ids"], True) if arm == "jev" else None
                    result = {"status": "valid", "run": run, "categorical_score": categorical,
                              "native_score": native}
                except Exception as exc:
                    # Existing adapters archive redacted failure diagnostics and reserve cost.
                    result = {"status": "failed", "error_type": type(exc).__name__}
                write_new_json(path, result)
            row["arms"][arm] = result
            if result["status"] == "valid":
                if arm == "jev":
                    row["scores"].update(jev_native=result["native_score"], jev_category=result["categorical_score"])
                else:
                    row["scores"]["compact"] = result["categorical_score"]
            print(entry["event_id"], arm, result["status"], flush=True)
        records.append(row)
    frozen = {"schema_version": "earnings-features-v1", "frozen_at": utc_now(),
        "all_feature_attempts_frozen": True, "prices_opened": False,
        "cohort_sha256": inputs["cohort_sha256"], "windows_sha256": inputs["windows_sha256"],
        "inputs_sha256": file_hash(root / "feature-inputs.v1.json"), "records": records}
    write_new_json(root / "features.v1.json", frozen)
    return frozen
