"""Evaluate sourced document judgments, never investment returns.

Labels and predictions are separate inputs. Agent-reviewed and synthetic labels
are deliberately isolated from human labels; neither is human gold evidence.
Accuracy and coverage use *all* labels in a group, including missing predictions.
Ordinary Brier scores describe answered rows only. Separately named worst-case
scores penalize every unanswered row by 2.0004 (choice, allowing bounded mass
rounding) or 1 (noul).
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any
from urllib.parse import urlsplit

from .jev import JevValidationError, choice_probability_mass_delta


class MetricsValidationError(ValueError):
    """An input cannot support an unambiguous benchmark comparison."""


def _text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _number(value: Any, *, upper: float = math.inf) -> bool:
    try:
        return type(value) in (int, float) and 0 <= value <= upper and math.isfinite(value)
    except OverflowError:
        return False


def _labels(value: Any) -> list[dict]:
    if not isinstance(value, dict) or value.get("schema_version") != "benchmark_labels_v1":
        raise MetricsValidationError("Labels require schema_version benchmark_labels_v1.")
    records = value.get("records")
    if not isinstance(records, list):
        raise MetricsValidationError("Label records must be a list.")
    seen = set()
    question_kinds = {}
    document_splits = {}
    for record in records:
        if not isinstance(record, dict) or any(
            not _text(record.get(field)) for field in ("document_id", "question_id")
        ):
            raise MetricsValidationError("Each label requires document_id and question_id.")
        key = (record["document_id"], record["question_id"])
        if key in seen:
            raise MetricsValidationError("Duplicate document/question labels are not allowed.")
        seen.add(key)
        expected = record.get("expected")
        if type(expected) is not bool and not _text(expected):
            raise MetricsValidationError("Expected labels must be choice text or a noul boolean.")
        kind = "noul" if type(expected) is bool else "choice"
        previous_kind = question_kinds.setdefault(record["question_id"], kind)
        if previous_kind != kind:
            raise MetricsValidationError("A question cannot mix choice and noul labels.")
        if record.get("label_origin") not in ("human", "agent_reviewed", "synthetic"):
            raise MetricsValidationError("Label origin must be human, agent_reviewed or synthetic.")
        if record.get("split") not in ("development", "holdout"):
            raise MetricsValidationError("Label split must be development or holdout.")
        old_split = document_splits.setdefault(record["document_id"], record["split"])
        if old_split != record["split"]:
            raise MetricsValidationError("A document cannot occur in both development and holdout.")
        evidence = record.get("evidence", [])
        if not isinstance(evidence, list):
            raise MetricsValidationError("Label evidence must be a list.")
        if record["label_origin"] != "synthetic" and not evidence:
            raise MetricsValidationError("Nonsynthetic labels require source evidence.")
        for item in evidence:
            if not isinstance(item, dict) or not _text(item.get("excerpt")):
                raise MetricsValidationError("Evidence requires a nonempty source excerpt.")
            url = item.get("source_url")
            try:
                parsed = urlsplit(url) if isinstance(url, str) else None
                valid_url = parsed and parsed.scheme in {"http", "https"} and parsed.hostname
            except ValueError:
                valid_url = False
            if not valid_url:
                raise MetricsValidationError("Evidence requires an HTTP(S) source URL.")
    return records


def _prediction(record: Any) -> dict:
    if not isinstance(record, dict) or any(
        not _text(record.get(field))
        for field in ("model", "document_id", "question_id", "input_hash")
    ):
        raise MetricsValidationError("Predictions require model, document_id, question_id and input_hash.")
    kind = record.get("kind")
    if kind not in ("choice", "noul"):
        raise MetricsValidationError("Only choice and noul predictions are supported; score is not inferred.")
    for field in ("abstain", "input_available"):
        if field in record and type(record[field]) is not bool:
            raise MetricsValidationError(f"{field} must be a boolean.")
    if "run_id" in record and not _text(record["run_id"]):
        raise MetricsValidationError("run_id must be nonempty text when provided.")
    for field in ("cost_usd", "latency_ms"):
        if field in record and not _number(record[field]):
            raise MetricsValidationError(f"{field} must be finite and nonnegative.")
    uncovered = record.get("abstain", False) or not record.get("input_available", True)
    result = dict(record)
    if kind == "choice":
        probabilities = record.get("probabilities")
        if "probabilities" not in record and uncovered:
            return result
        if (not isinstance(probabilities, dict) or len(probabilities) < 2
                or not all(_text(key) for key in probabilities)
                or not all(_number(value, upper=1) for value in probabilities.values())):
            raise MetricsValidationError("Choice probabilities require at least two options and sum to one.")
        try:
            result["probability_mass_delta"] = choice_probability_mass_delta(probabilities)
        except JevValidationError:
            raise MetricsValidationError("Choice probabilities require unit mass or bounded two-decimal rounding.") from None
        # Lexical tie breaking is fixed independently of the expected label.
        result["predicted"] = min(probabilities, key=lambda key: (-probabilities[key], key))
    else:
        if "noul" in record and "probability" in record and record["noul"] != record["probability"]:
            raise MetricsValidationError("Conflicting noul and probability fields.")
        probability = record.get("noul", record.get("probability"))
        if probability is None and uncovered and "noul" not in record and "probability" not in record:
            return result
        if not _number(probability, upper=1):
            raise MetricsValidationError("Noul probability must be finite and in [0,1].")
        result["noul"] = probability
        result["predicted"] = probability >= 0.5
    return result


def _divide(numerator: float, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _group(model: str, split: str, origin: str, labels: list[dict], predictions: dict) -> dict:
    answered = correct = missing = abstained = unavailable = 0
    totals = Counter()
    scores = {"choice": [], "noul": []}
    rounded_choice_rows = []
    class_support, class_predictions, class_correct = Counter(), Counter(), Counter()
    for label in labels:
        kind = "noul" if type(label["expected"]) is bool else "choice"
        totals[kind] += 1
        expected_key = (label["question_id"], label["expected"])
        class_support[expected_key] += 1
        prediction = predictions.get((model, label["document_id"], label["question_id"]))
        if prediction is None:
            missing += 1
            continue
        if not prediction.get("input_available", True):
            unavailable += 1
            continue
        if prediction.get("abstain", False):
            abstained += 1
            continue
        answered += 1
        correct += prediction["predicted"] == label["expected"]
        class_predictions[(label["question_id"], prediction["predicted"])] += 1
        if prediction["predicted"] == label["expected"]:
            class_correct[expected_key] += 1
        if kind == "noul":
            scores[kind].append((prediction["noul"] - int(label["expected"])) ** 2)
        else:
            if abs(prediction["probability_mass_delta"]) > 1e-5:
                rounded_choice_rows.append({
                    "document_id": label["document_id"], "question_id": label["question_id"],
                    "probability_mass_delta": prediction["probability_mass_delta"],
                })
            scores[kind].append(math.fsum(
                (probability - int(option == label["expected"])) ** 2
                for option, probability in prediction["probabilities"].items()
            ))
    count = len(labels)
    result = {
        "model": model, "split": split, "label_origin": origin,
        "human_gold_labels": origin == "human",
        "interpretation": {
            "human": "Extraction benchmark against supplied human labels; independence is not verified by this evaluator.",
            "agent_reviewed": "Provisional agreement with agent-reviewed labels; not human gold quality evidence.",
            "synthetic": "Software self-test only; not empirical model-quality evidence.",
        }[origin],
        "label_count": count, "answered": answered, "correct": correct,
        "missing_predictions": missing, "abstentions": abstained,
        "input_unavailable": unavailable, "coverage": _divide(answered, count),
        "accuracy": _divide(correct, count),
        "answered_accuracy": _divide(correct, answered),
        "choice_rounded_probability_count": len(rounded_choice_rows),
        "choice_rounded_probability_rows": rounded_choice_rows,
        "choice_brier_basis": "Raw returned probabilities, without renormalization; approximate for rounded rows and not a calibration claim.",
        "per_question_class_metrics": [
            {"question_id": key[0], "class_label": key[1],
             "support": class_support[key], "true_positives": class_correct[key],
             "false_negatives_including_unanswered": class_support[key] - class_correct[key],
             "false_positives": class_predictions[key] - class_correct[key],
             "precision": _divide(class_correct[key], class_predictions[key]),
             "recall": _divide(class_correct[key], class_support[key])}
            for key in sorted(set(class_support) | set(class_predictions), key=lambda key: (key[0], str(key[1])))
        ],
    }
    for kind, maximum in (("choice", 2.0004), ("noul", 1)):
        result[f"{kind}_label_count"] = totals[kind]
        result[f"{kind}_brier_answered_count"] = len(scores[kind])
        result[f"{kind}_brier_mean"] = _divide(math.fsum(scores[kind]), len(scores[kind]))
        result[f"{kind}_brier_worst_case_mean"] = _divide(
            math.fsum(scores[kind]) + maximum * (totals[kind] - len(scores[kind])), totals[kind]
        )
    return result


def evaluate_labels(labels: dict, predictions: list[dict]) -> dict:
    """Validate and score a flat prediction panel without I/O or input mutation.

    Label schema is ``benchmark_labels_v1`` with ``records``. Every label needs
    document_id, question_id, expected (str/bool), label_origin and split; real
    labels also need evidence [{source_url, excerpt}]. Predictions need model,
    document_id, question_id, input_hash, kind, and probabilities or noul.
    Explicit abstain/input_available flags may omit answer probabilities.

    Costs and latency, if supplied, describe whole calls and are deduplicated by
    (model, run_id), falling back to input_hash. Missing metadata stays unknown.
    A model with zero prediction records cannot be inferred and has no group.
    More than one label from a document does not create independent observations.
    """
    records = _labels(labels)
    if not isinstance(predictions, list):
        raise MetricsValidationError("Predictions must be a list.")
    label_index = {(row["document_id"], row["question_id"]): row for row in records}
    index = {}
    runs = {}
    models = set()
    supports = {}
    unmatched = Counter()
    matched_input_hashes = defaultdict(dict)
    for raw in predictions:
        record = _prediction(raw)
        key = (record["model"], record["document_id"], record["question_id"])
        if key in index:
            raise MetricsValidationError("Duplicate model/document/question predictions are not allowed.")
        index[key] = record
        models.add(record["model"])
        label = label_index.get(key[1:])
        if label is None:
            unmatched[record["model"]] += 1
        else:
            expected_kind = "noul" if type(label["expected"]) is bool else "choice"
            if record["kind"] != expected_kind:
                raise MetricsValidationError("Prediction kind does not match the expected label.")
            if record["kind"] == "choice" and "probabilities" in record:
                distribution = record["probabilities"]
                if label["expected"] not in distribution:
                    raise MetricsValidationError("Choice distribution omits the expected label.")
                support = frozenset(distribution)
                old_support = supports.setdefault(record["question_id"], support)
                if support != old_support:
                    raise MetricsValidationError("Choice options must be consistent for a question across predictions.")
            matched_input_hashes[key[1:]][record["model"]] = record["input_hash"]
        run_key = (record["model"], record.get("run_id", record["input_hash"]))
        run = runs.setdefault(run_key, {"input_hash": record["input_hash"]})
        if run["input_hash"] != record["input_hash"]:
            raise MetricsValidationError("A run_id cannot identify different input hashes.")
        for field in ("cost_usd", "latency_ms"):
            if field in record:
                if field in run and run[field] != record[field]:
                    raise MetricsValidationError("Repeated call metadata must agree; do not allocate call cost per question.")
                run[field] = record[field]
    grouped_labels = defaultdict(list)
    for row in records:
        grouped_labels[(row["split"], row["label_origin"])].append(row)
    groups = [_group(model, split, origin, group, index)
              for model in sorted(models)
              for (split, origin), group in sorted(grouped_labels.items())]
    model_totals = []
    for model in sorted(models):
        calls = [run for (call_model, _), run in runs.items() if call_model == model]
        entry = {"model": model, "unique_calls": len(calls),
                 "unmatched_predictions": unmatched[model]}
        for field in ("cost_usd", "latency_ms"):
            available = [call[field] for call in calls if field in call]
            entry[f"{field}_known_calls"] = len(available)
            entry[f"{field}_total"] = math.fsum(available) if available else None
            entry[f"{field}_complete"] = len(available) == len(calls)
        model_totals.append(entry)
    origin_counts = Counter(row["label_origin"] for row in records)
    return {
        "schema_version": "benchmark_metrics_v1",
        "status": "no_labels" if not records else "no_predictions" if not predictions else "evaluated",
        "label_count": len(records),
        "label_origins": {origin: origin_counts[origin] for origin in ("human", "agent_reviewed", "synthetic")},
        "human_gold_available": bool(origin_counts["human"]),
        "groups": groups, "model_totals": model_totals,
        "cross_model_input_hash_mismatches": [
            {"document_id": document, "question_id": question, "model_input_hashes": hashes}
            for (document, question), hashes in sorted(matched_input_hashes.items())
            if len(hashes) > 1 and len(set(hashes.values())) > 1
        ],
        "notes": [
            "Accuracy and coverage include every label, including missing and abstained predictions.",
            "Choice Brier is the raw multiclass squared-error sum (0–2 for unit mass, up to approximately 2.0004 with allowed rounding); noul Brier is binary (range 0–1).",
            "Two-decimal choice probabilities may have bounded mass error; their raw-value Brier is approximate. No probabilities are renormalized or claimed calibrated.",
            "Worst-case Brier assigns loss 2.0004 (choice) or 1 (noul) to each unanswered label; it is a coverage penalty, not measured calibration.",
            "Choice ties use lexical option order; noul uses a fixed 0.5 threshold.",
            "Input hashes are reported, not proof of equal evidence; model-specific full-request hashes may differ.",
            "Multiple questions on a document are correlated. These metrics make no claim about alpha or trading returns.",
            "Call latency totals are summed service durations, not elapsed wall time for parallel execution.",
        ],
    }
