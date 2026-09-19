"""Summarize closed-document decisions; never infer a return from model scores."""
from __future__ import annotations

import math
from collections import Counter


def headline_direction(record: dict) -> str:
    """Reviewed-number midpoint heuristic, not proof of a management revision.

    The reviewer supplies period, basis and finite USD-billion bounds. Open or
    missing bounds remain unknown. No inferred upper bound for 'greater than'.
    """
    if record.get("comparability_supported") is not True:
        return "unknown"
    prior, current = record.get("prior", {}), record.get("current", {})
    if any(not prior.get(k) or prior[k] != current.get(k) for k in ("period", "basis")):
        return "unknown"
    if any(row.get("kind") != "forecast" for row in (prior, current)):
        return "unknown"
    values = [row.get(k) for row in (prior, current) for k in ("low", "high")]
    if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in values):
        return "unknown"
    pl, ph, cl, ch = values
    if pl > ph or cl > ch:
        raise ValueError("Reversed reviewed numeric bounds")
    delta = (cl + ch) / 2 - (pl + ph) / 2
    return "increase" if delta > 0 else "decrease" if delta < 0 else "unchanged"


def summarize_routes(paths: list[dict]) -> dict:
    """Keep epistemic unknowns separate from supported/rejected source claims."""
    indexed = {}
    allowed = {"supported", "rejected", "unresolved"}
    arms = ("jev", "jev_categorical", "baseline")
    for row in paths:
        key = (row["fixture_id"], row["route_id"], row["arm"])
        if (key in indexed or row["arm"] not in arms
                or row["status"] not in allowed or row["reference_status"] not in allowed):
            raise ValueError("Invalid or duplicate route result")
        indexed[key] = row
    units = sorted({(key[0], key[1]) for key in indexed})
    if not units:
        raise ValueError("Route results are empty")
    metrics = {}
    for arm in arms:
        rows = []
        for eid, rid in units:
            if (eid, rid, arm) not in indexed:
                raise ValueError("Every registered route must retain all three arms")
            rows.append(indexed[eid, rid, arm])
        metrics[arm] = {
            "route_count": len(rows),
            "exact_reference_matches": sum(r["status"] == r["reference_status"] for r in rows),
            "correctly_supported": sum(r["status"] == r["reference_status"] == "supported" for r in rows),
            "unsupported_positive_claims": sum(r["status"] == "supported" and r["reference_status"] != "supported" for r in rows),
            "missed_supported_references": sum(r["status"] != "supported" and r["reference_status"] == "supported" for r in rows),
            "statuses": dict(Counter(r["status"] for r in rows)),
        }
    native_changes = []
    for eid, rid in units:
        native, categorical, baseline = [indexed[eid, rid, arm] for arm in arms]
        reference = native["reference_status"]
        if any(r["reference_status"] != reference for r in (categorical, baseline)):
            raise ValueError("Inconsistent reference labels across model arms")
        if native["status"] != categorical["status"]:
            native_changes.append({
                "event_id": eid, "route_id": rid, "reference": reference,
                "categorical": categorical["status"], "native": native["status"],
                "improves_exact_match": native["status"] == reference and categorical["status"] != reference,
                "worsens_exact_match": categorical["status"] == reference and native["status"] != reference,
                "avoids_unsupported_positive": categorical["status"] == "supported" and reference != "supported" and native["status"] != "supported",
                "loses_supported_positive": categorical["status"] == reference == "supported" and native["status"] != "supported",
            })
    return {"metrics": metrics, "native_changes": native_changes,
            "event_count": len({eid for eid, _ in units}),
            "route_observations_are_independent": False,
            "alpha_established": False}
