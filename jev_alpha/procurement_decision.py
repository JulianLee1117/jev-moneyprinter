"""Bind a separately reviewed Jev contribution judgment to frozen results."""
from copy import deepcopy

from .procurement import digest
from .store import utc_now


def review_template(report, quality, profile, signals):
    if (report.get("signals_sha256") != signals.get("sha256")
            or signals.get("inputs_sha256") != quality.get("inputs_sha256")
            or signals.get("inputs_sha256") != profile.get("inputs_sha256")
            or signals.get("profile_sha256") != digest(profile)
            or signals.get("review_sha256") != quality.get("review_sha256")
            or signals.get("run_hashes") != quality.get("run_hashes")):
        raise ValueError("Return, quality and profile artifacts must describe the same frozen experiment")
    return {"schema_version":"procurement-contribution-review-v1",
        "return_report_sha256":digest(report), "quality_report_sha256":digest(quality),
        "profile_sha256":digest(profile), "status":"pending", "reviewer":None,
        "verdict":"inconclusive", "basis":None, "evidence":[], "rationale":None,
        "instructions":"Review useful correct projects and achievable entry opportunities. A cheaper/faster API response alone is insufficient. Historical replay cannot establish actual arrival-time advantage. Conventional equivalence drops the Jev-specific claim even if economics warrant further research."}


def attach_review(report, quality, profile, signals, review):
    expected = review_template(report, quality, profile, signals)
    if any(review.get(k) != expected[k] for k in ("schema_version", "return_report_sha256", "quality_report_sha256", "profile_sha256")):
        raise ValueError("Contribution review must match the exact result, quality and development profile")
    if review.get("status") != "reviewed" or not review.get("reviewer") or not review.get("rationale"):
        raise ValueError("An explicit contribution review is required")
    verdict = review.get("verdict")
    if verdict not in {"jev_warrants_prospective", "conventional_equivalent", "inconclusive"}:
        raise ValueError("Unknown contribution verdict")
    if verdict == "jev_warrants_prospective":
        if review.get("basis") not in {"incremental_correct_useful_projects", "achievable_entry_advantage"} or not review.get("evidence"):
            raise ValueError("Specify event-level discovery or achievable-entry evidence, not just API cost/latency")
        if review["basis"] == "achievable_entry_advantage" and not any(
                w.get("availability") == "prospective_first_seen" for w in report.get("windows", [])):
            raise ValueError("Historical replays do not establish actual operational timing advantage")
    result = deepcopy(report)
    result.update(contribution_review=review, contribution_review_sha256=digest(review),
        quality_report_sha256=digest(quality), profile_sha256=digest(profile), reviewed_at=utc_now(),
        jev_contribution_verified=verdict == "jev_warrants_prospective",
        contribution_status=verdict, prospective_authorized=False, proven_alpha=False)
    result["next_decision"] = ("retain_economic_hypothesis_drop_jev_claim" if verdict == "conventional_equivalent"
        and any(d.get("decision", "").startswith("promising_") for d in report.get("decisions",{}).values())
        else "opt_in_prospective_research_only" if verdict == "jev_warrants_prospective"
        and report.get("decisions",{}).get("jev",{}).get("decision", "").startswith("promising_")
        else "stop_or_inconclusive_no_expansion")
    return result
