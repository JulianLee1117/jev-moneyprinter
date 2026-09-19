"""Join immutable arm outputs for independent source review, without model calls."""
from pathlib import Path

from .experiment import digest
from .operating import keyword_operating_baseline
from .store import read_json, utc_now


def comparison_manifest(cohort: dict, run_directories: dict[str, list[Path]]) -> dict:
    """Require every split's frozen provenance; preserve unavailable comparisons."""
    cohort_hash = digest(cohort)
    population = {(r["record_id"], v) for r in cohort["records"] for v in r["vendor_ids"]}
    keywords = keyword_operating_baseline(cohort["records"])
    selected = {"keyword": {(r["record_id"], v) for r in keywords["records"]
                            if r["candidate"] for v in r["vendor_ids"]}}
    states = {"keyword": {"cohort_sha256": cohort_hash, "status": "complete",
                         "response_coverage_complete": True, "evaluated_pairs": population,
                         "unknown_pairs": set()}}
    provenance = []
    configurations = {}
    if set(run_directories) - {"jev", "baseline", "semantic_retrieval"}:
        raise ValueError("Unknown operating arm")
    for arm in ("jev", "baseline", "semantic_retrieval"):
        chosen, evaluated, unknown, splits = set(), set(), set(), set()
        complete = True
        for directory in run_directories.get(arm, []):
            directory = Path(directory)
            report, selection, plan = (read_json(directory / name) for name in ("report.json", "selection.json", "plan.json"))
            if arm == "semantic_retrieval":
                expected_schemas = ("operating-semantic-run-v1", "operating-semantic-selection-v1", "operating-semantic-plan-v1")
                configuration = (plan.get("profile_sha256"), plan.get("model_pin"))
                applied = report.get("summary", {}).get("models_applied", [])
                if any(model != plan.get("model_pin") for model in applied):
                    raise ValueError("Semantic response model differs from its frozen pin")
                config_key = "semantic"
            else:
                expected_schemas = ("operating-screen-run-v1", "operating-selection-v1", "operating-screen-plan-v1")
                if report.get("summary", {}).get("arm") != arm or selection.get("arm") != arm:
                    raise ValueError("Mislabeled operating comparison arm")
                configuration = (plan.get("profile_sha256"), digest(plan.get("models")))
                config_key = "screen"
            if tuple(v.get("schema_version") for v in (report, selection, plan)) != expected_schemas:
                raise ValueError("Run schemas do not match the requested arm")
            if not all(configuration) or configurations.setdefault(config_key, configuration) != configuration:
                raise ValueError("Comparison model or profile differs across arms/splits")
            if (report.get("cohort_sha256") != cohort_hash or plan.get("cohort_sha256") != cohort_hash
                    or report.get("plan_sha256") != digest(plan) or selection.get("plan_sha256") != digest(plan)
                    or report.get("selection_sha256") != digest(selection)
                    or report.get("protocol_sha256") != cohort.get("protocol_sha256")
                    or plan.get("protocol_sha256") != cohort.get("protocol_sha256")):
                raise ValueError("Run does not match the frozen cohort and plan")
            split = report["split"]
            if split in splits or split not in {"development", "evaluation"} or plan.get("split") != split:
                raise ValueError("Duplicate or invalid comparison split")
            splits.add(split)
            expected_records = sorted((r for r in cohort["records"]
                                       if cohort["splits"][r["record_id"]] == split), key=lambda r: r["record_id"])
            if digest(plan.get("records")) != digest(expected_records):
                raise ValueError("Plan source evidence differs from the frozen cohort")
            expected = {(r["record_id"], v) for r in cohort["records"]
                        if cohort["splits"][r["record_id"]] == split for v in r["vendor_ids"]}
            outcomes = selection["outcomes"]
            pairs = [(r["record_id"], r["vendor_id"]) for r in outcomes]
            if len(set(pairs)) != len(pairs) or set(pairs) != expected:
                raise ValueError("Run outcome coverage differs from the registered split")
            for row in outcomes:
                pair = row["record_id"], row["vendor_id"]
                if row["status"] in {"candidate", "selected"}:
                    chosen.add(pair)
                elif row["status"] == "unknown":
                    unknown.add(pair)
                elif row["status"] not in {"screened_out", "not_selected"}:
                    raise ValueError("Unknown operating outcome status")
            evaluated.update(pairs)
            complete = (complete and report["summary"]["status"] == "complete"
                        and selection.get("status") == "complete" and selection.get("response_coverage_complete") is True)
            provenance.append({"arm": arm, "split": split, "directory": str(directory),
                               "report_sha256": digest(report), "selection_sha256": digest(selection),
                               "summary": report["summary"]})
        complete = complete and evaluated == population
        selected[arm] = chosen
        states[arm] = {"cohort_sha256": cohort_hash, "status": "complete" if complete else "incomplete",
                       "response_coverage_complete": complete, "evaluated_pairs": evaluated, "unknown_pairs": unknown}
    def rows(pairs):
        return [{"record_id": rid, "vendor_id": vendor} for rid, vendor in sorted(pairs)]
    return {"schema_version": "operating-comparison-manifest-v1", "created_at": utc_now(),
            "cohort_sha256": cohort_hash, "selected_pairs": {k: rows(v) for k, v in selected.items()},
            "arm_status": {k: {**v, "evaluated_pairs": rows(v["evaluated_pairs"]),
                                "unknown_pairs": rows(v["unknown_pairs"])} for k, v in states.items()},
            "runs": provenance, "alpha_proven": False}
