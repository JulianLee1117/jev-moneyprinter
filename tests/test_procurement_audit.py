import copy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from jev_alpha.procurement import digest
from jev_alpha.procurement_audit import compare, freeze_profile, validate_profile
from jev_alpha.procurement_models import MODELS, build_chat_request, build_panel


PROTOCOL = {"budget_usd": 50, "max_cost_usd": .25}


def fixture(count=9, holdout=False):
    records = []
    for i in range(count + int(holdout)):
        text = f"Project {i}. Orion Marine Construction approved contract amount $20 million; Granite Construction was an unsuccessful bidder."
        candidates = [{"candidate_id": cid, "symbol": symbol, "aliases": [alias],
                       "verified_aliases": [{"alias": alias, "ownership_share": 1}]}
                      for cid, symbol, alias in (("orn", "ORN", "Orion Marine Construction"),
                                                  ("gva", "GVA", "Granite Construction"))]
        amount = {"candidate_id": "a0", "value_usd": 20000000, "evidence": text}
        state = {"passages": [{"text": text}], "issuer_candidates": candidates,
                 "amount_candidates": [amount], "target_amount_id": "a0", "prior_records": []}
        records.append({"packet_id": f"p{i}", "document_id": f"d{i}", "source_id": "agency", "project_id": f"project-{i}",
                        "split": "holdout" if i == count else "development", "association_date": "2025-08-01",
                        "model_required": True, "target_amount": amount, "request": build_panel(state)})
    inputs = {"protocol_sha256": digest(PROTOCOL), "records": records, "annual_revenues": [
        {"symbol": symbol, "published_date": "2025-03-01", "period_end": "2024-12-31",
         "revenue_usd": 800000000, "source_url": "https://example.org/annual"} for symbol in ("ORN", "GVA")]}
    answer = {"recipient": "orn", "stage": "approved", "amount": "a0", "amount_kind": "contractor_value",
              "scope": "new_work", "conditions": "funded", "prior_known": "new_information"}
    review = {"price_blind": True, "inputs_sha256": digest(inputs), "label_origin": "independent_source_review",
              "records": [{"packet_id": e["packet_id"], "document_id": e["document_id"], "project_id": e["project_id"],
                           "project_identity_verified": True, "status": "reviewed", "source_label": dict(answer),
                           "source_signal": "eligible", "evidence": "Synthetic source: Orion approved, Granite unsuccessful."}
                          for e in records[:count]]}
    return inputs, review, answer


def row(entry, arm, answer):
    req = entry["request"]
    return {"packet_id": entry["packet_id"], "model": MODELS[arm], "status": "completed",
            "input_sha256": digest(req["state"]), "request_hash": digest(req if arm == "jev" else build_chat_request(req, MODELS[arm])),
            "answers": dict(answer), "cached": False, "latency_eligible": True,
            "timing_ms": {"queue": 0, "inference": 4, "validation": 1, "end_to_end": 5}, "cost_usd": .001}


def benchmarks(inputs, answer):
    ordered = sorted((e for e in inputs["records"] if e["split"] == "development"), key=lambda e: (digest(e["request"]), e["packet_id"]))
    groups = {str(w): ordered[i::3] for i, w in enumerate((1, 4, 8))}
    result = []
    for arm in MODELS:
        plan = {"arm": arm, "development_only": True, "protocol_sha256": digest({**PROTOCOL, "development_only": True}),
                "groups": {w: [{"packet_id": e["packet_id"], "request_sha256": digest(e["request"])} for e in group] for w, group in groups.items()}}
        runs = {w: {"arm": arm, "model": MODELS[arm], "workers": int(w), "protocol_sha256": plan["protocol_sha256"],
                    "fresh_execution_wall_ms": {"1": 30, "4": 20, "8": 10}[w], "max_submitted_group_size": min(int(w), len(group)),
                    "records": [row(e, arm, answer) for e in group]} for w, group in groups.items()}
        result.append({"plan": plan, "runs": runs})
    return result


class ProcurementAuditTests(unittest.TestCase):
    def test_profile_selection_excludes_caches_and_failures_and_detects_tampering(self):
        inputs, review, answer = fixture()
        bench = benchmarks(inputs, answer)
        for benchmark in bench:
            for record in benchmark["runs"]["8"]["records"]:
                record.update(cached=True, timing_ms=None, latency_eligible=False)
        # One arm's four-worker group has a failed call: it cannot win on speed.
        bench[0]["runs"]["4"]["records"][0].update(status="invalid_response", answers=None)
        with TemporaryDirectory() as tmp:
            profile = freeze_profile(Path(tmp), inputs, bench, review, PROTOCOL)
            self.assertEqual(profile["arms"][bench[0]["plan"]["arm"]]["workers"], 1)
            for arm in (b["plan"]["arm"] for b in bench[1:]):
                self.assertEqual(profile["arms"][arm]["workers"], 4)
                self.assertEqual(profile["arms"][arm]["selection_audit"]["4"]["category_accuracy"], 1)
                self.assertFalse(profile["arms"][arm]["selection_audit"]["8"]["eligible_for_selection"])
            validate_profile(profile, inputs, PROTOCOL)
            changed = copy.deepcopy(profile)
            changed["arms"]["jev"]["workers"] = 8
            with self.assertRaises(ValueError): validate_profile(changed, inputs, PROTOCOL)
            self.assertFalse(profile["alpha_proven"])

    def test_profile_rejects_holdout_labels_changed_questions_and_group_replays(self):
        inputs, review, answer = fixture(holdout=True)
        bench = benchmarks(inputs, answer)
        altered = copy.deepcopy(review)
        holdout = inputs["records"][-1]
        altered["records"].append({**altered["records"][0], "packet_id": holdout["packet_id"], "project_id": holdout["project_id"]})
        with TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError): freeze_profile(Path(tmp), inputs, bench, altered, PROTOCOL)
            changed = copy.deepcopy(bench)
            changed[0]["runs"]["1"]["records"][0]["request_hash"] = "different-questions"
            with self.assertRaises(ValueError): freeze_profile(Path(tmp), inputs, changed, review, PROTOCOL)
            changed = copy.deepcopy(bench)
            changed[0]["plan"]["groups"]["8"] = changed[0]["plan"]["groups"]["1"]
            with self.assertRaises(ValueError): freeze_profile(Path(tmp), inputs, changed, review, PROTOCOL)
            self.assertFalse((Path(tmp) / "profile.json").exists())

    def test_wrong_recipient_and_failed_or_missing_rows_remain_recall_failures(self):
        inputs, review, answer = fixture(3)
        review["records"][0]["review_seconds"] = 12
        wrong = row(inputs["records"][0], "nano", {**answer, "recipient": "gva"})
        failed = row(inputs["records"][1], "nano", answer)
        failed["status"] = "invalid_response"  # Retained answers/timing must not be scored as success.
        run = {"arm": "nano", "model": MODELS["nano"], "protocol_sha256": digest(PROTOCOL), "records": [wrong, failed]}
        result = compare(inputs, [run], review)
        metric = result["arms"]["nano"]
        self.assertEqual(metric["reviewed_qualifying_projects"], 3)
        self.assertEqual(metric["recalled_qualifying_projects"], 0)
        self.assertEqual(metric["qualifying_project_recall"], 0)
        self.assertEqual(metric["false_positive_reviewed_projects"], 1)
        self.assertEqual(metric["missing_reviewed_predictions"], 2)
        self.assertEqual(metric["status_counts"], {"completed": 1, "invalid_response": 1, "missing_result": 1})
        self.assertEqual(metric["fresh_latency_rows"], 1)
        self.assertEqual(result["arms"]["mini"]["missing_required_results"], 3)
        self.assertEqual(result["measured_review_seconds"], 12)
        self.assertIsNone(result["review_seconds"])
        self.assertEqual(result["unknown_review_time_packets"], 2)

    def test_comparison_authenticates_development_protocol_and_source_review_origin(self):
        inputs, review, answer = fixture(3, holdout=True)
        run = {"arm": "jev", "model": MODELS["jev"], "protocol_sha256": digest({**PROTOCOL, "development_only": True}),
               "records": [row(inputs["records"][0], "jev", answer)]}
        with self.assertRaises(ValueError): compare(inputs, [run], review)
        compare(inputs, [run], review, PROTOCOL)
        changed = copy.deepcopy(run)
        changed["records"] = [row(inputs["records"][-1], "jev", answer)]
        with self.assertRaises(ValueError): compare(inputs, [changed], review, PROTOCOL)
        changed = copy.deepcopy(review)
        changed["label_origin"] = "rule_predictions"
        with self.assertRaises(ValueError): compare(inputs, [run], changed, PROTOCOL)


if __name__ == "__main__":
    unittest.main()
