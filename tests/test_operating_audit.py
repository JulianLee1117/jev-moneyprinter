import copy
import hashlib
import unittest

from jev_alpha.experiment import digest
from jev_alpha.operating_audit import freeze_operating_cohort, review_sample, evaluate_operating_reviews


def protocol():
    return {"schema_version": "operating-changes-protocol-v1", "experiment_id": "test",
            "window": {"start": "2026-06-20T00:00:00Z", "end_exclusive": "2026-09-19T00:00:00Z"},
            "max_records": 1000, "vendors": [{"vendor_id": "ddog"}, {"vendor_id": "net"}],
            "development_fraction": .2, "selection_seed": "frozen-test",
            "feasibility_gate": {"minimum_distinct_qualifying_episodes": 20, "minimum_vendors": 4}}


def record(rid, text="We moved our paid production systems in July.", thread=None):
    return {"record_id": rid, "source": "hackernews", "native_id": rid,
            "text": text, "content_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "url": "https://news.ycombinator.com/item?id=" + rid,
            "published_at": "2026-08-01T10:00:00Z", "state_observed_at": None,
            "captured_at": "2026-09-19T02:00:00Z", "thread_id": thread,
            "vendor_ids": ["ddog", "net"]}


def cohort(rows, p=None):
    p = p or protocol()
    return p, freeze_operating_cohort(p, {"protocol_sha256": digest(p), "records": rows, "status": "collected",
        "coverage": [{"status": "complete_query", "completeness": "complete", "deadline_partial": False,
                      "truncated": False, "quota": 1000}]})


ARMS = ("jev", "baseline", "keyword", "semantic_retrieval")


def complete_states(c, split=None):
    return {arm: {"cohort_sha256": digest(c), "status": "complete", "response_coverage_complete": True,
            "evaluated_pairs": [{"record_id": r["record_id"], "vendor_id": v} for r in c["records"]
                if split is None or c["splits"][r["record_id"]] == split for v in r["vendor_ids"]],
            "unknown_pairs": []} for arm in ARMS}


def all_negative(c, split=None):
    return [{"record_id": r["record_id"], "vendor_id": v, "qualifies": False} for r in c["records"]
            if split is None or c["splits"][r["record_id"]] == split for v in r["vendor_ids"]]


def small_gate():
    p = protocol()
    p["feasibility_gate"] = {"minimum_distinct_qualifying_episodes": 1, "minimum_vendors": 1}
    return p


class OperatingAuditTests(unittest.TestCase):
    def test_source_mutation_and_naive_time_rejected(self):
        for edit in ({"text": "altered"}, {"published_at": "2026-08-01"}):
            r = record("a")
            r.update(edit)
            with self.assertRaises(ValueError):
                cohort([r])

    def test_transitive_duplicate_threads_share_split(self):
        _, c = cohort([record("a", "same", "1"), record("b", "other", "1"),
                       record("c", "same", "2"), record("d", "fourth", "2")])
        self.assertEqual(c["cluster_count"], 1)
        self.assertEqual(len(set(c["splits"].values())), 1)

    def test_input_order_does_not_change_splits(self):
        rows = [record(str(i), str(i), str(i)) for i in range(40)]
        _, a = cohort(rows)
        _, b = cohort(list(reversed(rows)))
        self.assertEqual(a["splits"], b["splits"])

    def test_merged_reference_threads_and_known_parent_links_remain_connected(self):
        merged = record("a", "same", "first-root")
        merged["source_references"] = [
            {"record_id": "hackernews:a", "thread_id": "hackernews:first-root", "thread_resolution": "root"},
            {"record_id": "hackernews:copied", "thread_id": "hackernews:second-root", "thread_resolution": "root"},
            {"record_id": "reddit:copy", "thread_id": "reddit:third-root", "thread_resolution": "root"}]
        reddit = record("reddit:b", "third", "reddit:third-root")
        reddit["source"] = "reddit"
        child = record("child", "reply")
        child["parent_id"] = "hackernews:copied"
        _, c = cohort([merged, record("b", "different", "second-root"), reddit, child])
        self.assertEqual(c["cluster_count"], 1)
        self.assertEqual(len(set(c["splits"].values())), 1)
        self.assertIn("child", c["unresolved_thread_record_ids"])

    def test_normalized_unicode_copies_connect_and_freeze_does_not_alias_nested_inputs(self):
        row = record("a", "Ｃｌｏｕｄ", "one")
        row["source_references"] = [{"record_id": "hackernews:a", "thread_id": "one", "thread_resolution": "root"}]
        _, c = cohort([row, record("b", "Cloud", "two")])
        frozen_hash = digest(c)
        row["vendor_ids"].append("made-up")
        row["source_references"][0]["thread_id"] = "mutated"
        self.assertEqual(digest(c), frozen_hash)
        self.assertEqual(c["cluster_count"], 1)

    def test_unresolved_verified_partial_hn_chains_share_intermediate_ancestor(self):
        def api(native, parent):
            return {"source": "official_hn_api", "native_id": native, "parent_id": "hackernews:" + parent,
                    "status": 200, "item_type": "comment", "observation_id": 1,
                    "response_sha256": "a" * 64,
                    "url": f"https://hacker-news.firebaseio.com/v0/item/{native}.json"}
        rows = []
        for rid, parent in (("1", "11"), ("2", "22")):
            row = record("hackernews:" + rid, "distinct text " + rid)
            row["parent_id"] = "hackernews:" + parent
            row["root_provenance"] = {"status": "needs_item", "chain": [
                {"source": "original_capture", "record_id": row["record_id"], "parent_id": row["parent_id"]},
                api(parent, "100")]}
            rows.append(row)
        _, c = cohort(rows)
        self.assertEqual(c["cluster_count"], 1)
        self.assertEqual(len(set(c["splits"].values())), 1)
        self.assertEqual(c["unresolved_thread_record_ids"], ["hackernews:1", "hackernews:2"])
        # Merged copies retain their own ancestry, not just the canonical row's.
        merged = record("reddit:copy", "merged text", "reddit:root")
        merged["source"] = "reddit"
        merged["source_references"] = [rows[0]]
        _, c = cohort([merged, rows[1]])
        self.assertEqual(c["cluster_count"], 1)
        for edit in ({"status": 403}, {"native_id": "999"}, {"observation_id": None},
                     {"parent_id": "reddit:100"}, {"item_type": "unverified"},
                     {"url": "https://example.com/item/11"}):
            invalid = copy.deepcopy(rows)
            invalid[0]["root_provenance"]["chain"][1].update(edit)
            with self.subTest(edit=edit):
                _, c = cohort(invalid)
                self.assertEqual(c["cluster_count"], 2)

    def test_missing_thread_is_disclosed(self):
        _, c = cohort([record("a")])
        self.assertEqual(c["unresolved_thread_record_ids"], ["a"])
        self.assertFalse(c["historical_alpha_eligible"])

    def test_outside_window_and_unknown_vendor_rejected(self):
        for edit in ({"published_at": "2026-09-19T00:00:00Z"}, {"vendor_ids": ["unknown"]}):
            r = record("a")
            r.update(edit)
            with self.assertRaises(ValueError):
                cohort([r])

    def test_review_sample_keeps_all_selected(self):
        _, c = cohort([record(str(i), str(i), str(i)) for i in range(10)])
        sample = review_sample(c, {"2", "7"}, 3)
        self.assertEqual(len(sample["review_record_ids"]), 5)
        self.assertTrue({"2", "7"} <= set(sample["review_record_ids"]))
        self.assertFalse(sample["full_population_reviewed"])

    def positive(self, rid, vendor="ddog", episode="one"):
        return {"record_id": rid, "vendor_id": vendor, "qualifies": True,
                "episode_id": episode, "organization": "Example Co",
                "firsthand_report": True, "paid_product": True, "production_use": True,
                "completed_or_active_change": True, "business_link": True, "timing_supported": True,
                "evidence": [{"record_id": rid, "quote": "paid production systems"}]}

    def test_many_records_one_customer_are_one_episode(self):
        p, c = cohort([record(str(i)) for i in range(25)])
        review = {"cohort_sha256": digest(c), "label_origin": "independent_agent_review",
                  "records": [self.positive(str(i)) for i in range(25)] +
                  [{"record_id": str(i), "vendor_id": "net", "qualifies": False} for i in range(25)]}
        ids = {(str(i), "ddog") for i in range(25)}
        result = evaluate_operating_reviews(p, c, review, {a: ids for a in ARMS}, arm_status=complete_states(c))
        self.assertEqual(result["qualifying_episodes"], 1)
        self.assertEqual(result["decision"], "stop_family")
        self.assertFalse(result["alpha_proven"])

    def test_vendor_misattribution_is_not_credit(self):
        p, c = cohort([record("a")])
        review = {"cohort_sha256": digest(c), "label_origin": "independent_agent_review",
                  "records": [self.positive("a"), {"record_id": "a", "vendor_id": "net", "qualifies": False}]}
        result = evaluate_operating_reviews(p, c, review, {"jev": {("a", "net")}})
        self.assertEqual(result["arms"]["jev"]["qualifying_episodes"], 0)
        self.assertEqual(result["decision"], "incomplete")

    def test_unsupported_quote_or_missing_timing_rejected(self):
        p, c = cohort([record("a")])
        for update in ({"timing_supported": False}, {"evidence": [{"record_id": "a", "quote": "invented"}]}):
            r = self.positive("a")
            r.update(update)
            review = {"cohort_sha256": digest(c), "label_origin": "independent_agent_review", "records": [r]}
            with self.assertRaises(ValueError):
                evaluate_operating_reviews(p, c, review, {"jev": {("a", "ddog")}})

    def test_missing_failed_incomplete_or_unbound_controls_cannot_pass(self):
        p, c = cohort([record("a")], small_gate())
        rows = [self.positive("a"), {"record_id": "a", "vendor_id": "net", "qualifies": False}]
        review = {"cohort_sha256": digest(c), "label_origin": "independent_agent_review", "records": rows}
        selected = {arm: set() for arm in ARMS}
        selected["jev"] = {("a", "ddog")}
        for failure in ("missing", "failed", "unknown_coverage", "partial_population"):
            states = complete_states(c)
            if failure == "missing":
                states = None
            elif failure == "failed":
                states["baseline"]["status"] = "failed"
            elif failure == "unknown_coverage":
                states["baseline"]["response_coverage_complete"] = None
            else:
                states["baseline"]["evaluated_pairs"].pop()
            with self.subTest(failure=failure):
                result = evaluate_operating_reviews(p, c, review, selected, arm_status=states)
                self.assertEqual(result["decision"], "incomplete")
                self.assertEqual(result["jev_incremental_episode_ids"], [])
                self.assertFalse(result["all_arm_runs_complete"])
        states = complete_states(c)
        states["baseline"]["cohort_sha256"] = "wrong"
        with self.assertRaises(ValueError):
            evaluate_operating_reviews(p, c, review, selected, arm_status=states)

    def test_unknown_review_labels_do_not_become_negative_precision_or_full_recall(self):
        p, c = cohort([record("a"), record("b", "Other paid production systems changed")])
        rows = [self.positive("a"), {"record_id": "b", "vendor_id": "ddog", "qualifies": None}]
        rows += [{"record_id": rid, "vendor_id": "net", "qualifies": False} for rid in ("a", "b")]
        review = {"cohort_sha256": digest(c), "label_origin": "independent_agent_review", "records": rows}
        selected = {arm: {("a", "ddog"), ("b", "ddog")} for arm in ARMS}
        result = evaluate_operating_reviews(p, c, review, selected, arm_status=complete_states(c))
        self.assertEqual(result["arms"]["jev"]["precision_on_reviewed_records"], 1)
        self.assertEqual(result["arms"]["jev"]["resolved_selected_records"], 1)
        self.assertEqual(result["decision"], "incomplete")
        self.assertFalse(result["full_recall_established"])
        without_models = evaluate_operating_reviews(p, c, review, {})
        self.assertEqual(without_models["decision"], "incomplete")
        self.assertEqual(without_models["source_feasibility_decision"], "incomplete")
        self.assertFalse(result["selected_review_complete"])

    def test_eighty_deterministic_rejects_require_all_vendor_pairs(self):
        p, c = cohort([record(str(i), str(i) + " paid production systems changed", str(i)) for i in range(120)], small_gate())
        selected = {arm: set() for arm in ARMS}
        selected["jev"] = {("0", "ddog")}
        sample = review_sample(c, {"0"})
        required = set(sample["review_record_ids"])
        rows = [r for r in all_negative(c) if r["record_id"] in required]
        rows = [self.positive("0") if (r["record_id"], r["vendor_id"]) == ("0", "ddog") else r for r in rows]
        review = {"cohort_sha256": digest(c), "label_origin": "independent_agent_review", "records": rows[:-1]}
        result = evaluate_operating_reviews(p, c, review, selected, arm_status=complete_states(c))
        self.assertEqual(result["decision"], "incomplete")
        self.assertFalse(result["reject_review_complete"])
        review["records"] = rows
        result = evaluate_operating_reviews(p, c, review, selected, arm_status=complete_states(c))
        self.assertEqual(result["required_reject_sample_size"], 80)
        self.assertEqual(result["decision"], "eligible_for_financial_hypothesis_registration")
        self.assertEqual(result["jev_incremental_episode_ids"], ["one"])
        self.assertFalse(result["full_recall_established"])

    def test_unresolved_control_keeps_evidence_without_false_success_or_incremental_claim(self):
        p, c = cohort([record("a")], small_gate())
        rows = [self.positive("a"), {"record_id": "a", "vendor_id": "net", "qualifies": False}]
        review = {"cohort_sha256": digest(c), "label_origin": "independent_agent_review", "records": rows}
        selected = {arm: set() for arm in ARMS}
        selected["jev"] = {("a", "ddog")}
        states = complete_states(c)
        states["baseline"]["unknown_pairs"] = [{"record_id": "a", "vendor_id": "ddog"}]
        result = evaluate_operating_reviews(p, c, review, selected, arm_status=states)
        self.assertEqual(result["decision"], "no_demonstrated_jev_advantage")
        self.assertEqual(result["arms"]["baseline"]["qualifying_episodes"], 0)
        self.assertEqual(result["control_unknown_qualifying_episode_ids"], ["one"])
        self.assertEqual(result["jev_incremental_episode_ids"], [])

    def test_partial_source_coverage_cannot_pass_or_conclude_family_failure(self):
        p, c = cohort([record("a")], small_gate())
        c["source_coverage"][0]["truncated"] = None
        review = {"cohort_sha256": digest(c), "label_origin": "independent_agent_review", "records": all_negative(c)}
        result = evaluate_operating_reviews(p, c, review, {a: set() for a in ARMS}, arm_status=complete_states(c))
        self.assertEqual(result["decision"], "incomplete")
        self.assertFalse(result["source_capture_complete"])

    def test_zero_qualifiers_have_complete_review_but_undefined_recall(self):
        p, c = cohort([record("a")], small_gate())
        review = {"cohort_sha256": digest(c), "label_origin": "independent_agent_review",
                  "records": all_negative(c)}
        result = evaluate_operating_reviews(p, c, review, {a: set() for a in ARMS}, arm_status=complete_states(c))
        self.assertTrue(result["full_population_review_resolved"])
        self.assertTrue(result["comparison_complete"])
        self.assertFalse(result["full_recall_established"])
        without_models = evaluate_operating_reviews(p, c, review, {})
        self.assertEqual(without_models["decision"], "incomplete")
        self.assertEqual(without_models["source_feasibility_decision"], "stop_family")

    def test_split_scope_does_not_require_or_accept_other_split_outcomes(self):
        p, c = cohort([record(str(i), str(i) + " paid production systems changed", str(i)) for i in range(40)], small_gate())
        rows = all_negative(c, "development")
        review = {"cohort_sha256": digest(c), "split": "development", "label_origin": "independent_agent_review", "records": rows}
        result = evaluate_operating_reviews(p, c, review, {a: set() for a in ARMS}, arm_status=complete_states(c, "development"))
        self.assertEqual(result["decision"], "stop_family")
        self.assertTrue(result["comparison_complete"])
        other = next(r for r in all_negative(c) if c["splits"][r["record_id"]] == "evaluation")
        review["records"].append(other)
        with self.assertRaises(ValueError):
            evaluate_operating_reviews(p, c, review, {a: set() for a in ARMS}, arm_status=complete_states(c, "development"))

    def test_duplicated_body_cannot_inflate_episode_count_with_new_ids(self):
        p, c = cohort([record("a"), record("b")])
        review = {"cohort_sha256": digest(c), "label_origin": "independent_agent_review",
                  "records": [self.positive("a", episode="one"), self.positive("b", episode="two")]}
        with self.assertRaisesRegex(ValueError, "multiple economic episodes"):
            evaluate_operating_reviews(p, c, review, {"jev": {("a", "ddog"), ("b", "ddog")}})

    def test_qualifiers_cannot_borrow_another_customers_evidence_or_accept_boolean_numbers(self):
        p, c = cohort([record("a", "paid production systems A", "A"), record("b", "paid production systems B", "B")])
        for update in ({"qualifies": 1}, {"evidence": [{"record_id": "b", "quote": "paid production systems"}]},
                       {"evidence": [{"record_id": "a", "quote": "paid production systems"}, {"record_id": "b", "quote": "paid production systems"}]}):
            row = self.positive("a")
            row.update(update)
            review = {"cohort_sha256": digest(c), "label_origin": "independent_agent_review", "records": [row]}
            with self.subTest(update=update), self.assertRaises(ValueError):
                evaluate_operating_reviews(p, c, review, {"jev": {("a", "ddog")}})


if __name__ == "__main__":
    unittest.main()
