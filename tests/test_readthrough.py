import copy
import unittest

from jev_alpha.readthrough import evaluate, evaluate_route, prepare, validate_plan, valid_jev_identity


def fixture():
    return {"fixtures": [{"fixture_id": "case-a", "issuer": "Example",
        "sources": [{"source_id": "current", "url": "https://example.org/release",
                     "date": "2025-01-01", "role": "current", "excerpt": "Buildings rise; equipment falls.",
                     "expected": "leaked-label", "future_return": "leaked-outcome"}],
        "questions": {"equipment": {"type": "choice", "instructions": "Equipment direction?",
            "criteria": {"up": "Explicit increase", "down": "Explicit decrease", "unknown": "Unstated"}}},
        "expected": {"equipment": "down"}, "future_return": .90,
        "routes": [{"route_id": "supplier-demand-down", "conditions": {"equipment": "down"}}]}]}


def choice(category, probability):
    return {"choice": category, "probabilities": {category: probability}}


class ReadthroughTests(unittest.TestCase):
    def test_labels_and_future_outcomes_do_not_enter_model_state(self):
        original = fixture()
        before = copy.deepcopy(original)
        plan = prepare(original)
        state = str(plan["entries"][0]["request"]["state"])
        self.assertNotIn("leaked", state)
        self.assertNotIn("future_return", state)
        self.assertEqual(original, before)

    def test_incomplete_or_invalid_labels_block_preparation(self):
        rows = fixture()
        rows["fixtures"][0]["expected"] = {}
        with self.assertRaises(ValueError):
            prepare(rows)
        rows["fixtures"][0]["expected"] = {"equipment": "invented"}
        with self.assertRaises(ValueError):
            prepare(rows)

    def test_confident_false_prunes_path_despite_missing_other_condition(self):
        result = evaluate_route({"a": "yes", "b": "yes"}, {"b": choice("no", .95)})
        self.assertEqual(result["status"], "rejected")
        self.assertEqual(result["condition_states"], {"a": "unknown", "b": "false"})

    def test_low_confidence_and_explicit_unknown_abstain(self):
        self.assertEqual(evaluate_route({"a": "yes"}, {"a": choice("yes", .79)})["status"], "unresolved")
        self.assertEqual(evaluate_route({"a": "yes"}, {"a": choice("unknown", .99)})["status"], "unresolved")
        self.assertEqual(evaluate_route({"a": "yes"}, {"a": "unknown"})["status"], "unresolved")

    def test_control_uses_labels_without_fabricating_probabilities(self):
        plan = prepare(fixture())
        report = evaluate(plan, {("baseline", "case-a"): {"answers": {"equipment": "down"}}})
        self.assertEqual(report["metrics"]["baseline"]["correct"], 1)
        self.assertIsNone(report["metrics"]["baseline"]["confident_answers"])
        self.assertEqual(report["metrics"]["jev"]["registered_questions"], 1)
        self.assertEqual(report["metrics"]["jev"]["coverage"], 0)
        self.assertIsNone(report["metrics"]["jev"]["accuracy_answered"])

    def test_probabilities_are_not_multiplied(self):
        result = evaluate_route({"a": "yes", "b": "yes"},
                                {"a": choice("yes", .85), "b": choice("yes", .90)})
        self.assertEqual(result["status"], "supported")
        self.assertFalse(result["is_joint_probability"])
        self.assertNotIn("probability", result)

    def test_frozen_plan_owns_labels_and_detects_changes(self):
        source = fixture()
        plan = prepare(source)
        source["fixtures"][0]["expected"]["equipment"] = "up"
        self.assertEqual(plan["entries"][0]["expected"]["equipment"], "down")
        validate_plan(plan)
        plan["entries"][0]["expected"]["equipment"] = "up"
        with self.assertRaises(ValueError):
            validate_plan(plan)

    def test_duplicate_fixture_changes_are_rejected(self):
        from jev_alpha.experiment import digest
        plan = prepare(fixture())
        plan["entries"].append(copy.deepcopy(plan["entries"][0]))
        plan["entries_sha256"] = digest(plan["entries"])
        with self.assertRaises(ValueError):
            validate_plan(plan)

    def test_jev_serving_identity_rejects_other_model_and_invalid_suffix(self):
        self.assertTrue(valid_jev_identity("typesafe/jev-1.13-20260917"))
        self.assertTrue(valid_jev_identity("typesafe/jev-1.13"))
        for model in ("typesafe/jev-1.14", "typesafe/jev-1.13-20269999", "openai/gpt-5.4-mini"):
            self.assertFalse(valid_jev_identity(model))


if __name__ == "__main__":
    unittest.main()
