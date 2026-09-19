import unittest

from jev_alpha.readthrough_results import headline_direction, summarize_routes


def number_record():
    return {"comparability_supported": True,
            "prior": {"period": "FY2030", "basis": "net", "kind": "forecast", "low": 10, "high": 14},
            "current": {"period": "FY2030", "basis": "net", "kind": "forecast", "low": 12, "high": 16}}


def routes(reference="rejected", native="unresolved", categorical="supported"):
    return [{"fixture_id": "invented", "route_id": "equipment", "arm": arm,
             "reference_status": reference, "status": status}
            for arm, status in (("jev", native), ("jev_categorical", categorical), ("baseline", "unresolved"))]


class ReadthroughResultsTests(unittest.TestCase):
    def test_overlapping_ranges_are_only_a_midpoint_headline_heuristic(self):
        self.assertEqual(headline_direction(number_record()), "increase")

    def test_open_bounds_and_incomparable_periods_stay_unknown(self):
        r = number_record()
        r["current"]["high"] = None
        self.assertEqual(headline_direction(r), "unknown")
        r = number_record()
        r["prior"]["period"] = "FY2029"
        self.assertEqual(headline_direction(r), "unknown")
        r = number_record()
        r["prior"]["kind"] = "actual"
        self.assertEqual(headline_direction(r), "unknown")

    def test_accounting_basis_change_or_unverified_comparability_abstains(self):
        r = number_record()
        r["current"]["basis"] = "gross"
        self.assertEqual(headline_direction(r), "unknown")
        r = number_record()
        r["comparability_supported"] = "true"
        self.assertEqual(headline_direction(r), "unknown")

    def test_invalid_numbers_do_not_become_signals(self):
        for value in (float("nan"), float("inf"), True, -1):
            r = number_record()
            r["current"]["low"] = value
            self.assertEqual(headline_direction(r), "unknown")

    def test_avoiding_an_unsupported_positive_is_not_an_exact_match(self):
        result = summarize_routes(routes())
        change = result["native_changes"][0]
        self.assertTrue(change["avoids_unsupported_positive"])
        self.assertFalse(change["improves_exact_match"])
        self.assertEqual(result["metrics"]["jev"]["exact_reference_matches"], 0)
        self.assertEqual(result["metrics"]["jev_categorical"]["unsupported_positive_claims"], 1)

    def test_abstention_can_lose_a_correct_supported_positive(self):
        result = summarize_routes(routes(reference="supported"))
        self.assertTrue(result["native_changes"][0]["loses_supported_positive"])
        self.assertEqual(result["metrics"]["jev"]["missed_supported_references"], 1)

    def test_identical_routes_do_not_show_native_benefit(self):
        result = summarize_routes(routes(native="rejected", categorical="rejected"))
        self.assertEqual(result["native_changes"], [])

    def test_missing_and_duplicate_arms_are_rejected(self):
        for rows in (routes()[:2], routes() + routes()[:1]):
            with self.assertRaises(ValueError):
                summarize_routes(rows)


if __name__ == "__main__":
    unittest.main()
