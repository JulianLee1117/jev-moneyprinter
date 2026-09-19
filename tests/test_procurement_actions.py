import copy
import hashlib
from pathlib import Path
import tempfile
import unittest

from jev_alpha.procurement_actions import attach_review, review_template
from jev_alpha.procurement_returns import evaluate
from tests.test_procurement_returns import event, fixture


class ProcurementActionReviewTests(unittest.TestCase):
    def test_templates_and_missing_reviews_leave_selected_return_unknown(self):
        signals, market = fixture([event()])
        market["actions"] = {}
        review = review_template(signals, market, {})
        self.assertEqual(len(review["records"]), 7)
        self.assertTrue(all(r["status"] == "unreviewed" for r in review["records"]))
        review["records"] = []
        with tempfile.TemporaryDirectory() as tmp:
            derived = attach_review(Path(tmp), signals, market, review, {}, out=Path(tmp) / "derived.json")
        account = evaluate(signals, derived, {})["accounts"]["jev|manual|all"]["10"]
        self.assertEqual(account["status"], "unknown")
        self.assertIsNone(account["pnl_usd"])
        self.assertEqual(derived["action_review_summary"]["unresolved_intervals"], 7)
        self.assertEqual(market["actions"], {})

    def test_bound_reviews_hash_evidence_and_write_only_a_new_receipt(self):
        signals, market = fixture([event()])
        market["actions"] = {}
        review = review_template(signals, market, {})
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = b"Independently reviewed primary issuer and exchange fixture records."
            (root / "evidence.txt").write_bytes(raw)
            for row in review["records"]:
                row.update(status="none", reviewed=True, reviewer="fixture reviewer", coverage_complete=True,
                           coverage_statement="Reviewed the entire interval and all supported action categories.",
                           evidence=[{"source_url": "https://issuer.example/dated-record",
                                      "capture_path": "evidence.txt", "capture_sha256": hashlib.sha256(raw).hexdigest(),
                                      "locator": "Independent fixture action assessment"}])
            altered = copy.deepcopy(market)
            altered["status"] = "different"
            with self.assertRaisesRegex(ValueError, "not bound"):
                attach_review(root, signals, altered, review, {}, out=root / "rejected.json")
            derived = attach_review(root, signals, market, review, {}, out=root / "derived.json")
            self.assertEqual(derived["quotes"], market["quotes"])
            self.assertEqual(evaluate(signals, derived, {})["accounts"]["jev|manual|all"]["10"]["status"], "complete")
            changed = copy.deepcopy(derived)
            changed["actions"] = {}
            with self.assertRaisesRegex(ValueError, "Market receipt hash mismatch"):
                evaluate(signals, changed, {})
            with self.assertRaises((ValueError, FileExistsError)):
                attach_review(root, signals, market, review, {}, out=root / "derived.json")
            (root / "evidence.txt").write_bytes(b"modified")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                attach_review(root, signals, market, review, {}, out=root / "other.json")
            self.assertFalse((root / "other.json").exists())

    def test_empty_evidence_is_not_none_and_dividend_must_be_owned(self):
        signals, market = fixture([event()])
        review = review_template(signals, market, {})
        row = next(r for r in review["records"] if r["symbol"] == "ORN")
        row.update(status="none", reviewed=True, reviewer="fixture", coverage_complete=True,
                   coverage_statement="Reviewed complete interval.")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(ValueError, "captured evidence"):
                attach_review(root, signals, market, review, {}, out=root / "empty.json")
            raw = b"Dividend fixture"
            (root / "evidence.txt").write_bytes(raw)
            row.update(status="cash_dividend", evidence=[{"source_url": "https://issuer.example/dividend",
                "capture_path": "evidence.txt", "capture_sha256": hashlib.sha256(raw).hexdigest(), "locator": "ex-date"}],
                dividends=[{"ex_date": row["entry_date"], "amount_per_share": .1}])
            with self.assertRaisesRegex(ValueError, "outside the owned interval"):
                attach_review(root, signals, market, review, {}, out=root / "bad-dividend.json")
            row["dividends"][0]["ex_date"] = row["exit_date"]
            derived = attach_review(root, signals, market, review, {}, out=root / "dividend.json")
            key = "|".join(("ORN", row["entry_date"], row["exit_date"]))
            self.assertEqual(derived["actions"][key]["cash_dividend_per_entry_share"], .1)
            report = evaluate(signals, derived, {})
            self.assertIsNotNone(report["windows"][0]["outcome"]["return"])
            self.assertIsNone(report["windows"][0]["peer_excess"])


if __name__ == "__main__":
    unittest.main()
