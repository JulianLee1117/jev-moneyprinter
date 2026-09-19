import copy
import unittest
from unittest.mock import patch

from jev_alpha.claims import DIMENSIONS, prepare_claim_screen, select_claim_evidence
from jev_alpha.fanout import prepare_fanout, select_evidence
from jev_alpha.jev import MODEL, JevValidationError, estimate_request


def state(texts=None):
    texts = texts if texts is not None else ["Pending customs recovery.", "Refund received.", "Revenue rose."]
    return {"episode_manifest": {"document_id": "issuer-filing", "source_sha256": "a"*64,
            "title": "Synthetic filing", "source_url": "https://example.test/filing", "passage_selection": "all"},
            "current_source_passages_with_ids": [{"passage_id": f"source-{i}", "text": text} for i,text in enumerate(texts)]}


def responses(plan, overrides=None):
    overrides = overrides or {}
    result = {}
    for chunk in plan["chunks"]:
        answers = {}
        for question, fid in chunk["questions"].items():
            dimension = chunk["question_dimensions"][question]
            value = overrides.get((fid, dimension))
            if dimension == "claim_lifecycle":
                probs = {"outstanding": 0, "realized_or_settled": 0, "mixed": 0, "historical_or_other": 1, "unclear": 0}
                if value is not None: probs = value
                answers[question] = {"type": "choice", "choice": max(probs,key=probs.get), "probabilities": probs}
            else:
                answers[question] = {"type": "noul", "noul": 0 if value is None else value}
        result[chunk["chunk_id"]] = {"model": MODEL, "answers": answers, "usage": {"input_tokens": 100, "output_tokens": 100}}
    return result


class ClaimScreenTests(unittest.TestCase):
    def test_four_independent_questions_per_fragment_no_network(self):
        source = state()
        source["episode_manifest"].update(symbol="UFPI", role="latest", report_date="2026-06-27")
        with patch("jev_alpha.jev.urllib.request.build_opener") as network:
            plan = prepare_claim_screen(source)
        network.assert_not_called()
        self.assertEqual(len(plan["fragments"])*4, sum(len(c["questions"]) for c in plan["chunks"]))
        self.assertEqual(plan["question_dimensions"], list(DIMENSIONS))
        context=plan["chunks"][0]["request"]["state"]["evidence"]["document_context"]
        self.assertEqual((context["symbol"],context["role"],context["report_date"]),("UFPI","latest","2026-06-27"))
        for chunk in plan["chunks"]:
            for question, fid in chunk["questions"].items():
                self.assertIn(repr(fid), chunk["request"]["questions"][question]["instructions"])

    def test_long_unicode_empty_and_table_text_split_losslessly_under_limits(self):
        source = state([" customs 鋼板 🙂\n"*300, "", "Claims | $250 | received\n"])
        plan = prepare_claim_screen(source, max_input_tokens=6500, max_questions=8)
        for original in source["current_source_passages_with_ids"]:
            parts=[f for f in plan["fragments"] if f["parent_passage_id"]==original["passage_id"]]
            self.assertEqual("".join(f["text"] for f in parts), original["text"])
            self.assertEqual(parts[0]["offset_start"],0)
            self.assertEqual(parts[-1]["offset_end"],len(original["text"]))
        for chunk in plan["chunks"]:
            self.assertLessEqual(len(chunk["questions"]),8)
            self.assertLessEqual(estimate_request(chunk["request"])["conservative_input_tokens"],6500)
        self.assertEqual(select_claim_evidence(plan,responses(plan))["status"], "complete")

    def test_default_chunks_no_more_than_forty_questions(self):
        plan=prepare_claim_screen(state(["text"]*31))
        self.assertTrue(all(len(c["questions"])<=40 for c in plan["chunks"]))
        self.assertGreaterEqual(len(plan["chunks"]),4)

    def test_lifecycle_mass_or_offset_or_update_and_amount_only_not_selected(self):
        plan=prepare_claim_screen(state(["text"]*5))
        fids=[f["passage_id"] for f in plan["fragments"]]
        uncertain={"outstanding": .08,"realized_or_settled":0,"mixed":.07,"historical_or_other":.79,"unclear":.06}
        values={(fids[0],"claim_lifecycle"):uncertain,(fids[1],"cash_offset"):.2,
                (fids[2],"subsequent_update"):.2,(fids[3],"explicit_amount"):1}
        result=select_claim_evidence(plan,responses(plan,values),neighbor_radius=0)
        self.assertEqual(result["selected_fragment_ids"],fids[:3])
        self.assertAlmostEqual(result["decisions"][0]["pending_or_unclear_probability"],.21)
        self.assertEqual(result["decisions"][3]["answers"]["explicit_amount"]["noul"],1)

    def test_neighbors_expand_once_and_reject_audit_deterministic(self):
        plan=prepare_claim_screen(state(["text"]*9));fids=[f["passage_id"] for f in plan["fragments"]]
        raw=responses(plan,{(fids[4],"cash_offset"):1})
        result=select_claim_evidence(plan,raw)
        self.assertEqual(result["selected_fragment_ids"],fids[3:6])
        self.assertEqual(len(result["audit_sample_fragment_ids"]),3)
        self.assertEqual(result,select_claim_evidence(plan,raw))

    def test_missing_invalid_chunks_preserve_all_evidence(self):
        plan=prepare_claim_screen(state(["text"]*12),max_questions=4)
        raw=responses(plan)
        del raw[plan["chunks"][0]["chunk_id"]]
        raw[plan["chunks"][1]["chunk_id"]]["answers"]={}
        result=select_claim_evidence(plan,raw,neighbor_radius=0)
        self.assertEqual(result["status"],"incomplete")
        self.assertEqual(result["selected_fragment_ids"],[f["passage_id"] for f in plan["fragments"][:2]])
        self.assertEqual(len(result["missing_chunks"]),1);self.assertEqual(len(result["invalid_chunks"]),1)

    def test_schema_separates_trade_and_claim_selectors(self):
        claims=prepare_claim_screen(state());trade=prepare_fanout(state())
        with self.assertRaises(JevValidationError):select_evidence(claims,{})
        with self.assertRaises(JevValidationError):select_claim_evidence(trade,{})

    def test_tampered_dimension_duplicate_unknown_chunk_and_gap_rejected(self):
        original=prepare_claim_screen(state())
        for issue in ["dimension","duplicate","unknown","gap"]:
            plan=copy.deepcopy(original);raw={}
            if issue=="dimension":
                name=next(iter(plan["chunks"][0]["question_dimensions"]))
                plan["chunks"][0]["question_dimensions"][name]="wrong"
            elif issue=="duplicate":plan["chunks"].append(copy.deepcopy(plan["chunks"][0]))
            elif issue=="unknown":raw["unknown"]={}
            else:plan["fragments"][0]["offset_start"]=1
            with self.subTest(issue=issue),self.assertRaises(JevValidationError):select_claim_evidence(plan,raw)

    def test_invalid_limits_and_pure_inputs(self):
        source=state();before=copy.deepcopy(source)
        plan=prepare_claim_screen(source);raw=responses(plan);old=copy.deepcopy((plan,raw))
        select_claim_evidence(plan,raw)
        self.assertEqual(source,before);self.assertEqual((plan,raw),old)
        for limit in [0,True,3]:
            with self.assertRaises(JevValidationError):prepare_claim_screen(source,max_questions=limit)
        with self.assertRaises(JevValidationError):select_claim_evidence(plan,raw,cutoff=float("nan"))


if __name__ == "__main__":unittest.main()
