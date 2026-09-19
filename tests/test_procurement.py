import json
import hashlib
from pathlib import Path
import tempfile
import unittest

from jev_alpha.procurement import (alias_present, aliases_asof, amount_candidates, chunks, digest,
    evidence_state, freeze_signals, inference_gate, later_report_date, linked_bid_targets, packet_request, prepare, review_template, rules_answers, signal_for)
from jev_alpha.procurement_models import MODELS, build_chat_request, build_panel
from jev_alpha.procurement_reference import annual_revenues, revenue_asof


class ProcurementWorkflowTests(unittest.TestCase):
    def test_full_text_partition_and_currency_units(self):
        text = ("é contract $12.5 million\n" * 1600) + "end"
        parts = list(chunks(text, 1300))
        self.assertEqual("".join(p[2] for p in parts), text)
        self.assertTrue(all(len(p[2].encode()) <= 1300 for p in parts))
        rows = amount_candidates("$12.5 million and $3,456,789.12", 100)
        self.assertEqual([r["value_usd"] for r in rows], [12500000, 3456789.12])
        self.assertEqual(rows[0]["candidate_id"], "a100")
        malformed = amount_candidates("$2,182032.19 and $2,925.214.00; budget ~$37-$40M")
        self.assertEqual(len(malformed), 4)
        self.assertTrue(all(r["value_usd"] is None for r in malformed))
        self.assertEqual(amount_candidates("Contract amount USD 21,496,775.00")[0]["value_usd"],21496775)
        self.assertIsNone(amount_candidates("Amounts in thousands\nContract award $25,000")[0]["value_usd"])
        mixed = amount_candidates("Appendix: amounts in thousands\nAward $20 million; table row $25,000")
        self.assertEqual(mixed[0]["value_usd"], 20000000)
        self.assertIsNone(mixed[1]["value_usd"])

    def test_mapping_and_revenues_do_not_use_future_information(self):
        self.assertTrue(alias_present("C.W. Roberts Contracting, Incorporated", "C. W. ROBERTS CONTRACTING, INC."))
        self.assertTrue(alias_present("James Construction Group, LLC", "James Construction Group, L.L.C."))
        self.assertFalse(alias_present("James Construction Group, LLC", "SER Construction Partners, LLC"))
        alias = {"symbol":"ORN", "alias":"Orion Marine Construction", "verified":True,
                 "effective_from":"2025-01-01", "evidence_date":"2025-03-01", "source_url":"https://issuer.example/report"}
        self.assertFalse(aliases_asof({"aliases":[alias]}, "2025-02-01"))
        self.assertTrue(aliases_asof({"aliases":[alias]}, "2025-04-01"))
        facts = {"facts":{"us-gaap":{"Revenues":{"units":{"USD":[
            {"start":"2024-01-01","end":"2024-12-31","filed":"2025-03-01","form":"10-K","fp":"FY","fy":2024,"val":800000000,"accn":"a"},
            {"start":"2024-01-01","end":"2024-12-31","filed":"2026-03-01","form":"10-K","fp":"FY","fy":2025,"val":900000000,"accn":"b"}]}}}}}
        records = annual_revenues(facts,"ORN")
        self.assertIsNone(revenue_asof(records,"ORN","2025-03-01"))
        self.assertEqual(revenue_asof(records,"ORN","2025-04-01")["revenue_usd"],800000000)
        self.assertEqual(later_report_date("Staff report\nDate: June 9, 2026\nContract award", "2025-12-09")["date"],"2026-06-09")
        self.assertIsNone(later_report_date("Completion date: June 9, 2026\nThe project opens in 2026", "2025-12-09"))

    def test_report_date_repair_moves_holdout_and_preserves_meeting_without_using_completion_date(self):
        protocol = json.loads(Path("research/experiments/procurement-protocol.v1.json").read_text())
        texts = {
            "later_report": "STAFF REPORT\nDate:\nOctober 6, 2025\nSUBJECT\nContract P1001: proposed new work $20 million.",
            "future_report": "STAFF REPORT\nDate:\nJune 9, 2026\nSUBJECT\nContract P1002: proposed new work $20 million.",
            "completion_table": "PROCUREMENT SUMMARY\nData current as of 9/2/25\nOriginal Complete\nDate:\nJanuary 2, 2026 Pending\nContract P1003: proposed new work $20 million.",
            "old_minutes": "BUSINESS MEETING\nJUNE 17, 2025, 9:30 AM\nThe board approved Contract P1004: new work $20 million.",
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            documents = []
            for document_id, text in texts.items():
                path = root / (document_id + ".txt")
                path.write_bytes(text.encode("utf-8"))
                documents.append({"document_id": document_id,
                    "source_id": "metro" if document_id == "completion_table" else "kingcounty",
                    "url": "https://example.org/" + document_id,
                    "association_date": "2025-09-30", "association_dates": ["2025-09-30", "2025-10-14"],
                    "availability": "unverified_historical", "published_at": None,
                    "blob_path": path.name, "text_path": path.name,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "extraction_status": "extracted_plain_text"})
                if document_id == "old_minutes":
                    documents[-1].update(source_id="port_tampa", kind="minutes", title="June 17, 2025 Brd Mtg Minutes",
                                         association_date="2025-08-19", association_dates=["2025-08-19"])
            result = prepare(root, {"documents": documents}, {"aliases": []}, {"annual_revenues": []}, protocol)
        records = {entry["document_id"]: entry for entry in result["records"]}
        self.assertEqual(set(records), {"later_report", "completion_table"})
        repaired = records["later_report"]
        self.assertEqual(repaired["association_date"], "2025-10-06")
        self.assertEqual(repaired["split"], "holdout")
        self.assertEqual(repaired["original_meeting_association_date"], "2025-09-30")
        self.assertEqual(repaired["original_association_dates"], ["2025-09-30", "2025-10-14"])
        self.assertEqual(repaired["association_date_evidence"], {"date": "2025-10-06", "evidence": "Date:\nOctober 6, 2025"})
        self.assertEqual(repaired["association_date_basis"], "later_explicit_report_header_not_publication")
        self.assertEqual(repaired["availability"], "unverified_historical")
        self.assertIsNone(repaired["published_at"])
        self.assertEqual(evidence_state(repaired)["association_date"], "2025-10-06")
        completion = records["completion_table"]
        self.assertEqual(completion["association_date"], "2025-09-30")
        self.assertEqual(completion["split"], "development")
        self.assertIsNone(completion["association_date_evidence"])
        self.assertEqual(len(result["source_failures"]), 2)
        failures = {entry["document_id"]: entry for entry in result["source_failures"]}
        excluded = failures["future_report"]
        self.assertEqual(excluded["document_id"], "future_report")
        self.assertEqual(excluded["reason"], "known_post_association_report_date")
        self.assertEqual(excluded["conflicting_header"]["date"], "2026-06-09")
        old = failures["old_minutes"]
        self.assertEqual(old["reason"], "explicit_meeting_document_outside_period")
        self.assertEqual(old["association_date"], "2025-08-19")
        self.assertEqual(old["document_date_evidence"]["date"], "2025-06-17")

    def fixture(self):
        alias = {"alias":"Orion Marine Construction", "ownership_share":1}
        amount = {"candidate_id":"a0","value_usd":20000000,"evidence":"Orion Marine Construction awarded contract amount $20 million"}
        state = {"text":amount["evidence"],"passages":[{"text":amount["evidence"]}],"issuer_candidates":[
            {"candidate_id":"orn","symbol":"ORN","aliases":[alias["alias"]],"verified_aliases":[alias]}],
            "amount_candidates":[amount],"target_amount_id":"a0","prior_records":[]}
        entry = {"packet_id":"p","document_id":"d","project_id":"P1","source_id":"port_tampa",
            "association_date":"2025-10-02","split":"holdout","availability":"unverified_historical",
            "published_at":None,"source_url":"https://example.org/agency","request":build_panel(state),"target_amount":amount}
        revenue = {"symbol":"ORN","published_date":"2025-03-01","period_end":"2024-12-31", "revenue_usd":800000000,"source_url":"https://example.org/annual"}
        return entry, {"annual_revenues":[revenue]}

    def test_budget_amount_and_jv_are_not_firm_attributable_awards(self):
        entry,inputs = self.fixture()
        answer = {"recipient":"orn","stage":"approved","amount":"a0","amount_kind":"contractor_value","scope":"new_work"}
        review = {"status":"reviewed","project_identity_verified":True}
        self.assertEqual(signal_for(entry,answer,inputs,review)["status"],"signal")
        self.assertEqual(signal_for(entry,{**answer,"amount_kind":"project_budget"},inputs,review)["status"],"no_signal")
        entry["target_amount"]["evidence"] += " joint venture"
        evidence_state(entry)["passages"][0]["text"] += " joint venture"
        self.assertEqual(signal_for(entry,answer,inputs,review)["status"],"unknown")

    def test_attribution_uses_shared_passages_and_opening_without_proximity_gate(self):
        entry, inputs = self.fixture()
        text = "Orion Marine Construction is the contractor. " + "Detailed project scope. " * 30 + "New contract award $20 million."
        amount = amount_candidates(text)[0]
        entry["target_amount"] = amount
        state = evidence_state(entry)
        state["passages"] = [{"text": text}]
        answer = {"recipient":"orn", "stage":"recommendation", "amount":amount["candidate_id"],
                  "amount_kind":"contractor_value", "scope":"new_work", "prior_known":"unknown"}
        review = {"status":"reviewed", "project_identity_verified":True}
        self.assertNotIn("Orion", amount["evidence"])
        self.assertEqual(signal_for(entry, answer, inputs, review)["status"], "signal")
        state["passages"] = [{"text":amount["evidence"]}]
        self.assertEqual(signal_for(entry, answer, inputs, review)["reason"], "attribution_or_jv_share_unknown")
        state["document_opening_context"] = "Orion Marine Construction is the contractor."
        self.assertEqual(signal_for(entry, answer, inputs, review)["status"], "signal")
        state["document_opening_context"] += " This award is to a joint venture."
        self.assertEqual(signal_for(entry, answer, inputs, review)["status"], "unknown")

    def test_fixed_price_nte_is_distinct_from_unexercised_capacity(self):
        entry, _ = self.fixture()
        entry["target_amount"]["evidence"] = "Recommend award to Orion Marine Construction of a firm fixed price contract, not to exceed $20 million."
        self.assertEqual(rules_answers(entry)["amount_kind"], "contractor_value")
        entry["target_amount"]["evidence"] = "Recommend award to Orion Marine Construction of a fixed-price IDIQ contract, maximum $20 million with no guaranteed spend."
        self.assertEqual(rules_answers(entry)["amount_kind"], "ceiling")

    def test_explicit_project_prefix_variants_share_prior_records(self):
        protocol = json.loads(Path("research/experiments/procurement-protocol.v1.json").read_text())
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            docs = []
            for index, prefix in enumerate(("Contract No. AE60979000", "Contract AE60979000")):
                text = prefix + ": recommend award to Orion Marine Construction of $20 million."
                path = root / (str(index) + ".txt")
                path.write_text(text, encoding="utf-8")
                docs.append({"document_id":str(index), "source_id":"metro", "url":"https://example.org/"+str(index),
                             "association_date":"2025-07-0"+str(index+1), "blob_path":path.name,
                             "text_path":path.name, "sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
                             "extraction_status":"extracted_plain_text"})
            result = prepare(root, {"documents":docs}, {"aliases":[]}, {"annual_revenues":[]}, protocol)
        first, second = result["records"]
        self.assertEqual(first["project_id"], "AE60979000")
        self.assertEqual(first["project_id"], second["project_id"])
        self.assertEqual(first["project_id_source_span"], "Contract No. AE60979000")
        self.assertEqual(second["earlier_project_records"][0]["document_id"], "0")
        self.assertEqual(evidence_state(second)["prior_records"][0]["document_id"], "0")

    def test_capability_skips_remain_unknown_and_keep_low_amount_negative(self):
        entry, inputs = self.fixture()
        state = evidence_state(entry)
        cases = [(None, "no_target_amount"),
                 ({**entry["target_amount"], "value_usd":None}, "source_amount_unresolved"),
                 (entry["target_amount"], "no_verified_alias_in_supplied_evidence")]
        for amount, reason in cases:
            with self.subTest(reason=reason):
                state["passages"] = [{"text":"Unidentified contractor, contract amount $20 million."}]
                required, skip, status = inference_gate(amount, state, 16000000)
                self.assertFalse(required)
                self.assertEqual(skip, reason)
                skipped = {**entry, "model_required":required, "skip_inference_reason":skip,
                           "deterministic_signal_status":status}
                result = signal_for(skipped, {}, inputs, None)
                self.assertEqual(result["status"], "unknown")
                self.assertIsNone(result["materiality"])
        low = {**entry["target_amount"], "value_usd":5000000}
        self.assertEqual(inference_gate(low, state, 16000000),
                         (False, "amount_below_all_six_materiality_floors", "no_signal"))
        state["document_opening_context"] = "Orion Marine Construction is the contractor."
        self.assertEqual(inference_gate(entry["target_amount"], state, 16000000), (True,None,None))

    def test_external_preparation_preserves_grouped_targets_and_legacy_crlf_text(self):
        protocol = json.loads(Path("research/experiments/procurement-protocol.v1.json").read_text())
        _, fixture_inputs = self.fixture()
        revenue = fixture_inputs["annual_revenues"][0]
        references = {"annual_revenues": [{**revenue, "symbol": symbol} for symbol in protocol["symbols"]]}
        issuer_map = {"aliases": [{"symbol": "ORN", "alias": "Orion Marine Construction",
            "ownership_share": 1, "verified": True, "effective_from": "2025-03-01",
            "evidence_date": "2025-03-01", "source_url": "https://example.org/annual"}]}
        text = ("Contract P1000: Orion Marine Construction recommends new work.\n"
                "Small values $1 million and $2 million.\n"
                "Material awards $20 million and $21 million.\n"
                "Unresolved values $2,182032.19 and $2,925.214.00.\n"
                + "Detailed café project scope without additional amounts.\n" * 100)
        expected = amount_candidates(text)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_path = root / "source.bin"
            raw_path.write_bytes(text.encode("utf-8"))
            text_path = root / (hashlib.sha256(text.encode("utf-8")).hexdigest() + ".txt")
            text_path.write_bytes(text.replace("\n", "\r\n").encode("utf-8"))
            self.assertNotEqual(hashlib.sha256(text_path.read_bytes()).hexdigest(), text_path.stem)
            document = {"document_id": "legacy", "source_id": "metro", "url": "https://example.org/source",
                "association_date": "2025-07-01", "blob_path": raw_path.name, "text_path": text_path.name,
                "sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(), "extraction_status": "extracted_plain_text"}
            result = prepare(root, {"documents": [document]}, issuer_map, references, protocol, external_requests=True)
            self.assertEqual(result["source_failures"], [])
            self.assertEqual(result["request_storage"], "content_addressed_external")
            represented, cores = [], {}
            for row in result["records"]:
                self.assertNotIn("request", row)
                request_bytes = Path(row["request_path"]).read_bytes()
                request = packet_request(row)
                self.assertEqual(hashlib.sha256(request_bytes).hexdigest(), row["request_sha256"])
                self.assertEqual(digest(request), row["request_sha256"])
                self.assertEqual(digest(request["state"]), row["input_sha256"])
                state = evidence_state(row)
                coverage = state["coverage"]
                start, end = coverage["chunk_start"], coverage["chunk_end"]
                visible = state["passages"][0]["text"]
                self.assertEqual(visible, text[coverage["context_start"]:coverage["context_end"]])
                cores[start] = visible[start-coverage["context_start"]:end-coverage["context_start"]]
                self.assertEqual(state["document_opening_context"], text[:2000])
                self.assertEqual(len(state["issuer_candidates"]), len(protocol["symbols"]))
                self.assertEqual(next(c for c in state["issuer_candidates"] if c["symbol"] == "ORN")["aliases"],
                                 ["Orion Marine Construction"])
                represented.extend(row.get("aggregated_amount_candidates", [row["target_amount"]])
                                   if row.get("target_amount") else [])
            self.assertEqual("".join(cores[start] for start in sorted(cores)), text)
            self.assertEqual(sorted(represented, key=lambda a: a["start"]), expected)
            grouped = {row["skip_inference_reason"]: row for row in result["records"]
                       if row.get("aggregated_target_count", 0) > 1}
            self.assertEqual(set(grouped), {"amount_below_all_six_materiality_floors", "source_amount_unresolved"})
            self.assertEqual([row["aggregated_target_count"] for row in grouped.values()], [2, 2])
            self.assertEqual(grouped["amount_below_all_six_materiality_floors"]["deterministic_signal_status"], "no_signal")
            self.assertEqual(grouped["source_amount_unresolved"]["deterministic_signal_status"], "unknown")
            required = [row for row in result["records"] if row["model_required"]]
            self.assertEqual([row["target_amount"]["value_usd"] for row in required], [20000000, 21000000])
            self.assertEqual(result["target_accounting"]["represented_targets"],
                             len(expected) + sum(row.get("target_amount") is None for row in result["records"]))
            # Retain the original raw capture and digest-named derived path, but
            # change substantive normalized text: this must remain a failure.
            text_path.write_bytes(text.replace("Small values", "Changed values").replace("\n", "\r\n").encode("utf-8"))
            tampered = prepare(root, {"documents": [document]}, issuer_map, references, protocol,
                               external_requests=True, out=root / "tampered-inputs.json")
            self.assertEqual(tampered["records"], [])
            self.assertEqual(tampered["source_failures"], [{"document_id": "legacy", "reason": "ValueError"}])

    def test_unknown_model_arm_is_not_cash(self):
        protocol = json.loads(Path("research/experiments/procurement-protocol.v1.json").read_text())
        entry,inputs = self.fixture()
        inputs.update(protocol_sha256=digest(protocol),records=[entry])
        answer = rules_answers(entry)
        runs = [{"arm":arm,"model":MODELS[arm],"workers":1,"created_at":"2026-09-18T12:00:00+00:00","protocol_sha256":digest(protocol),"records":[{
                 "packet_id":"p","model":MODELS[arm],"status":"completed" if arm=="jev" else "failed",
                 "input_sha256":digest(entry["request"]["state"]),
                 "request_hash":digest(entry["request"] if arm=="jev" else build_chat_request(entry["request"],MODELS[arm])),
                 "answers":answer if arm=="jev" else None,"cost_usd":0}]} for arm in ("nano","mini","jev")]
        review = review_template(inputs,runs)
        for row in review["records"]:
            row.update(status="reviewed",project_identity_verified=True,project_id="P1",source_label=answer,evidence="source span",
                       reviewer="independent fixture",label_origin="independent_source_review")
        with tempfile.TemporaryDirectory() as tmp:
            profile = {"schema_version":"procurement-profile-v1","inputs_sha256":digest(inputs),"protocol_sha256":digest(protocol),
                "frozen_at":"2026-09-18T11:00:00+00:00","arms":{a:{"workers":1,"model":MODELS[a]} for a in ("nano","mini","jev")}}
            profile["sha256"] = digest(profile)
            Path(tmp,"profile.json").write_text(json.dumps(profile))
            result=freeze_signals(Path(tmp),inputs,runs,review,protocol)
        self.assertEqual(result["events"][0]["arms"]["nano"]["status"],"unknown")
        self.assertEqual(result["events"][0]["arms"]["jev"]["status"],"signal")

    def test_prior_bid_and_later_approval_do_not_duplicate_recommendation(self):
        protocol = json.loads(Path("research/experiments/procurement-protocol.v1.json").read_text())
        entry, inputs = self.fixture()
        entries = [{**entry,"packet_id":str(i),"association_date":day,"split":"development"}
                   for i,day in enumerate(("2025-07-02","2025-07-03","2025-07-08"))]
        inputs.update(protocol_sha256=digest(protocol),records=entries)
        runs=[]
        for arm in ("nano","mini","jev"):
            rows=[]
            for e,stage in zip(entries,("bid","recommendation","approved")):
                answers={**rules_answers(e),"stage":stage,"scope":"new_work"}
                rows.append({"packet_id":e["packet_id"],"model":MODELS[arm],"status":"completed", "answers":answers,"cost_usd":0,
                    "input_sha256":digest(e["request"]["state"]),
                    "request_hash":digest(e["request"] if arm=="jev" else build_chat_request(e["request"],MODELS[arm]))})
            runs.append({"arm":arm,"model":MODELS[arm],"protocol_sha256":digest(protocol),"records":rows})
        review=review_template(inputs,runs)
        for row in review["records"]:
            row.update(status="reviewed",project_identity_verified=True,project_id="P1",source_label=rules_answers(entry),
                       reviewer="independent fixture",label_origin="independent_source_review",evidence="Source fixture")
        with tempfile.TemporaryDirectory() as tmp:
            result=freeze_signals(Path(tmp),inputs,runs,review,protocol)
        chosen=[e for e in result["events"] if e["arms"]["jev"]["status"]=="signal"]
        self.assertEqual([e["association_date"] for e in chosen],["2025-07-03"])
        later=[e for e in result["events"] if e["association_date"]=="2025-07-08"]
        self.assertEqual(later[0]["arms"]["jev"]["reason"],"duplicate_later_project_award")


    def test_explicit_award_link_uses_only_earlier_bid_summary_and_keeps_provenance(self):
        prior,_ = self.fixture()
        prior.update(association_date="2025-08-20",source_sha256="raw-bid",minimum_material_amount_usd=16000000)
        prior["target_amount"].update(start=30,end=42)
        current = {**prior,"packet_id":"award","document_id":"award-doc","source_url":"https://example.org/award",
                   "association_date":"2025-08-21","target_amount":None}
        state = {**evidence_state(prior),"amount_candidates":[],"target_amount_id":None,"research_policy":"Use current source stage.",
                 "passages":[{"text":"Contract P1 awarded to Orion Marine Construction on August 21, 2025."}]}
        current["request"] = build_panel(state)
        docs={"d":{},"award-doc":{"related_bid_document_urls":[prior["source_url"]]}}
        derived=linked_bid_targets([prior,current],docs,{"d":200})
        self.assertEqual(len(derived),1)
        self.assertEqual(derived[0]["target_amount"]["source_sha256"],"raw-bid")
        self.assertEqual(derived[0]["association_date"],"2025-08-21")
        self.assertTrue(derived[0]["model_required"])
        self.assertEqual(linked_bid_targets([prior,current],docs,{"d":20}),[])
        prior["association_date"]="2025-08-22"
        self.assertEqual(linked_bid_targets([prior,current],docs,{"d":200}),[])


if __name__ == "__main__": unittest.main()
