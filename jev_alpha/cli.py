"""Small, explicit research commands; network and inference are opt-in actions."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import date
from pathlib import Path

from .audit import dossier_template, evaluate_audit
from .research import census, evidence_packet, select_pilot, source_observations
from .research import SCREENING_QUESTIONS
from .sources import FederalRegister, SourceError
from .store import Store, canonical_json, read_json, utc_now, validate_document_id, write_new_json


def import_discovery(store: Store, path: Path) -> dict:
    data = read_json(path)
    records = data.get("records", data.get("results"))
    if not isinstance(records, list):
        raise ValueError("Discovery input must contain records or results")
    for record in records:
        validate_document_id(record.get("document_number"))
        if not isinstance(record.get("title"), str) or not record.get("publication_date"):
            raise ValueError("Discovery record is missing its title/publication date")
    observation = store.observe(data.get("request_url", path.resolve().as_uri()), path.read_bytes(),
                                kind="imported_discovery", original_captured_at=data.get("captured_at_utc"))
    for record in records:
        store.put_document(record, observation)
    return {"imported_records": len(records), "archive_documents": len(store.documents()),
            "note": "Imported metadata capture time is preserved separately from this local ingestion."}


def generate_dossiers(store: Store, manifest: dict, out: Path) -> dict:
    selected = manifest.get("selected")
    if not isinstance(selected, list) or not selected:
        raise ValueError("Invalid or empty selection manifest")
    pending = []
    for document in selected:
        doc_id = validate_document_id(document.get("document_number"))
        target = out / f"{doc_id}.json"
        if target.exists():
            raise ValueError(f"Dossier already exists: {target}; annotations will not be overwritten")
        dossier = dossier_template(document)
        dossier["source_observations"] = source_observations(store, doc_id)
        dossier["selection_manifest_sha256"] = hashlib.sha256(canonical_json(manifest)).hexdigest()
        pending.append((target, dossier))
    for target, dossier in pending:
        write_new_json(target, dossier)
    return {"dossiers_created": len(pending), "directory": str(out), "status": "unreviewed"}


def _daily_inputs(protocol: dict, signals: dict, *, complete: bool = True) -> None:
    """Verify the frozen document/date/arm cohort before reading outcomes."""
    from .experiment import digest
    if signals.get("protocol_sha256") != digest(protocol):
        raise ValueError("Signals do not match the frozen daily protocol")
    if signals.get("prediction_manifest_sha256") != protocol.get("prediction_manifest_sha256"):
        raise ValueError("Signals do not match the registered prediction manifest")
    if complete and signals.get("status") not in ("complete", "complete_with_failures"):
        raise ValueError("Fully attempted frozen daily predictions are required")
    if protocol.get("symbols") != ["NUE", "STLD"] or protocol.get("benchmark") != "SPY":
        raise ValueError("Daily evaluator supports the registered NUE/STLD and SPY cohort")
    selected, rows = protocol.get("selected"), signals.get("signals")
    if not isinstance(selected, list) or not selected or not isinstance(rows, list):
        raise ValueError("Registered selection and frozen signal rows are required")
    expected = {}
    for row in selected:
        document, published = row["document_id"], row["publication_date"]
        if not isinstance(document, str) or not document or document in expected:
            raise ValueError("Registered daily cohort contains invalid or duplicate IDs")
        if date.fromisoformat(published).isoformat() != published:
            raise ValueError("Publication dates must use YYYY-MM-DD")
        expected[document] = published
    ids = protocol.get("document_ids")
    if not isinstance(ids, list) or len(ids) != len(expected) or set(ids) != set(expected):
        raise ValueError("Registered daily document IDs and selection differ")
    registered_arms = protocol.get("arms")
    if registered_arms != ["jev", "baseline", "always_long", "cash"]:
        raise ValueError("Unexpected daily diagnostic arms")
    actual = {}
    for row in rows:
        document = row["document_id"]
        if document in actual or document not in expected:
            raise ValueError("Frozen signal cohort has an unexpected or duplicate document")
        actual[document] = row["publication_date"]
        if actual[document] != expected[document]:
            raise ValueError("Frozen signal publication date changed")
        arms = row.get("arms")
        if not isinstance(arms, dict) or set(arms) != set(registered_arms):
            raise ValueError("Frozen signal arm cohort changed")
        if complete and any(arm.get("status") not in ("completed", "failed") for arm in arms.values()):
            raise ValueError("Frozen signals contain an unattempted arm")
    if actual != expected:
        raise ValueError("Frozen signal cohort does not match every registered document")


def evaluate_daily_capture(store: Store, protocol: dict, signals: dict, directory: Path) -> dict:
    """Read a complete immutable capture and evaluate it entirely offline."""
    from .daily_returns import evaluate_daily, parse_yahoo_chart
    from .experiment import digest
    _daily_inputs(protocol, signals)
    capture = read_json(directory / "capture.json")
    if capture.get("status") != "complete":
        raise ValueError("Daily price capture must be complete before evaluation")
    if capture.get("protocol_sha256") != digest(protocol) or capture.get("signals_sha256") != digest(signals):
        raise ValueError("Daily capture protocol or frozen signal hash mismatch")
    if capture.get("signals_frozen_at") != signals.get("frozen_at"):
        raise ValueError("Daily capture frozen-signal timestamp differs")
    records = capture.get("symbols")
    if (not isinstance(records, list) or len(records) != 3
            or {record.get("symbol") for record in records} != {"NUE", "STLD", "SPY"}
            or any(record.get("status") != "captured" for record in records)):
        raise ValueError("Capture must contain exactly three successfully captured symbols")
    prices, provenance = {}, []
    for record in records:
        symbol, raw_hash = record["symbol"], record.get("raw_sha256")
        if not isinstance(raw_hash, str) or len(raw_hash) != 64 or any(c not in "0123456789abcdef" for c in raw_hash):
            raise ValueError("Capture source digest must be a lowercase SHA256")
        blob_path = store.blobs / raw_hash
        raw = blob_path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != raw_hash:
            raise ValueError("Archived daily source bytes do not match their digest")
        original = read_json(blob_path)
        payload = read_json(directory / f"{symbol}.json")
        if digest(original) != digest(payload):
            raise ValueError("Daily price JSON differs from its archived source")
        prices[symbol] = parse_yahoo_chart(payload, symbol)
        provenance.append({**record, "canonical_payload_sha256": digest(payload),
                           "price_file": str((directory / f"{symbol}.json").resolve())})
    report = evaluate_daily(protocol, signals["signals"], prices)
    report.update({"created_at": utc_now(), "protocol_sha256": digest(protocol),
                   "signals_sha256": digest(signals), "capture_sha256": digest(capture),
                   "protocol": protocol, "experiment_id": protocol.get("experiment_id"),
                   "source_provenance": provenance, "capture_created_at": capture.get("created_at"),
                   "signals_frozen_at": signals["frozen_at"], "offline_evaluation": True,
                   "interpretation": "Historical development association proxy; not an as-of strategy backtest, executable return, or proven alpha.",
                   "alpha_proven": False, "trade_eligible": False})
    return report


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Evidence-first Jev alpha research. No trading orders.")
    root.add_argument("--data-dir", type=Path, default=Path("data"), help="Local ignored archive directory")
    commands = root.add_subparsers(dest="command", required=True)
    commands.add_parser("init", help="Initialize local archive")
    imp = commands.add_parser("import-discovery", help="Import recorded discovery metadata offline")
    imp.add_argument("path", type=Path)
    pi = commands.add_parser("import-pi", help="Import recorded public-inspection metadata offline")
    pi.add_argument("path", type=Path)
    discover = commands.add_parser("discover", help="Fetch public Federal Register metadata")
    discover.add_argument("--term", default='"corrosion-resistant steel"')
    discover.add_argument("--start", required=True)
    discover.add_argument("--end", required=True)
    commands.add_parser("census", help="Report document counts, never assumed trade counts")
    select = commands.add_parser("select", help="Freeze a price-blind rejection-test sample")
    select.add_argument("--count", type=int, default=20)
    select.add_argument("--seed", default="core-pilot-v1")
    select.add_argument("--out", type=Path, required=True)
    collect = commands.add_parser("collect", help="Archive detail, text and public-inspection records")
    collect.add_argument("document_ids", nargs="*")
    collect.add_argument("--selection", type=Path, help="Collect every document in one frozen pilot manifest")
    dossiers = commands.add_parser("dossiers", help="Create unreviewed evidence worksheets")
    dossiers.add_argument("--selection", type=Path, required=True)
    dossiers.add_argument("--out", type=Path, required=True)
    audit = commands.add_parser("audit", help="Evaluate reviewed dossiers before buying prices")
    audit.add_argument("--dossiers", type=Path, required=True)
    audit.add_argument("--selection", type=Path, required=True, help="Frozen manifest guards against quietly dropping selected cases")
    audit.add_argument("--out", type=Path)
    packet = commands.add_parser("packet", help="Assemble source evidence without model calls")
    packet.add_argument("document_id")
    packet.add_argument("--prior", action="append", default=[])
    packet.add_argument("--passages", type=Path, help="JSON array of explicitly selected passage IDs")
    packet.add_argument("--out", type=Path, required=True)
    prepare = commands.add_parser("jev-prepare", help="Build and estimate request; no network or credentials")
    prepare.add_argument("--state", type=Path, required=True)
    prepare.add_argument("--pack", type=Path, default=Path("research/jev-core-question-pack.v1.json"))
    prepare.add_argument("--panel", choices=["screening", "full"], default="screening")
    prepare.add_argument("--out", type=Path, required=True)
    run = commands.add_parser("jev-run", help="Explicitly submit one paid request using local environment key")
    run.add_argument("--request", type=Path, required=True)
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--max-cost-usd", type=float, default=0.01)
    run.add_argument("--phase-budget-usd", type=float, default=20)
    compare = commands.add_parser("compare-prepare", help="Freeze identical evidence for Jev, rules and a conventional model")
    compare.add_argument("--selection", type=Path, default=Path("research/experiments/core-pilot-v2.json"))
    compare.add_argument("--labels", type=Path, default=Path("research/benchmarks/core-labels.v1.json"))
    compare.add_argument("--pack", type=Path, default=Path("research/jev-core-question-pack.v1.json"))
    compare.add_argument("--out", type=Path, required=True)
    comparison = commands.add_parser("compare-run", help="Run offline rules, or explicit paid model comparison with --live")
    comparison.add_argument("--manifest", type=Path, required=True)
    comparison.add_argument("--out", type=Path, required=True)
    comparison.add_argument("--live", action="store_true")
    comparison.add_argument("--budget-usd", type=float, default=1)
    comparison.add_argument("--transport", choices=["urllib", "curl"], default="urllib")
    fanout = commands.add_parser("fanout-prepare", help="Prepare lossless parallel passage judgments; no paid calls")
    fanout.add_argument("--state", type=Path, required=True)
    fanout.add_argument("--out", type=Path, required=True)
    fanrun = commands.add_parser("fanout-run", help="Run prepared passage judgments with --live; dry-run otherwise")
    fanrun.add_argument("--plan", type=Path, required=True)
    fanrun.add_argument("--out", type=Path, required=True)
    fanrun.add_argument("--live", action="store_true")
    fanrun.add_argument("--budget-usd", type=float, default=1)
    fanrun.add_argument("--transport", choices=["urllib", "curl"], default="urllib")
    panelcheck = commands.add_parser("panel-check", help="Check interpretation claims against supplied evidence prerequisites")
    panelcheck.add_argument("--request", type=Path, required=True)
    panelcheck.add_argument("--response", type=Path, required=True)
    panelcheck.add_argument("--out", type=Path, required=True)
    transition = commands.add_parser("transition-prepare", help="Freeze a source-bound preliminary/operative transition test")
    transition.add_argument("--spec", type=Path, required=True)
    transition.add_argument("--pack", type=Path, default=Path("research/benchmarks/transition-questions.v2.json"))
    transition.add_argument("--packets", type=Path, default=Path("data/evidence-v2"))
    transition.add_argument("--out", type=Path, required=True)
    transrun = commands.add_parser("transition-run", help="Evaluate a frozen transition; paid calls only with --live")
    transrun.add_argument("--manifest", type=Path, required=True)
    transrun.add_argument("--out", type=Path, required=True)
    transrun.add_argument("--live", action="store_true")
    transrun.add_argument("--with-baseline", action="store_true")
    transrun.add_argument("--budget-usd", type=float, default=1)
    daily_predict = commands.add_parser("daily-predict", help="Freeze daily proxy signals; paid inference only with --live")
    daily_predict.add_argument("--manifest", type=Path, required=True)
    daily_predict.add_argument("--protocol", type=Path, required=True)
    daily_predict.add_argument("--out", type=Path, required=True, help="New prediction output directory")
    daily_predict.add_argument("--live", action="store_true")
    daily_predict.add_argument("--resume-signals", type=Path, help="Frozen prior prediction result for explicit resumption")
    daily_prices = commands.add_parser("daily-prices", help="Prepare daily-price collection; public network requests only with --live")
    daily_prices.add_argument("--protocol", type=Path, required=True)
    daily_prices.add_argument("--signals", type=Path, required=True)
    daily_prices.add_argument("--out", type=Path, required=True, help="New capture or dry-plan directory")
    daily_prices.add_argument("--live", action="store_true")
    daily_evaluate = commands.add_parser("daily-evaluate", help="Verify frozen provenance and evaluate daily associations offline")
    daily_evaluate.add_argument("--protocol", type=Path, required=True)
    daily_evaluate.add_argument("--signals", type=Path, required=True)
    daily_evaluate.add_argument("--prices", type=Path, required=True, help="Complete capture directory")
    daily_evaluate.add_argument("--out", type=Path, required=True, help="New report JSON; never overwrites")
    claims_prepare = commands.add_parser("claims-prepare", help="Freeze all-passage issuer-claim questions without inference")
    claims_prepare.add_argument("--packet", type=Path, required=True)
    claims_prepare.add_argument("--out", type=Path, required=True)
    claims_run = commands.add_parser("claims-run", help="Screen a frozen full-passage plan; paid calls only with --live")
    claims_run.add_argument("--plan", type=Path, required=True)
    claims_run.add_argument("--out", type=Path, required=True)
    claims_run.add_argument("--live", action="store_true")
    claims_run.add_argument("--arm", choices=["jev", "baseline"], default="jev")
    claims_run.add_argument("--workers", type=int, default=4)
    claims_run.add_argument("--budget-usd", type=float, default=5)
    for name, help_text in (
        ("operating-collect", "Capture the bounded product-mention cohort; network only with --live"),
        ("operating-freeze", "Freeze source hashes and thread-grouped development/evaluation splits"),
        ("operating-prepare", "Prepare the frozen twelve-dimension operating screen"),
        ("operating-semantic", "Run the semantic retrieval control; paid calls only with --live"),
    ):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--protocol", type=Path, default=Path("research/experiments/operating-changes-protocol.v1.json"))
        command.add_argument("--out", type=Path, required=True)
        if name == "operating-collect":
            command.add_argument("--live", action="store_true")
        elif name == "operating-freeze":
            command.add_argument("--capture", type=Path, required=True)
        else:
            command.add_argument("--cohort", type=Path, required=True)
            command.add_argument("--split", choices=["development", "evaluation"], default="development")
            if name == "operating-semantic":
                command.add_argument("--live", action="store_true")
                command.add_argument("--model", help="Pin the model observed in the development control")
    operating_run = commands.add_parser("operating-run", help="Run Jev or compact chat against the frozen evidence")
    operating_run.add_argument("--plan", type=Path, required=True)
    operating_run.add_argument("--out", type=Path, required=True)
    operating_run.add_argument("--arm", choices=["jev", "baseline"], default="jev")
    operating_run.add_argument("--live", action="store_true")
    operating_run.add_argument("--workers", type=int, default=4)
    operating_run.add_argument("--budget-usd", type=float, default=10)
    operating_run.add_argument("--resume-report", type=Path,
                               help="Continue only previously unattempted chunks; reuse cached successes and preserve failures")
    operating_compare = commands.add_parser("operating-compare", help="Join frozen arm runs and retain missing-control coverage")
    operating_compare.add_argument("--cohort", type=Path, required=True)
    operating_compare.add_argument("--jev", type=Path, action="append", default=[])
    operating_compare.add_argument("--baseline", type=Path, action="append", default=[])
    operating_compare.add_argument("--semantic", type=Path, action="append", default=[])
    operating_compare.add_argument("--out", type=Path, required=True)
    operating_threads = commands.add_parser("operating-threads", help="Resolve HN roots using bounded original-API reads")
    operating_threads.add_argument("--capture", type=Path, required=True)
    operating_threads.add_argument("--out", type=Path, required=True)
    operating_threads.add_argument("--live", action="store_true")
    operating_threads.add_argument("--max-requests", type=int, default=500)
    for name in ("operating-review-sample", "operating-evaluate"):
        command = commands.add_parser(name, help="Independent evidence review; no network or model calls")
        command.add_argument("--cohort", type=Path, required=True)
        command.add_argument("--arms", type=Path, required=True, help="Frozen arm selections/status manifest")
        command.add_argument("--out", type=Path, required=True)
        if name == "operating-evaluate":
            command.add_argument("--review", type=Path, required=True)
            command.add_argument("--protocol", type=Path, default=Path("research/experiments/operating-changes-protocol.v1.json"))
    for name in ("alpaca-smoke", "alpaca-collect"):
        command = commands.add_parser(name, help="Read-only historical SIP quotes; credentials required locally")
        command.add_argument("--query", type=Path, required=True)
        command.add_argument("--out", type=Path, required=True)
        command.add_argument("--live", action="store_true")
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        with Store(args.data_dir) as store:
            if args.command == "init":
                result = {"archive": str(store.root.resolve()), "status": "ready"}
            elif args.command == "import-discovery":
                result = import_discovery(store, args.path)
            elif args.command == "import-pi":
                wrapper = read_json(args.path)
                payload = wrapper.get("response", wrapper)
                doc_id = validate_document_id(payload.get("document_number"))
                store.observe(wrapper.get("request_url", args.path.resolve().as_uri()), canonical_json(payload),
                              kind="public_inspection", document_id=doc_id, original_captured_at=wrapper.get("captured_at_utc"))
                result = {"imported_public_inspection": doc_id, "earliest_disclosure_verified": False}
            elif args.command == "discover":
                result = FederalRegister(store).discover(args.term, args.start, args.end)
            elif args.command == "census":
                result = census(store.documents())
            elif args.command == "select":
                result = select_pilot(store.documents(), args.count, args.seed)
                write_new_json(args.out, result)
                result = {"selected_count": result["selected_count"], "manifest": str(args.out), "price_data_used": False}
            elif args.command == "collect":
                client = FederalRegister(store)
                ids = args.document_ids
                if args.selection:
                    if ids:
                        raise ValueError("Supply document IDs or a selection, not both")
                    ids = [d["document_number"] for d in read_json(args.selection)["selected"]]
                if not ids or len(ids) > 50:
                    raise ValueError("Collect between 1 and 50 explicitly selected documents")
                outcomes = []
                for doc_id in ids:
                    try:
                        outcomes.append(client.collect(doc_id))
                    except SourceError as exc:
                        outcomes.append({"document_id": doc_id, "error": str(exc)})
                result = {"documents": outcomes, "failed_documents": sum("error" in d for d in outcomes)}
            elif args.command == "dossiers":
                result = generate_dossiers(store, read_json(args.selection), args.out)
            elif args.command == "audit":
                dossiers = [read_json(path) for path in sorted(args.dossiers.glob("*.json"))]
                manifest = read_json(args.selection)
                expected = {d["document_number"] for d in manifest["selected"]}
                actual = {d.get("document_id") for d in dossiers}
                if expected != actual:
                    raise ValueError("Dossier IDs do not match the frozen selection; preserve exclusions and unresolved cases")
                manifest_hash = hashlib.sha256(canonical_json(manifest)).hexdigest()
                if any(d.get("selection_manifest_sha256") != manifest_hash for d in dossiers):
                    raise ValueError("Dossier selection hash mismatch")
                result = evaluate_audit(dossiers)
                if args.out:
                    write_new_json(args.out, result)
                    result = {key: result[key] for key in ("readiness", "alpha_proven", "counts", "interpretation")}
                    result["report"] = str(args.out)
            elif args.command == "packet":
                result = evidence_packet(store, args.document_id, prior_ids=args.prior,
                                         passage_ids=read_json(args.passages) if args.passages else None)
                write_new_json(args.out, result)
                result = {"packet": str(args.out), "passages": len(result["current_source_passages_with_ids"]), "review_status": "unreviewed"}
            elif args.command == "jev-prepare":
                from .jev import build_request, estimate_request
                request = build_request(read_json(args.state), read_json(args.pack),
                                        SCREENING_QUESTIONS if args.panel == "screening" else None)
                estimate = estimate_request(request)
                write_new_json(args.out, request)
                result = {"request": str(args.out), "panel": args.panel, "questions": len(request["questions"]), "estimate": estimate, "submitted": False}
            elif args.command == "jev-run":
                from .jev import JevClient, validate_response, estimate_request, JevRequestError
                from .credentials import openrouter_key
                request = read_json(args.request)
                if args.out.exists():
                    raise ValueError("Output already exists; choose a new path")
                cached = store.cached_response(request)
                if cached is not None:
                    response = validate_response(cached, request)
                    was_cached = True
                else:
                    estimate = estimate_request(request)
                    key = openrouter_key()
                    client = JevClient(api_key=key, max_estimated_cost_usd=args.max_cost_usd)
                    if not key:
                        raise JevRequestError("Set OPENROUTER_API_KEY locally or in ignored .env before live submission; no request sent")
                    if estimate["conservative_input_tokens"] > client.max_input_tokens or estimate["estimated_cost_usd"] > client.max_estimated_cost_usd:
                        raise ValueError("Request exceeds the configured conservative input/cost limit; no request sent")
                    attempt = store.reserve_attempt(request, estimate["estimated_cost_usd"], args.phase_budget_usd)
                    try:
                        response = client.submit(request)
                        store.cache_response(request, response)
                    except BaseException:
                        store.finish_attempt(attempt, None)
                        raise
                    store.finish_attempt(attempt, response)
                    was_cached = False
                write_new_json(args.out, response)
                result = {"response": str(args.out), "cached": was_cached, "model": response.get("model"), "usage": response.get("usage"), "alpha_proven": False}
            elif args.command == "compare-prepare":
                from .experiment import prepare_comparison
                result = prepare_comparison(store, read_json(args.selection), read_json(args.pack), read_json(args.labels), args.out)
            elif args.command == "compare-run":
                from .experiment import run_comparison
                result = run_comparison(store, args.manifest, args.out, live=args.live, phase_budget_usd=args.budget_usd, transport=args.transport)
                result["metrics_summary"] = [{key: group.get(key) for key in ("model", "label_count", "correct", "coverage", "label_origin")} for group in result.pop("metrics")["groups"]]
            elif args.command == "fanout-prepare":
                from .fanout import prepare_fanout
                plan = prepare_fanout(read_json(args.state))
                write_new_json(args.out, plan)
                result = {"plan": str(args.out), "chunks": len(plan["chunks"]), "passages": len(plan["original_passage_ids"]), "estimated_cost_usd": plan["estimated_total_cost_usd"], "submitted": False}
            elif args.command == "fanout-run":
                from .experiment import run_fanout
                result = run_fanout(store, read_json(args.plan), args.out, live=args.live, phase_budget_usd=args.budget_usd, transport=args.transport)
            elif args.command == "panel-check":
                from .domain import guard_panel
                raw = read_json(args.response)
                result = guard_panel(read_json(args.request), raw.get("response", raw))
                write_new_json(args.out, result)
                result = {"report": str(args.out), "status": result["status"], "trade_eligible": result["trade_eligible"], "blockers": result["blockers"], "contradictions": result["contradictions"]}
            elif args.command == "transition-prepare":
                from .transitions import prepare_transition
                spec = read_json(args.spec)
                ids = [validate_document_id(d["document_id"]) for d in spec["documents"]]
                packets = {doc: read_json(args.packets / f"{doc}.json") for doc in ids}
                manifest = prepare_transition(spec, packets, read_json(args.pack))
                write_new_json(args.out, manifest)
                result = {"manifest": str(args.out), "episode_id": manifest["episode_id"],
                          "questions": len(manifest["request"]["questions"]), "estimate": manifest["estimate"], "submitted": False}
            elif args.command == "transition-run":
                from .transitions import run_transition
                report = run_transition(store, read_json(args.manifest), args.out, live=args.live,
                                        baseline=args.with_baseline, budget_usd=args.budget_usd)
                result = {"report": str(args.out), "status": report["status"], "trade_eligible": False,
                          "arms": [{key: row.get(key) for key in ("arm", "status", "matched", "total")} for row in report["runs"]]}
            elif args.command == "daily-predict":
                from .daily_signals import predict_daily
                resume = {"resume": read_json(args.resume_signals)} if args.resume_signals else {}
                report = predict_daily(store, read_json(args.manifest), read_json(args.protocol), args.out,
                                       live=args.live, **resume)
                result = {"signals": str(args.out / "signals.json"), "status": report["status"],
                          "documents": len(report["signals"]), "live": args.live, "prices_opened": False}
            elif args.command == "daily-prices":
                from .daily_prices import collect_daily_prices
                from .experiment import digest
                protocol, signals = read_json(args.protocol), read_json(args.signals)
                if args.out.exists():
                    raise ValueError("Output exists; choose a new daily capture or dry-plan directory")
                _daily_inputs(protocol, signals, complete=args.live)
                if args.live:
                    report = collect_daily_prices(store, protocol, signals, args.out)
                    result = {"capture": str(args.out / "capture.json"), "status": report["status"],
                              "symbols": report["symbols"], "live": True}
                else:
                    report = {"schema_version": "daily-price-plan-v1", "status": "dry_run", "created_at": utc_now(),
                              "protocol_sha256": digest(protocol), "signals_sha256": digest(signals),
                              "network_requests": 0, "symbols": ["NUE", "STLD", "SPY"],
                              "requires_complete_frozen_signals": True,
                              "interpretation": "No prices requested; use --live and a new output directory to collect."}
                    write_new_json(args.out / "plan.json", report)
                    result = {"plan": str(args.out / "plan.json"), "status": "dry_run", "network_requests": 0}
            elif args.command == "daily-evaluate":
                if args.out.exists():
                    raise ValueError("Report exists; choose a new path to preserve previous results")
                report = evaluate_daily_capture(store, read_json(args.protocol), read_json(args.signals), args.prices)
                write_new_json(args.out, report)
                result = {"report": str(args.out), "document_count": report["document_count"],
                          "priced_document_count": report["priced_document_count"], "cluster_count": report["cluster_count"],
                          "arms": report["arms"], "interpretation": report["interpretation"], "alpha_proven": False}
            elif args.command == "claims-prepare":
                from .claims import prepare_claim_screen
                plan = prepare_claim_screen(read_json(args.packet))
                write_new_json(args.out, plan)
                result = {"plan": str(args.out), "chunks": len(plan["chunks"]),
                          "passages": len(plan["original_passage_ids"]),
                          "estimated_cost_usd": plan["estimated_total_cost_usd"], "submitted": False}
            elif args.command == "claims-run":
                from .claim_experiment import run_claim_screen
                result = run_claim_screen(store.root, read_json(args.plan), args.out, live=args.live,
                                          arm=args.arm, workers=args.workers, budget_usd=args.budget_usd)
                result["report"] = str(args.out / "report.json")
            elif args.command == "operating-collect":
                from .operating_sources import collect_operating_sources
                report = collect_operating_sources(store, read_json(args.protocol), args.out, live=args.live)
                result = {"capture": str(args.out), "status": report["status"],
                          "records": len(report["records"]), "spending": report.get("billing")}
            elif args.command == "operating-freeze":
                from .operating_audit import freeze_operating_cohort
                report = freeze_operating_cohort(read_json(args.protocol), read_json(args.capture))
                write_new_json(args.out, report)
                result = {"cohort": str(args.out), "records": len(report["records"]),
                          "development": report["development_count"], "evaluation": report["evaluation_count"],
                          "unresolved_threads": len(report["unresolved_thread_record_ids"])}
            elif args.command == "operating-prepare":
                from .operating import prepare_operating_screen
                report = prepare_operating_screen(read_json(args.protocol), read_json(args.cohort), split=args.split)
                write_new_json(args.out, report)
                result = {"plan": str(args.out), "records": len(report["records"]), "chunks": len(report["chunks"]),
                          "oversized_targets": len(report["oversized_targets"]), "submitted": False}
            elif args.command == "operating-run":
                from .operating import run_operating_screen
                result = run_operating_screen(store.root / "models", read_json(args.plan), args.out,
                                              live=args.live, arm=args.arm, workers=args.workers, budget_usd=args.budget_usd,
                                              resume_report=read_json(args.resume_report) if args.resume_report else None)
            elif args.command == "operating-semantic":
                from .operating_semantic import run_semantic_control
                result = run_semantic_control(store, read_json(args.protocol), read_json(args.cohort), args.out,
                                              split=args.split, live=args.live, model=args.model)
            elif args.command == "operating-compare":
                from .operating_workflow import comparison_manifest
                report = comparison_manifest(read_json(args.cohort),
                    {"jev": args.jev, "baseline": args.baseline, "semantic_retrieval": args.semantic})
                write_new_json(args.out, report)
                result = {"manifest": str(args.out), "selected_pairs": {k: len(v) for k, v in report["selected_pairs"].items()},
                          "statuses": {k: v["status"] for k, v in report["arm_status"].items()}, "alpha_proven": False}
            elif args.command == "operating-threads":
                from .operating_threads import resolve_operating_threads
                report = resolve_operating_threads(store, read_json(args.capture), args.out, live=args.live,
                                                   max_requests=args.max_requests)
                result = {"capture": str(args.out), "resolution": report["thread_resolution_run"]}
            elif args.command in {"operating-review-sample", "operating-evaluate"}:
                from .operating_audit import evaluate_operating_reviews, review_sample
                from .experiment import digest
                cohort, arms = read_json(args.cohort), read_json(args.arms)
                if arms.get("cohort_sha256") != digest(cohort):
                    raise ValueError("Arm manifest does not match the frozen cohort")
                selected = {name: {(pair["record_id"], pair["vendor_id"]) for pair in rows}
                            for name, rows in arms["selected_pairs"].items()}
                if args.command == "operating-review-sample":
                    retained = {rid for pairs in selected.values() for rid, _vendor in pairs}
                    unknown = set()
                    for arm in arms.get("arm_status", {}).values():
                        unknown.update(pair["record_id"] for pair in arm.get("unknown_pairs", []))
                    report = review_sample(cohort, retained, unknown_ids=unknown)
                else:
                    report = evaluate_operating_reviews(read_json(args.protocol), cohort, read_json(args.review),
                        selected, costs=arms.get("costs"), arm_status=arms.get("arm_status"))
                write_new_json(args.out, report)
                result = {"report": str(args.out), "decision": report.get("decision"),
                          "source_feasibility_decision": report.get("source_feasibility_decision"),
                          "qualifying_episodes": report.get("qualifying_episodes"),
                          "review_records": len(report.get("review_record_ids", [])), "alpha_proven": False}
            elif args.command in {"alpaca-smoke", "alpaca-collect"}:
                from .alpaca_prices import alpaca_entitlement_smoke, collect_alpaca_prices
                if args.live:
                    call = alpaca_entitlement_smoke if args.command == "alpaca-smoke" else collect_alpaca_prices
                    report = call(store, read_json(args.query), args.out)
                    result = {"receipt": str(args.out / "receipt.json"), "status": report.get("status"),
                              "entitlement_status": report.get("entitlement_status")}
                else:
                    result = {"status": "dry_run", "query": read_json(args.query), "network_requests": 0}
                    write_new_json(args.out / "plan.json", result)
            print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
        return 0
    except (ValueError, OSError, SourceError, KeyError, TypeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        # Provider exceptions deliberately contain no request headers or key.
        from .jev import JevRequestError, JevValidationError
        if isinstance(exc, (JevRequestError, JevValidationError)):
            print(f"Error: {exc}", file=sys.stderr)
            return 2
        raise


if __name__ == "__main__":
    raise SystemExit(main())
