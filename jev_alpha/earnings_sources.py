"""Bounded SEC earnings-release intake for a disclosed development cohort.

Current-list sampling is survivorship biased. These files establish as-filed
inputs and SEC availability, not a representative small-cap universe or alpha.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import time
from urllib.parse import urlparse

from .filing_text import filing_packet

SEED = "jev-earnings-pilot-v1"
START, END = "2026-04-01", "2026-06-30"


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


class Capture:
    def __init__(self, root: Path, maximum: int = 120, allowed_hosts: set[str] | None = None):
        self.root = root
        self.maximum = maximum
        self.log = root / "requests.jsonl"
        self.rows = [json.loads(x) for x in self.log.read_text().splitlines()] if self.log.exists() else []
        self.last = 0.0
        self.allowed_hosts = allowed_hosts or {"www.sec.gov", "data.sec.gov"}
        self.executable = shutil.which("curl.exe") or shutil.which("curl")
        if not self.executable:
            raise RuntimeError("curl is required")

    def get(self, url: str, name: str) -> bytes:
        parsed = urlparse(url)
        if parsed.scheme != "https" or parsed.hostname not in self.allowed_hosts or parsed.username or parsed.password:
            raise ValueError("Only explicitly allowed ordinary public HTTPS GET requests are allowed")
        for row in self.rows:
            if row["url"] == url:
                if row["http_status"] != 200 or row["returncode"] != 0:
                    raise RuntimeError("Previously failed URL is not retried")
                raw = (self.root / row["body_path"]).read_bytes()
                if sha(raw) != row["body_sha256"]:
                    raise ValueError("Cached source bytes changed")
                return raw
        if len(self.rows) >= self.maximum:
            raise RuntimeError("SEC request cap reached")
        time.sleep(max(0.0, 0.3 - (time.monotonic() - self.last)))
        body = self.root / "raw" / f"{name}.body"
        headers = self.root / "raw" / f"{name}.headers"
        body.parent.mkdir(parents=True, exist_ok=True)
        if body.exists() or headers.exists():
            raise ValueError("Archive output already exists without a logged request")
        self.last = time.monotonic()
        result = subprocess.run([self.executable, "-q", "--silent", "--show-error", "--proto", "=https",
            "--max-redirs", "0", "--retry", "0", "--connect-timeout", "10", "--max-time", "40",
            "--max-filesize", "25000000", "--user-agent", "jev-alpha-research/0.1 (public earnings research)",
            "--dump-header", str(headers), "--output", str(body), "--write-out", "%{http_code}", url],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=45, shell=False,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        raw = body.read_bytes() if body.exists() else b""
        if not body.exists():
            # An empty capture records a transport failure with no body bytes.
            body.write_bytes(b"")
        status = int(result.stdout) if result.stdout.strip().isdigit() else 0
        row = {"url": url, "captured_at": datetime.now(timezone.utc).isoformat(),
            "http_status": status, "returncode": result.returncode,
            "body_path": body.relative_to(self.root).as_posix(), "body_sha256": sha(raw),
            "bytes": len(raw), "headers_path": headers.relative_to(self.root).as_posix(),
            "headers_sha256": sha(headers.read_bytes()) if headers.exists() else None}
        self.rows.append(row)
        with self.log.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row) + "\n")
        if status != 200 or result.returncode:
            raise RuntimeError(f"Public source returned status {status}, transport {result.returncode}; archived without retry")
        return raw


def filing_rows(payload: dict) -> list[dict]:
    recent = payload["filings"]["recent"]
    keys = ("accessionNumber", "filingDate", "reportDate", "acceptanceDateTime", "form", "items", "primaryDocument")
    return [{k: recent[k][i] for k in keys} for i in range(len(recent["accessionNumber"]))]


def freeze_cohort(root: Path, capture: Capture, master_index: Path | None = None) -> dict:
    if master_index is None:
        index_url = "https://www.sec.gov/files/company_tickers_exchange.json"
        raw = capture.get(index_url, "current-ticker-index")
        index = json.loads(raw)
        rows = [dict(zip(index["fields"], row)) for row in index["data"]]
        eligible = [row for row in rows if row.get("exchange") in {"NYSE", "Nasdaq"}
                    and re.fullmatch(r"[A-Z]{1,5}", row.get("ticker", ""))]
    else:
        raw = master_index.read_bytes()
        index_url = "https://www.sec.gov/Archives/edgar/full-index/2026/QTR2/master.zip"
        if sha(raw) != "37c90ce7515c75a39b843b79c891ef73ebbeda6be850772efba5ec501587eca1":
            raise ValueError("Expected the previously verified complete Q2 index snapshot")
        save(root / "sampling-amendment.v1.json", {"reason": "Ticker index returned 403; do not retry blocked URL. Use existing complete official Q2 filing index.",
            "frozen_before_issuer_checks": True, "raw_master_sha256": sha(raw),
            "frame": "All unique issuers with original 8-K in Q2 2026; current exchange and ordinary-looking symbol checked from submissions", "seed": SEED})
        rows = []
        for line in raw.decode("utf-8").splitlines():
            fields = line.split("|")
            if len(fields) == 5 and fields[0].isdigit():
                rows.append(dict(zip(("cik", "name", "form", "filed", "path"), fields)))
        eligible = [{"cik": int(row["cik"]), "name": row["name"], "ticker": ""} for row in rows
                    if row["form"] == "8-K" and START <= row["filed"] <= END]
    by_cik = {}
    for row in sorted(eligible, key=lambda x: (int(x["cik"]), x["ticker"])):
        by_cik.setdefault(int(row["cik"]), row)
    ordered = sorted(by_cik.values(), key=lambda x: sha(f"{SEED}|{int(x['cik'])}".encode()))
    save(root / "sampling-frame.v1.json", {"index_url": index_url, "index_sha256": sha(raw),
        "seed": SEED, "cik_format_for_seed": "integer_without_zero_padding", "raw_index_count": len(rows),
        "eligible_issuer_count": len(ordered), "ordered_issuers": ordered,
        "current_survivorship_bias": True, "size_and_ordinary_share_status_verified": False})
    selected, checks = [], []
    for candidate in ordered[:32]:
        cik = int(candidate["cik"])
        check = {**candidate, "rank": len(checks) + 1}
        try:
            payload = json.loads(capture.get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json", f"submissions-{cik}"))
            if master_index is not None:
                listing = sorted((ticker, exchange) for ticker, exchange in zip(payload.get("tickers", []), payload.get("exchanges", []))
                    if exchange in {"NYSE", "Nasdaq"} and re.fullmatch(r"[A-Z]{1,5}", ticker))
                if not listing:
                    check["result"] = "no_current_eligible_exchange_symbol"
                    checks.append(check)
                    print(json.dumps(check), flush=True)
                    continue
                candidate = {**candidate, "ticker": listing[0][0], "exchange": listing[0][1], "all_current_listings": listing}
                check.update(ticker=candidate["ticker"], exchange=candidate["exchange"])
            original = [r for r in filing_rows(payload) if r["form"] == "8-K" and "2.02" in [x.strip() for x in r["items"].split(",")]]
            original.sort(key=lambda r: (r["acceptanceDateTime"], r["accessionNumber"]))
            in_window = [r for r in original if START <= r["filingDate"] <= END]
            if not in_window:
                check["result"] = "no_original_item_2_02_in_window_in_recent_array"
            else:
                current = in_window[-1]
                prior = [r for r in original if r["acceptanceDateTime"] < current["acceptanceDateTime"]
                         and r["reportDate"] != current["reportDate"]]
                record = {**candidate, "event_id": f"{candidate['ticker'].lower()}-{current['filingDate']}",
                    "current": current, "prior": prior[-1] if prior else None}
                selected.append(record)
                check.update(result="selected", event_id=record["event_id"], prior_present=bool(prior))
        except (ValueError, KeyError, RuntimeError, OSError, subprocess.SubprocessError) as exc:
            check.update(result="source_failed", error=str(exc))
        checks.append(check)
        print(json.dumps(check), flush=True)
        if len(selected) == 12:
            break
    cohort = {"schema_version": "earnings-development-cohort-v1", "frozen_at": datetime.now(timezone.utc).isoformat(),
        "seed": SEED, "window": [START, END], "selection": "latest original Item 2.02 filing per issuer; prior original filing with a distinct reportDate",
        "economic_quarter_deduplication": "Distinct reportDate only; not proof of distinct economic quarters",
        "index_sha256": sha(raw), "maximum_issuer_checks": 32, "requested_events": 12,
        "selected": selected, "checks": checks, "replacement_after_body_failure": False,
        "all_prices_and_model_outputs_unseen": True,
        "scope": "Current-listed deterministic development sample; not a representative small-cap or survivorship-free test"}
    save(root / "cohort.v1.json", cohort)
    return cohort


def split_documents(raw: bytes) -> list[dict]:
    text = raw.decode("utf-8-sig")
    docs = []
    for i, block in enumerate(re.findall(r"<DOCUMENT>(.*?)</DOCUMENT>", text, re.I | re.S)):
        def field(name):
            match = re.search(rf"<{name}>([^\r\n<]+)", block, re.I)
            return match.group(1).strip() if match else None
        body = re.search(r"<TEXT>(.*?)</TEXT>", block, re.I | re.S)
        docs.append({"index": i, "type": field("TYPE"), "filename": field("FILENAME"),
            "description": field("DESCRIPTION"), "body": body.group(1).strip() if body else None})
    return docs


def capture_pairs(root: Path, capture: Capture, cohort: dict) -> dict:
    pairs = []
    body_access_blocked = False
    for issuer in cohort["selected"]:
        pair = {"event_id": issuer["event_id"], "symbol": issuer["ticker"], "cik": issuer["cik"], "documents": [], "errors": []}
        for role in ("prior", "current"):
            if body_access_blocked:
                pair["errors"].append(f"{role}: not requested after SEC body-access block")
                continue
            filing = issuer[role]
            if filing is None:
                pair["errors"].append(f"{role}: required original filing missing")
                continue
            accession = filing["accessionNumber"]
            url = f"https://www.sec.gov/Archives/edgar/data/{int(issuer['cik'])}/{accession.replace('-', '')}/{accession}.txt"
            try:
                raw = capture.get(url, f"{issuer['ticker']}-{role}-{accession}")
                docs = split_documents(raw)
                exhibits = [d for d in docs if re.fullmatch(r"EX-99(?:\.\d+)?", d["type"] or "", re.I)]
                primary = [d for d in exhibits if d["type"].upper() == "EX-99.1"] or exhibits[:1]
                if not primary:
                    raise ValueError("No EX-99 exhibit in original SGML submission")
                doc_records = []
                for doc in primary:
                    body = (doc["body"] or "").encode("utf-8")
                    packet = filing_packet(body, document_id=f"{issuer['event_id']}-{role}", symbol=issuer["ticker"],
                        source_url=url, role="latest" if role == "current" else "prior_comparable",
                        report_date=filing["reportDate"], source_sha256=sha(body))
                    destination = root / "packets" / f"{issuer['event_id']}-{role}-{doc['index']}.json"
                    save(destination, packet)
                    doc_records.append({"exhibit_type": doc["type"], "filename": doc["filename"],
                        "description": doc["description"], "packet_path": destination.relative_to(root).as_posix(),
                        "packet_sha256": sha(destination.read_bytes()), "extraction": packet["extraction"]})
                acceptance = re.search(rb"<ACCEPTANCE-DATETIME>(\d{14})", raw)
                pair["documents"].append({"role": role, **filing, "source_url": url, "submission_sha256": sha(raw),
                    "sgml_acceptance_local_unconverted": acceptance.group(1).decode() if acceptance else None,
                    "availability_basis": "SEC submissions acceptanceDateTime; earnings may have been public earlier",
                    "all_exhibits": [{k: d[k] for k in ("type", "filename", "description")} for d in exhibits],
                    "selection": "EX-99.1, or first EX-99 if absent; all exhibit metadata retained", "exhibits": doc_records})
            except (ValueError, RuntimeError, OSError, subprocess.SubprocessError) as exc:
                pair["errors"].append(f"{role}: {exc}")
                if "status 403" in str(exc) or "status 429" in str(exc):
                    body_access_blocked = True
        pair["usable_paired_text"] = len(pair["documents"]) == 2 and not pair["errors"]
        pairs.append(pair)
        save(root / "pair-status" / f"{issuer['event_id']}.json", pair)
        print(json.dumps({"event_id": pair["event_id"], "usable": pair["usable_paired_text"], "errors": pair["errors"]}), flush=True)
    result = {"schema_version": "earnings-as-filed-pairs-v1", "cohort_sha256": sha((root / "cohort.v1.json").read_bytes()),
        "pairs": pairs, "request_count": len(capture.rows), "usable_pairs": sum(p["usable_paired_text"] for p in pairs),
        "prices_read": False, "models_called": False, "no_silent_truncation": True,
        "table_fidelity": "All visible text retained with cell delimiters; visual table alignment and images unverified"}
    save(root / "source-pairs.v1.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("select", "capture"))
    parser.add_argument("--root", type=Path, default=Path("data/earnings-study-v1/sources"))
    parser.add_argument("--master-index", type=Path)
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    capture = Capture(args.root)
    if args.action == "select":
        result = freeze_cohort(args.root, capture, args.master_index)
        print(f"Frozen selected issuers: {len(result['selected'])}")
    else:
        cohort = json.loads((args.root / "cohort.v1.json").read_text(encoding="utf-8"))
        result = capture_pairs(args.root, capture, cohort)
        print(f"Captured usable pairs: {result['usable_pairs']}")


if __name__ == "__main__":
    main()
