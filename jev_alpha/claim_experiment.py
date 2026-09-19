"""Bounded all-passage claim experiment with immutable results and no retries."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .claims import select_claim_evidence
from .experiment import digest, run_one
from .jev import JevRequestError
from .store import Store, utc_now, write_new_json


def run_claim_screen(archive: Path, plan: dict, out: Path, *, live: bool = False,
                     arm: str = 'jev', budget_usd: float = 5, workers: int = 4,
                     progress=None) -> dict:
    if out.exists():
        raise ValueError('Preserve existing claim runs; choose a new output directory')
    if arm not in {'jev', 'baseline'} or type(workers) is not int or not 1 <= workers <= 4:
        raise ValueError('Expected a supported arm and one to four workers')
    # Reconstruct the fixed profile and verify lossless offsets before any call.
    select_claim_evidence(plan, {})
    out.mkdir(parents=True)
    responses, runs, halted = {}, [], False

    def submit(chunk):
        try:
            # Independent SQLite connections serialize budget reservations using
            # BEGIN IMMEDIATE; no network request holds that transaction open.
            with Store(archive) as store:
                run = run_one(store, chunk['request'], arm=arm, transport='curl', phase_budget_usd=budget_usd)
            write_new_json(out / 'responses' / (chunk['chunk_id'] + '.json'), run)
            return {'chunk_id': chunk['chunk_id'], 'status': 'completed', 'run': run}
        except (JevRequestError, ValueError, OSError) as exc:
            return {'chunk_id': chunk['chunk_id'], 'status': 'failed', 'error_type': type(exc).__name__,
                    'diagnostic_sha256': getattr(exc, 'response_blob_sha256', None),
                    'validation_reason': getattr(exc, 'validation_reason', None),
                    'halt': not (getattr(exc, 'response_blob_sha256', None)
                                 and getattr(exc, 'validation_reason', None))}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for first in range(0, len(plan['chunks']), workers):
            batch = plan['chunks'][first:first + workers]
            if live and not halted:
                # Wait for each bounded batch before admitting further requests.
                results = list(pool.map(submit, batch))
            else:
                results = [{'chunk_id': c['chunk_id'], 'status': 'skipped_after_failure' if halted else 'dry_run'} for c in batch]
            for result in results:
                run = result.pop('run', None)
                if run is not None:
                    responses[result['chunk_id']] = run['response']
                    result.update({k: run[k] for k in ('request_hash', 'cached', 'latency_ms', 'incremental_cost_usd')})
                    result['reported_original_cost_usd'] = run['response'].get('usage', {}).get('cost')
                halted = halted or bool(result.get('halt'))
                runs.append(result)
            if progress:
                progress({'arm': arm, 'processed_chunks': len(runs), 'total_chunks': len(plan['chunks']),
                          'valid_chunks': len(responses), 'halted': halted})
    selection = select_claim_evidence(plan, responses)
    selected_chars = sum(len(f['text']) for f in selection['selected_fragments'])
    all_chars = sum(len(f['text']) for f in plan['fragments'])
    summary = {'status': selection['status'] if live else 'dry_run', 'arm': arm,
               'chunks_total': len(plan['chunks']), 'chunks_valid': len(responses),
               'passages_total': len(plan['original_passage_ids']),
               'passages_selected': len(selection['selected_passage_ids']),
               'source_text_characters': all_chars, 'selected_text_characters': selected_chars,
               'selected_character_fraction': selected_chars / all_chars if all_chars else None,
               'known_valid_incremental_cost_usd': sum(r.get('incremental_cost_usd') or 0 for r in runs),
               'known_valid_original_cost_usd': sum(r.get('reported_original_cost_usd') or 0 for r in runs),
               'cost_caveat': 'Valid-response costs only here; failed/uncertain attempt charges and reservations remain in the shared attempt ledger.'}
    report = {'schema_version': 'issuer-claim-screen-run-v1', 'created_at': utc_now(),
              'plan_sha256': digest(plan), 'live_requested': live, 'workers': workers,
              'cumulative_archive_budget_usd': budget_usd, 'runs': runs, 'summary': summary,
              'alpha_proven': False, 'interpretation': 'Full extracted-text coverage experiment, not case correctness, economic surprise or profitability.'}
    write_new_json(out / 'report.json', report)
    write_new_json(out / 'selection.json', selection)
    return summary
