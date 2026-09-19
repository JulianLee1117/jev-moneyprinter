# Jev alpha research

A command-line research system testing whether Jev's native structured decisions uncover incremental financially useful information. **Profitability remains unproven.** No order execution is implemented. Cheaper inference alone is not a success criterion.

This public snapshot includes implementation, offline tests, synthetic examples, experiment protocols and research summaries. Credentials, collected datasets, issuer source bodies, model-response archives and market quotes are excluded. Results described below were measured in local archives; the public checkout alone cannot reproduce those historical model runs. See [PUBLICATION.md](PUBLICATION.md).

The latest completed experiment tests **semiconductor capex composition and supplier read-through**. Across 13 frozen events, Jev matched 121/139 unambiguous source judgments versus 78/139 for a conventional compact model, and rejected three unsupported aggregate-equipment claims that the control accepted. Native probability routing changed **zero primary decisions** versus categorical Jev. All 13 primary references were negative, so no opportunity-detection sensitivity or profitability is established. See the [closed comparison review](research/experiments/read-through-closed-review-2026-09-18.md) and [aggregate results](research/experiments/read-through-closed-results-2026-09-18.json).

This aggregate-WFE rule/source combination is closed as an opportunity-generating test. Its narrow interpretation improvement is preserved. **The objective is trading alpha specifically.** The tender census is archived preparation, not the current flagship. No event-trading returns have been measured. The next research recommendation is a broad, dated earnings/guidance cohort testing whether Jev semantic features improve delayed net returns beyond numerical, price and conventional-model controls. This remains an unproven hypothesis; see [PLAN.md](PLAN.md).

The `jev_alpha.readthrough` runner compares native Jev decisions with identical-evidence compact chat, preserves probability distributions and uses explicit condition logic with abstention. Original source text is separated from reviewer labels. The completed comparison used 26 valid requests, no retries and one shared $2 model cap. `jev_alpha.readthrough_results` distinguishes correct support, unsupported claims and lost support from abstention. The original financial protocol remains frozen; market returns have not been measured.

The source archive now contains all 15 current/predecessor issuer documents and seven Micron slide decks (526 pages combined), plus dated Lam relationship evidence. `jev_alpha.financial_sources` assembles source references for all 13 events, verifies archived hashes and chronology, and rejects future supplier filings. It preserves missing Q&A, intervening-guidance gaps and post-call amendments; source capture does not authorize prices or establish alpha. See the [TSMC version audit](research/experiments/read-through-tsmc-source-review-2026-09-18.md), [Micron supplemental audit](research/experiments/read-through-micron-supplemental-review-2026-09-18.md), and [Lam exposure audit](research/experiments/read-through-lam-exposure.v1.json). Current transcript versions cannot support executable historical-profit claims without independent availability evidence.

The previous **paid-software customer operating changes** sample failed its evidence gate: independent agent review of all 749 records found no episode meeting every required qualification. The conventional-model comparison remains partial after HTTP 429 failures, and four Jev responses failed the frozen validation rule. See `research/experiments/operating-changes-review-2026-09-18.md`. Preserve this result; do not expand it because its API calls were cheap.

The latest CORE test completed twenty Jev judgments and nineteen valid baseline judgments. The frozen Jev rule produced zero long signals. All inputs lacked separately supplied prior documents; this is a contract limitation, not evidence that Jev cannot find alpha. One baseline response remains invalid/unknown, one diagnostic retry is recorded, and all three public daily-price downloads failed. **No returns were measured.** See `research/experiments/core-daily-signal-review-2026-09-18.md`.

The Jev treatment is a panel of atomic evidence decisions with native uncertainty, followed by deterministic economic-condition propagation. Compare it with numeric rules, conventional models and Jev's own categorical outputs. These methods are also possible with older models; incremental decision quality and post-alert returns must be demonstrated. Model confidence is not a profit probability. See [PLAN.md](PLAN.md).

## Run locally

Python 3.11 or newer; no runtime packages, API keys or installation needed for offline commands. Run from the repository root. `python -m jev_alpha --help` lists commands. An optional `pip install -e .` exposes `jev-alpha`, but is unnecessary.

Run the tests and a fully offline synthetic example from a fresh checkout:

```powershell
python -m unittest discover -s tests -q
python -m jev_alpha.readthrough prepare --fixtures research/examples/readthrough-synthetic.v1.json --out data/example/plan.json
python -m jev_alpha.readthrough run --plan data/example/plan.json --archive data/example/archive --out data/example/dry-run
```

The example contains invented companies and disclosures. It demonstrates preparation and validation, not model performance or an investment opportunity. Output paths are immutable; choose a new path for another run. Provider calls require local credentials and explicit `--live`.

The following commands are **local archive replay only** and require the unpublished original evidence. The read-through runner has a separate module entry point; its recorded plan is `data/readthrough-development-v1/plan.v2.json`:

The offline source inventory uses `python -m jev_alpha.financial_sources --help` for its manifest arguments. It accepts repeated `--interval` and `--supplement` paths and writes a new immutable inventory with `--out`. It makes no network/model requests and does not convert the inventory into a signal plan. The source manifests and copyrighted documents are local archive inputs, excluded from this public snapshot.

```powershell
python -m jev_alpha.readthrough prepare --fixtures data/readthrough-development-v1/source-fixtures.v2.json --out data/readthrough-replay/plan.json
python -m jev_alpha.readthrough run --plan data/readthrough-replay/plan.json --archive data/readthrough-development-v1/models --out data/readthrough-replay/run --live
```

Unchanged successful requests reuse the archive. Failed paid requests are not retried. Omitting `--live` produces a preparation check with no model submissions or performance claim. This module does not collect prices or place orders.

## Operating-change experiment

Use the **same `--data-dir` for every step** so source calls share one $5 ledger and Jev/chat calls share one $10 ledger. Put `SCRY_API_KEY` and `OPENROUTER_API_KEY` in the ignored `.env`. Commands default to offline preparation; `--live` authorizes bounded requests. Output paths must be new. Do not create a new archive to evade an exhausted budget.

```powershell
python -m jev_alpha --data-dir data/operating-changes-v1 operating-collect --out data/operating-changes-v1/capture.json --live
python -m jev_alpha --data-dir data/operating-changes-v1 operating-threads --capture data/operating-changes-v1/capture.json --out data/operating-changes-v1/capture-threads.json --live
python -m jev_alpha --data-dir data/operating-changes-v1 operating-freeze --capture data/operating-changes-v1/capture-threads.json --out data/operating-changes-v1/cohort.json
python -m jev_alpha --data-dir data/operating-changes-v1 operating-prepare --cohort data/operating-changes-v1/cohort.json --split development --out data/operating-changes-v1/development.json
python -m jev_alpha --data-dir data/operating-changes-v1 operating-run --plan data/operating-changes-v1/development.json --arm jev --out data/operating-changes-v1/dev-jev --live
python -m jev_alpha --data-dir data/operating-changes-v1 operating-run --plan data/operating-changes-v1/development.json --arm baseline --out data/operating-changes-v1/dev-chat --live
python -m jev_alpha --data-dir data/operating-changes-v1 operating-semantic --cohort data/operating-changes-v1/cohort.json --split development --out data/operating-changes-v1/dev-semantic --live
```

Freeze the profile after development before running `--split evaluation`; both semantic splits pin the verified `qwen3-rerank-4b` model by default. An explicit `--model` override must remain fixed across splits. Raw texts and full envelopes stay in ignored `data/`. Missing strata remain visible; neither unavailable controls nor unknown model answers count as negative evidence. `operating-compare` joins both split directories for each arm; `operating-review-sample` retains candidates, unknowns and eighty deterministic rejects. `operating-evaluate` requires independently completed reviews and control coverage before applying the twenty-episode/four-vendor feasibility gate. This historical current-text sample cannot establish a predictive trading return.

If a run halts, `operating-run --resume-report <prior-report.json> --out <new-directory>` continues only previously unattempted requests, with the same plan and data directory. Successful results must exist in the archive cache; failed attempts remain unknown and retain their cost reservations. It never retries a failed paid request. `worker_count` and continuation provenance remain in the report. A fully reviewed source population can independently fail the evidence-density gate even when the comparative model benchmark is incomplete.

Scry's public researcher offer is marked non-commercial, and production permission remains unresolved. This implementation registers private technical evaluation only; it does not establish commercial data rights. Original Hacker News API capture is available for explicitly supplied native IDs, with limited coverage recorded. Sources: [Scry pricing](https://scry.io/#pricing), [terms](https://scry.io/legal/terms), [rerank contract](https://scry.io/docs/rerank).

A September 18 integration test returned HTTP 200 and five historical SIP SPY quotes; its deliberately tiny pagination limit means it was an access check, not a complete price study. Configure your own local Alpaca credentials for live read-only requests. The adapter has no order endpoint and does not switch silently to IEX. Missing credentials produce an explicit receipt and no network call:

```powershell
python -m jev_alpha --data-dir data/operating-changes-v1 alpaca-smoke --query research/experiments/alpaca-sip-smoke.v1.json --out data/operating-changes-v1/alpaca-smoke-check --live
```

The supplied smoke query is a fixed September 17, 2026 regular-session window. Future queries must supply a verified exchange session, explicit historical symbol mapping, and timestamps at least fifteen minutes old. A successful quote capture establishes access for that request, not an executable fill or complete return study.

## Earlier development commands

The import commands below require original local captures under `research/data/`, which are not distributed in the public repository. Offline tests use synthetic fixtures and need none of these captures.

```powershell
python -m unittest discover -s tests -v
python -m jev_alpha import-discovery research/data/core-fr-discovery-2026-09-18.json
python -m jev_alpha import-pi research/data/core-public-inspection-example-2026-14026.json
python -m jev_alpha census
```

The recorded discovery reproduces 362 documents and 207 CORE title matches. This is a precision-oriented discovery screen: omnibus notices can be relevant too. Neither count measures independent events or trades.

The default `data/` directory is ignored by git. Source bytes are content-addressed under `data/blobs/`; SQLite preserves append-only observation/version history and a derived current metadata view. An imported capture timestamp is separate from local ingestion time. CLI outputs and human dossiers are never silently overwritten.

## Freeze a sample and collect public evidence

The current local run uses `data/pilot-selection-v2.json` and `data/dossiers-v2/`. To create a new sample in another directory:

```powershell
python -m jev_alpha select --count 20 --out data/my-pilot.json
python -m jev_alpha collect --selection data/my-pilot.json
python -m jev_alpha dossiers --selection data/my-pilot.json --out data/my-dossiers
python -m jev_alpha audit --selection data/my-pilot.json --dossiers data/my-dossiers --out data/my-audit.json
```

`collect` makes read-only requests for detail metadata, published text, public-inspection metadata and available inspection text. Missing sources produce warnings; raw responses are preserved. No source timestamp automatically becomes verified first-public availability. These API/text renditions do not establish that an earlier Commerce announcement or an official PDF contained the same information. Original official PDF links remain in source metadata for review; this version does not parse PDF tables.

Selection v2 takes a deterministic round robin across title-stage hints and years, alternating recent and older years, without outcome data. This sample is for rejecting an impractical hypothesis; it is not a representative return dataset. The original v1 selection was retained locally because oldest-first ordering underrepresented recent cases. All changes happened before price inspection or model inference.

The exact current selection is also preserved in `research/experiments/core-pilot-v2.json`. All 20 selected documents have archived metadata/text and evidence packets under `data/evidence-v2/`; complete economic and prior-state verification remains unfinished. Ten of the 20 full-text screening packets exceed the intentionally conservative default input guard. Lossless passage fanout now handles those large inputs; whole-document packets still need explicit selection before a compact interpretation panel.

For fresh discovery:

```powershell
python -m jev_alpha discover --start 2025-01-01 --end 2026-09-18
```

The collector partitions overly large date windows to respect the Federal Register's 2,000-result pagination limit, checks result counts and spaces requests. No scheduled collection has been installed.

## Review economics before paying for prices

Each dossier starts unknown. Fill it only from reviewed evidence:

- `episode_id`: reviewed shared-announcement/case grouping; separate documents, issuers and country rows are not automatically independent.
- `material_change`, `issuer_exposure`, `economic_magnitude`: separate `supported`, `unsupported` or `unknown` judgments, rationale and citations shaped as `{ "source_url": "https://…", "excerpt": "supporting passage" }`. An excerpt-shaped string is validated structurally; the application does not verify that the citation proves the assertion.
- Exposure also needs issuer IDs and an `as_of` date. Product relevance alone cannot support economic magnitude; document a material exposure/earnings mechanism and its uncertainty.
- `earliest_availability`: timezone-aware timestamp, earlier-source check and `content_version_matches_timestamp: true` only after reviewing which text was available then. A publication-day date is not an intraday timestamp.
- `market_expectations`: remains unknown without independent evidence. Legal novelty is not market surprise.
- `review`: named completed review with a timestamp. Templates/model outputs cannot complete review automatically.

The audit preserves all selected cases, including exclusions, checks their selection hash and groups assigned episodes. It returns `needs_review`, `reject` or `ready_for_price_pilot`. Three resolved candidate episodes is a small-pilot feasibility threshold, not statistical power or proof of profit. The entire selected cohort must be resolved. Citation and episode correctness still require a reviewer.

## Prepare Jev without spending money

```powershell
python -m jev_alpha collect 2026-14026
python -m jev_alpha packet 2026-14026 --out data/state.json
python -m jev_alpha jev-prepare --state data/state.json --panel screening --out data/request.json
```

`packet` retains identifiable source passages and explicit unknown rates/exposures. Add earlier documents with repeated `--prior DOCUMENT_ID`; the cutoff is conservative and same-day prior documents are rejected. Model history is not automatically linked or complete. Source-only prompts reduce hindsight risk but cannot eliminate model knowledge of later events.

The packet contains all text by default. For large sources, review its passage IDs and supply an explicit JSON array through `packet --passages PATH`. The selection and original passage count are recorded; dropped exceptions must be audited. Nothing is silently truncated. Use `--panel full` to prepare all 28 questions once the evidence merits deeper evaluation.

`jev-prepare` has no network or credential access. The adapter pins `typesafe/jev-1.13` on OpenRouter's `/api/alpha/decisions`, embeds the evidence-only policy, validates question types and preserves full response distributions. Missing choice/score distributions are rejected even though the gateway schema permits them. Live Jev inference has now succeeded, including the full 28-question panel.

## Explicit live inference

Configure `OPENROUTER_API_KEY` in your local environment or the ignored `.env` file; do not put the key in a tracked file or chat. The loader reads only this named value, never executes/interpolates it, and prefers the process environment. Once configured:

```powershell
python -m jev_alpha jev-run --request data/request.json --out data/response.json
```

This command can incur a model charge. Default local guards: conservative input estimate <=28,000, estimated cost <=$0.01 per request, cumulative accounted/reserved cost <=$20 per archive. Estimates use dated public prices and a deliberately conservative byte-based proxy; they are **not provider-enforced spend guarantees**. Verify pricing/account limits before live use. Configurable options are `--max-cost-usd` and `--phase-budget-usd`.

Completed exact requests reuse their cached response. Attempts are recorded before submission; timeouts, invalid responses and interruptions retain a reserved cost and block automatic resubmission. A deliberately requested diagnostic retry requires a recorded reason and keeps the original uncertain cost reservation; there is no automatic retry. Credentials appear only in the Authorization header, and provider errors are redacted. Finite parsed 200 responses that fail validation are archived for diagnosis when they contain no credential echo; malformed JSON and HTTP error bodies are discarded. Valid reported costs remain recorded even when subsequent validation fails.

## Current boundary

The collector, reproducible sample, review gate, provider adapter, dated agent-reviewed issuer snapshots, and source-bound transition comparisons are implemented. Automatic legal-rate extraction, exhaustive historical exposure/case linkage, calibration, executable price ingestion, event-return analysis and prospective performance are not yet implemented. The next milestone is to establish economically meaningful, independently timed episodes before a fixed price study. Preserve failures and compare simple rules, a small Jev screen, the full panel and a stronger model on identical evidence before claiming a Jev advantage.

## Parallel evidence screening and a live comparison

The implementation now supports a Jev-specific passage workflow: every passage receives its own typed decision while multiple decisions share a compact state. Oversized passages are split losslessly with parent IDs and character offsets. Every fragment must appear exactly once; missing/invalid answers keep their evidence and mark the result incomplete. Selection retains operative changes, scope/exceptions, uncertainty and neighboring context, along with a deterministic sample of rejected passages for review. Summing mutually exclusive classes from one distribution is allowed; multiplying unrelated question probabilities into a trade score is not.

```powershell
python -m jev_alpha fanout-prepare --state data/evidence-v2/2026-09903.json --out data/fanout-plan.json
python -m jev_alpha fanout-run --plan data/fanout-plan.json --out data/fanout-dry
python -m jev_alpha fanout-run --plan data/fanout-plan.json --out data/fanout-live --live --budget-usd 1
```

The first command only prepares requests. The second demonstrates fail-closed behavior with no responses and no charges. The third makes paid calls. `--budget-usd` caps cumulative estimated/accounted reservations in this local archive, including earlier attempts; it is not a fresh allowance on each invocation or a provider-enforced guarantee. No automatic retry follows an uncertain call.

A separate identical-evidence comparison runs rules, Jev and `openai/gpt-5.4-mini` on the same state/questions. All arms use a frozen keyword/neighborhood retrieval baseline; the labels' answers/excerpts are not used to choose input. The comparison therefore measures retrieval plus interpretation, not pure full-document interpretation. Post-retrieval diagnostics record omitted label-support passages. This small comparison does not itself measure Jev fanout recall or fixed-budget coverage gains.

```powershell
python -m jev_alpha compare-prepare --out data/my-comparison
python -m jev_alpha compare-run --manifest data/my-comparison/manifest.json --out data/my-rules-result
python -m jev_alpha compare-run --manifest data/my-comparison/manifest.json --out data/my-live-result --live --budget-usd 1
```

The 12 source-cited labels across eight documents in `research/benchmarks/core-labels.v1.json` are **agent-reviewed development examples**, including procedural and assessment/deposit distinctions. They are not independently adjudicated human ground truth, a representative sample or a holdout. Never use their scores to estimate financial alpha. This runner rejects holdout labels to prevent accidental reuse as development material.

Metrics include every labeled observation when reporting accuracy/coverage, retain missing/failed calls, and deduplicate cost metadata shared across questions in one call. Choice Brier uses the raw multiclass squared-error sum; noul Brier is binary. Baseline probabilities are self-reported chat outputs; Jev distributions are native outputs. Neither is assumed calibrated in this domain. Costs are actual reported usage when available; cached runs retain original inference costs for model comparisons but incur no new charge. Partial model runs are inconclusive for equal-evidence quality.

Live responses revealed two-decimal choice distributions whose mass sums to 0.99. The validator permits only narrowly bounded quantization error, records the deviation and preserves the raw values; it never renormalizes. Nonfinite/out-of-range values, missing choices and larger/arbitrary mass errors still fail. A saved rounded response was revalidated without another model call.

On this Windows host, the initial urllib transport added roughly 40 seconds even to public metadata requests. `compare-run` and `fanout-run` support `--transport curl`; it sends the key/body through stdin only, never shell arguments, files or logs. Curl's default configuration, redirects and retries are disabled. Curl was tested live; do not compare its latency directly with earlier urllib measurements as if the difference came from the model.

## First live development results

Recorded September 18, 2026 in `research/experiments/live-development-check-2026-09-18.json`. On the 12 agent-reviewed labels, Jev agreed with 8, GPT-5.4 Mini with 6, and keyword rules with 5 (rules abstained on 4). Reported inference cost for the eight-document identical-input comparison was $0.002949702 for Jev and $0.0412515 for GPT-5.4 Mini. This is one baseline and a tiny purposive development check, not a general model ranking, calibrated accuracy estimate or trading result.

The passage workflow evaluated all 195 passages of one actual document in 13 requests, retaining 167 passages including uncertainty/context. Its original validated responses cost $0.003653622, excluding an earlier uncertain failed attempt whose reservation remains recorded. Retention is not recall: rejected passages and missing exceptions still need independent review.

A full 28-question call using compact evidence completed through curl in about 499 ms and cost $0.000437052. This is a single observed call. Empty issuer evidence nevertheless produced some `none`/`no_documented_offset` answers, so `panel-check` explicitly flags unsupported absence claims and prior-state gaps while preserving the raw probabilities:

```powershell
python -m jev_alpha panel-check --request data/full-panel-live-request.json --response data/full-panel-live-response.json --out data/my-panel-review.json
```

This check never authorizes a trade. The profit audit, dated exposure research, independent labels and return study remain separate. Both economic relevance and sufficiently reliable interpretation still have to be established.

## Source transitions and the direct cash-flow experiment

The follow-up audit reviewed eight recent notices from the frozen twenty-document CORE sample. Some December 2025 formal notices repeat decisions already announced in September. An agent-reviewed exposure map now has six dated NUE/STLD annual snapshots, but product scale still does not quantify the earnings effect of a particular remedy. See `research/experiments/core-economic-timing-audit-2026-09-18.md` and `core-exposure-map.v1.json`. The other twelve dossiers remain unresolved; this is not a twenty-event negative-return study.

Two targeted, manually assembled development cases now separate the **prior preliminary proposal** from the **earlier operative deposit instruction**. The latter can itself be provisional: the Türkiye preliminary investigation imposed provisional deposits. Matching verifies case, exporter, rate type and, for proposal comparisons, review period. Decimal arithmetic calculates percentage-point differences. Prior-source dates, selected source IDs, model inputs, development labels and responses are frozen; historical version/first-public uncertainty remains explicit. Long repeated passage hash prefixes are compressed losslessly within each document.

```powershell
python -m jev_alpha transition-prepare --spec research/experiments/core-taiwan-transition.v2.json --out data/my-transition-manifest.json
python -m jev_alpha transition-run --manifest data/my-transition-manifest.json --out data/my-transition-dry.json --with-baseline
python -m jev_alpha transition-run --manifest data/my-transition-manifest.json --out data/my-transition-live.json --live --with-baseline --budget-usd 1
```

The two live cases returned 23/24 matched agent-reviewed labels for Jev and 21/24 for GPT-5.4 Mini. Costs were $0.000815052 and $0.01371315 respectively, about 16.8 times different for these calls. These are **two selected development cases**, not 24 independent events or a general accuracy estimate. Both models received manual table transcriptions and identical selected passages. Most questions concern evidence sufficiency; this does not establish autonomous discovery or financial forecasting.

Jev flagged the Taiwan source contradiction that the baseline missed. Both models misclassified Türkiye's mixed rate changes: Jev selected decrease and the baseline increase, while Borcelik rises from 0 to 2.04 and all-others falls from 15.18 to 8.61. `guard_transition` compares the raw model label to code-computed directions and requires review on disagreement. It does not change the saved probabilities or use the expected labels. Taiwan's contradiction also requires review. Neither case is trade-eligible. Results and limitations are saved in `research/experiments/transition-development-results-2026-09-18.json`.

The new `coverage.py` evaluates exact source-support spans and includes missing/failed selections. A retrospective diagnostic on two pre-existing Taiwan labels retained complete support for 2/2 with Jev versus 1/2 with keyword retrieval. At approximately matched review-text volume, keyword retained 10,869 characters and Jev 10,921; the result remained 1/2 versus 2/2. This tiny diagnostic is not exhaustive recall, matched total research cost, or measured reviewer time. It required no new paid calls.

The next economic hypothesis is **OCTG/Tenaris cash deposits, refunds and scope changes**, documented separately in `research/experiments/next-family-decision.md`. The causal link is more direct than competitor-stock effects, but source frequency is sparse and domestic-production offsets matter. The fresh 2024–September 2026 census has only six Argentina/Mexico-title notices. No return series has been inspected or used to choose the family. Incidental search snippets and issuer repurchase tables contained price information, so this work is not described as completely blinded.

The bounded protocol is saved as `research/experiments/octg-ts-protocol.v1.json`, with sourced state in `octg-ts-current-state.v1.json`. It limits the initial study to Mexico/Argentina and three episodes, separates historical development from genuinely new prospective information, and defines a conditional paper observation without activating orders or schedules. TAMSA's May 2026 preliminary 1.62% does not implement a reduction from the prior final 26.10%. ACCESS lists a candidate November 3, 2026 final-results announcement for the correct review period; the date may change. ACCESS also lists the preliminary notice/memorandum on May 8, before May 12 public inspection and May 13 publication, reinforcing why Federal Register-only timing is insufficient. Covered entered value, company-specific deposits, intervening instructions and net cash impact remain unresolved. The two official notices are now archived locally as `2026-09464` and `2025-17071` for the next comparison.

## Registered daily diagnostic

`daily-predict` is dry by default; `--live` submits the frozen requests. `--resume-signals` preserves completed and failed observations while attempting previously skipped requests. Safely archived invalid model responses remain unknown; transport/account/budget failures halt further submissions. Files are never overwritten. A separate explicit diagnostic retry is recorded in the original CORE execution amendment.

`daily-prices` is also dry by default; `--live` requests bounded public daily data only after all model requests are finalized. `daily-evaluate` is offline, checks source/protocol/prediction hashes and every cohort date, and refuses incomplete captures. Failed model arms make their complete-cohort means unavailable. It uses a fixed delayed entry, a five-session window, illustrative costs and overlap clusters; it never reports an executable strategy or annualized Sharpe.

```powershell
python -m jev_alpha daily-predict --manifest data/daily-diagnostic-v1/prediction-manifest.json --protocol research/experiments/core-daily-protocol.v2.json --out data/my-daily-plan
python -m jev_alpha daily-prices --protocol research/experiments/core-daily-protocol.v2.json --signals data/daily-diagnostic-v1/predictions-final/signals.json --out data/my-price-plan
```

The saved CORE signals produced 20 cash observations for Jev, versus 7 long, 12 cash and 1 invalid for the baseline. These are signal counts, not accuracy or profit comparisons. Yahoo returned errors for all three symbols, so no daily return report exists.

## Full-filing claim screening

`filing_text.filing_packet` extracts all visible HTML text, excludes scripts/hidden inline-XBRL facts, and preserves table cell separators and source hashes. It does not OCR images or certify visual table fidelity. `claims-prepare` creates four independent typed questions per fragment: claim lifecycle, explicit amounts, retained-cash offsets and later updates. Every supplied character is covered under request limits; fragments retain parent IDs and offsets. No keyword preselection is used.

```powershell
python -m jev_alpha claims-prepare --packet data/issuer-claims-v1/packets-v2/UFPI-latest.json --out data/my-claim-plan.json
python -m jev_alpha claims-run --plan data/my-claim-plan.json --out data/my-claim-dry
```

Paid execution requires `--live`; up to four concurrent requests use independently serialized cumulative budget reservations. Invalid/missing chunks retain their entire evidence. Transport failures stop new batches, with at most the current batch already in flight. Failed paid calls are never automatically retried. `--arm baseline` supports the same questions/evidence but is not yet a completed full-filing comparative result.

The frozen UFPI paired-filing development run completed **2,201 passages / 8,804 typed judgments / 248 requests**, with no invalid results, reporting **$0.065381442** of inference cost. The combined selected text was 23,687 of 288,158 extracted characters (8.22%). Both known IEEPA support paragraphs were retained. These are development support checks, not measured exhaustive recall. Frozen keywords also retained both; independent review of all 44 Jev-only direct selections found no incremental qualifying claim. Jev selected 16.8% fewer characters than the keyword queue (23,687 versus 28,471), while selecting more fragmented passages (162 versus 82). This is a review-volume observation, not a measured recall advantage. Low inference cost alone does not establish an alpha advantage. Protocol: `research/experiments/ufpi-full-passage-protocol.v1.json`.
