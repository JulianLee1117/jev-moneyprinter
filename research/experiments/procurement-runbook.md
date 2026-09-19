# Procurement experiment runbook

**Project paused, September 19, 2026.** These commands are preserved for reproducibility, not scheduled or authorized for execution. The [project-wide reopening conditions](../../PLAN.md) govern any future work.

This is the implemented workflow for the frozen [procurement protocol](procurement-protocol.v1.json). The September 19 bounded pilot is complete; its [review](procurement-review-2026-09-19.md) and [aggregate results](procurement-results-2026-09-19.json) record a stop/no-expansion decision. Zero reviewed eligible projects were found; no procurement alpha or Jev-specific trading advantage is established. Source archives, model responses and market quotes live in ignored local `data/` and are not distributed with the public repository.

The authoritative local artifacts are `final-review.v2.json`, `quality.v2.json`, `signals.json`, `market-reviewed.json`, `returns.json`, `contribution-review.json` and `decision.json` under `data/procurement-v1/`. The version-one review/quality files are superseded by seven source-backed project-ID corrections made before signals and prices. Their source/model categories did not change. Original files remain archived. The following commands document reproduction; they are not a recommendation to expand or rerun the completed pilot.

The experiment fixes ten buyer sources and GVA, ROAD, STRL, TPC, PRIM and ORN. July–September 2025 is development; October–December is holdout. Do not replace failed sources, alter the universe or tune rules using holdout prices. Preserve incomplete coverage and failed rows. `conditions` and `prior_known` are descriptive classifications; the current primary rule allows qualifying recommendations and does not add a funded-only or investor-surprise requirement. Model confidence is not a probability of profit.

## Workspace and command behavior

Run from the repository root with Python 3.11 or newer. Use one root for the entire experiment so all model arms and benchmarks share its ledger. The following PowerShell variables simplify the examples:

```powershell
$procurementRoot = "data/procurement-v1"
$procurementProtocol = "research/experiments/procurement-protocol.v1.json"
$procurementArgs = @("-m", "jev_alpha.procurement", "--root", $procurementRoot, "--protocol", $procurementProtocol)
python @procurementArgs --help
```

For PDF text extraction, install the optional dependency with `pip install -e '.[procurement]'`. If the desired Python interpreter is separate from the command interpreter, set `JEV_PDF_PYTHON` to its executable path; it must have `pypdf` available. The default fallback checks the bundled Codex Python runtime beneath the current user's home directory. No particular username or machine-specific absolute path is required.

Commands freeze the protocol under the root. Most artifacts are immutable; output directories and filenames must be new. Reusing the same root is mandatory even when choosing a new report path. Never create another ledger to bypass the budget or resubmit a failed paid request. Without `--live`, collector and model commands prepare or validate their work without provider inference. A dry run may still write a protocol or research report; it is not a filesystem-wide no-op.

`preflight`, `collect`, `references` and `models --live` make public read-only requests. `run --live` and `benchmark --live` can incur model charges using local `OPENROUTER_API_KEY`. `prices --live` uses the existing read-only Alpaca adapter and local credentials. No order endpoint or execution path is implemented. Do not put credentials into a source document, command argument, review or committed file.

## 1. Capture sources and references before inference

```powershell
python @procurementArgs preflight --live
python @procurementArgs references --live
python @procurementArgs models --live
python @procurementArgs collect --live
```

Preflight samples source packages to reveal access and extraction problems; it is not the final census. Collection archives raw bytes, extracted text, capture clocks, package association dates, source inventories and errors. Select the completed census manifest under `data/procurement-v1/sources/` only after inspecting its source results and document inventory. A returned manifest does not itself mean every source is complete. In particular, the HTML discovery adapters can retain `complete: false`; their discovered documents are a lower bound. A document date or meeting association is not a verified publication timestamp.

Model preflight archives unauthenticated OpenRouter metadata for the pinned Jev, nano and mini identifiers, with advertised pricing, context and endpoint availability. It makes no inference call. A paid run repeats necessary availability and price checks before submitting packets. Missing metadata or unsupported requests remain explicit preflight failures; the runner does not silently change the model or endpoint.

The reference capture writes `references/sec-facts.json`; it does not automatically construct a verified ownership map or the separately reviewed `issuer-references.v1.json` used below. Verify and assemble dated alias/ownership and annual-revenue evidence independently, including any needed primary-document captures when SEC access fails. A matching company name alone does not prove listed-parent exposure or a joint venture share. The current map is [procurement-issuer-map.v1.json](procurement-issuer-map.v1.json); its provenance must match local archived files. References published after the document's association date cannot supply historical knowledge.

Choose actual manifest paths before running preparation. `SOURCE_MANIFEST.json` below is a placeholder, not a file generated under that literal name:

```powershell
python @procurementArgs prepare --sources SOURCE_MANIFEST.json --issuer-map research/experiments/procurement-issuer-map.v1.json --references data/procurement-v1/references/issuer-references.v1.json
```

Preparation writes `inputs.json`, with provenance, development/holdout tags and per-amount requests. Use `--out` with a distinct path for a provisional preflight preparation; model commands consume the final root `inputs.json`. The same passage state and seven questions feed all model arms. A target amount anchors each panel, so a packet with several projects does not permit selecting an arbitrary award. Remainder chunks preserve text outside amount windows. Ambiguous punctuation or unsupported amounts stay unresolved; silently guessed amounts are not eligible value evidence.

For the completed initial 100-document feasibility review, add `--source-quality data/procurement-v1/reviews/source-quality-overlay.v2.json` to preparation. This binds matching document IDs and raw hashes to the independent extraction/layout observations and carries them into financial-report coverage. Observations from earlier preflight versions that do not match the final corpus remain counted as unmatched. This overlay does not supply trade labels or silently repair model inputs.

When an award row explicitly links an earlier bid-tab document, preparation also creates paired evidence packets for that bid summary's first-page amounts. Current-row text supplies the current action; earlier bid passages retain their own URL, hash, date and offsets. Models must establish the amount/recipient/contract linkage and distinguish losing bids and item prices. The bid value is never represented as a number printed in the award row or as newly disclosed information. Future-dated bid reports are excluded from that join; all remaining bid-document sections stay in the original corpus.

Common capability checks can skip inference where the frozen rule cannot produce an actionable signal: there is no target amount, its numeric value is unresolved, or no dated verified issuer alias appears in the supplied passages and document opening. Those rows remain **semantic unknowns**, with `model_required: false`, a specific `skip_inference_reason`, and `deterministic_signal_status: "unknown"`. This is a limit of the supplied extraction and mapping, not proof that the source contains no economically relevant event. A separate financial prefilter can mark a definite `no_signal` only where an unambiguous amount is below the applicable verified revenue/materiality bounds. Ambiguous currency units cannot justify that monetary rejection.

Every packet and source document remains in the archive and coverage accounting. All model arms receive the same frozen exemption policy. Exempt results have `status: "not_required"`, empty answers, zero inference cost and no latency; they are never fabricated seven-category model judgments. Failed required calls are distinct from these planned exemptions. Independently review a source-balanced sample, including skipped rows and actual PDF/layout content, to investigate missed amounts, aliases and projects. Model inference on selected packets is not a full-corpus semantic comparison, and a sparse result is not evidence of no opportunities throughout the sources. Copied rules, model answers and automatic parsers are not independent gold labels.

CLI preparation stores requests separately by content hash and keeps references in the input manifest, so the census does not require every request body in memory. Each read verifies its hash. Run commands from the repository root and retain the complete local archive. Same-chunk targets already excluded for the same reason/status may share a diagnostic packet; all original candidates and offsets remain in `aggregated_amount_candidates`, and target counts are reported separately from packet counts. Required targets remain individual. Gold categories describe the representative target, not every grouped amount.

An unprefixed numeric candidate below every issuer's materiality floor remains an unknown abstention when its currency/scale is unverified; a model cannot silently multiply the frozen candidate. Exact FDOT bid-tab source/header evidence can establish a bid-only stage and a nonqualifying signal. These documents remain archived, with individually addressable first-page amounts for later award-row links. This stage exemption does not apply to linked current award packets.

## 2. Benchmark only development and freeze concurrency

Run the benchmark before a full development run if fresh timing matters. Successful cached requests are reusable for categories and original costs but contribute no new latency observations. The default benchmark takes up to 48 required development packets, then assigns disjoint hash-balanced groups to 1, 4 and 8 workers. Each arm receives the same groups; a packet is not replayed nine times to measure throughput.

```powershell
python @procurementArgs benchmark --arm jev --limit 48 --out data/procurement-v1/bench-jev --live
python @procurementArgs benchmark --arm nano --limit 48 --out data/procurement-v1/bench-nano --live
python @procurementArgs benchmark --arm mini --limit 48 --out data/procurement-v1/bench-mini --live
```

Each output directory contains `plan.json`, a top-level `report.json`, and `workers-1/report.json`, `workers-4/report.json`, `workers-8/report.json`. The top-level benchmark reports feed `profile`; individual worker reports feed `label` or `compare`. Build the repeated run arguments in PowerShell without changing any report:

```powershell
$developmentRunArgs = @()
foreach ($arm in @("jev", "nano", "mini")) {
    foreach ($workers in @(1, 4, 8)) {
        $developmentRunArgs += @("--runs", "$procurementRoot/bench-$arm/workers-$workers/report.json")
    }
}
python @procurementArgs label @developmentRunArgs --out data/procurement-v1/development-review-template.json
```

Complete a separate development review file by reading sources without event prices. Every benchmark packet, including failed calls, needs an independent review. For profile selection, reviewed labels must be development-only; pending holdout rows in a template must remain unreviewed. If a benchmark packet is absent from the general audit template, add its input-bound review row. Record all seven category labels, project identity where determinable, `source_signal`, evidence locations, reviewer and measured review time. Use `unresolved` for unresolved source judgments. Set `label_origin: "independent_source_review"` at both review and completed-row level only after that source work; never turn generated model answers into gold labels by changing metadata.

```powershell
python @procurementArgs profile --benchmarks data/procurement-v1/bench-jev/report.json --benchmarks data/procurement-v1/bench-nano/report.json --benchmarks data/procurement-v1/bench-mini/report.json --review data/procurement-v1/development-review.json
```

`profile.json` binds the exact inputs, protocol, model identities, benchmark allocation and review. For each arm, only groups with complete, fresh, valid responses can support a measured choice; the lowest measured run-wall-time per fresh packet wins. The profile also reports source-label accuracy and unresolved gold. It does not infer an accuracy threshold or profitable performance from speed. If no group supports a fresh measurement, one worker is an explicit unmeasured fallback. Input disjointness reduces replay cost but leaves document difficulty and small-sample throughput variability; it is not a controlled identical-request speed trial.

## 3. Complete model coverage, review and freeze signals

```powershell
foreach ($arm in @("jev", "nano", "mini")) {
    python @procurementArgs run --arm $arm --split development --out "$procurementRoot/dev-$arm" --live
    python @procurementArgs run --arm $arm --split holdout --out "$procurementRoot/holdout-$arm" --live
}
```

Development `run` defaults to one worker unless `--workers` is provided. Holdout automatically uses that arm's authenticated frozen profile and rejects a conflicting worker override. Holdout cannot select or modify its own profile. All arms use `root/models`, one **$50** budget and a **$0.25** conservative per-call limit. The runner validates the whole input shape first, then atomically reserves only the next worker group. It stops on exhausted budget or uncertain failures without automatic retries. Actual reported cost releases unused reservations; unknown failed-call cost retains a conservative reservation. Long prompts and advertised maximum endpoint prices can make conservative estimates much higher than eventual bills. A partial run remains partial.

Post-run reporting correction (September 19, 2026): summaries now count `prior_attempt_blocked` rows as inherited failures and, when their cost is null, as unknown-cost records. The original `dev-mini/report.json` remains unchanged: its nine blocked HTTP 429 attempts were retained in the rows and ledger, but its `failed_records` and `unknown_cost_records` summary fields incorrectly read zero. The accounting companion at `data/procurement-v1/reviews/accounting-final.v1.json` records the correction and separates provider-reported charges from unresolved reservations. The pre-run implementation snapshot and all original run reports are preserved. This changes future report summaries only; it does not rerun inference or change cached responses, prompts, frozen inputs, classifications or financial rules.

Build a list of the six full-split run reports for final review and comparison. These reports include cached benchmark answers, so do not also include their worker reports and double-count the same packet:

```powershell
$allRunArgs = @()
foreach ($arm in @("jev", "nano", "mini")) {
    foreach ($split in @("dev", "holdout")) {
        $allRunArgs += @("--runs", "$procurementRoot/$split-$arm/report.json")
    }
}
python @procurementArgs label @allRunArgs --out data/procurement-v1/final-review-template.json
```

Complete `final-review.v2.json` independently while prices remain unopened. The required template includes candidates, disagreements, unknowns and source-stratified sampled rejects; retain exactly its required rows for signal freezing. Earlier independent labels may be reused only for identical bound inputs. Record actual source/issuer/project identities and evidence; separate a bid, proposed authorization ceiling, contingency, repeated historical status table and executed contractor value. Unknown review duration remains unknown rather than zero. Review does not correct a model's predicted categories or swap in a better issuer after outcomes are known.

```powershell
python @procurementArgs compare @allRunArgs --review data/procurement-v1/final-review.v2.json --out data/procurement-v1/quality.v2.json
python @procurementArgs freeze @allRunArgs --review data/procurement-v1/final-review.v2.json
```

Comparison includes deterministic rules, field accuracy, useful project detection, failed/missing rows, fresh timing, cost and review burden. Category accuracy uses the common model-required cohort. Exempt-row reason/status counts remain separate; skipped unknowns are not counted as correct negative answers. Useful-project recall uses independently labeled eligible source/project/issuer identities, including reviewed exempt rows, even when the prediction chose the wrong recipient or failed. It is recall within this purposive reviewed sample, conditional on incomplete extraction and mapping, not proven full-corpus recall. Missing model results and unresolved source judgments remain explicit. `signals.json` freezes the rule outputs, provenance, reviewed project identities, model costs and unknown coverage before market capture. The review sample does not turn an incomplete census into an exhaustive qualifying-project denominator.

## 4. Capture market evidence and evaluate hypothetical accounts

```powershell
python @procurementArgs prices
python @procurementArgs prices --live
python @procurementArgs actions-template --market data/procurement-v1/market/manifest.json --out data/procurement-v1/action-review-template.json
```

The first command prepares price windows; the second uses bounded read-only market requests after signals are frozen. A live market directory is immutable. Inspect incomplete quote/calendar/bar captures and preserve gaps. Complete the corporate-action review from hashed primary evidence, including evidence that covers the interval when claiming no action. An empty provider response is not evidence of no dividend, split or symbol change. Attach that review to a new derived manifest:

```powershell
python @procurementArgs actions-attach --market data/procurement-v1/market/manifest.json --review data/procurement-v1/action-review.json --out data/procurement-v1/market-reviewed.json
python @procurementArgs report --market data/procurement-v1/market-reviewed.json --out data/procurement-v1/returns.json
```

The primary registered account starts with $1,000 and acts only on frozen positive decisions, enforcing overlap and sizing constraints and charging slippage, fees and that arm's full operating cost. Semantically unknown observations explicitly abstain: their source labels remain unknown, and no investment return of zero is imputed to them. The stricter account that requires complete semantic evidence remains available under `complete_evidence_accounts` as a diagnostic. Abstention does not establish complete corpus coverage. Actual API execution failures for an arm/date split block acceptance of that arm/split; missing selected quotes or corporate actions leave its P&L null.

Return comparisons use matched-window IWM and contractor-peer gross midpoint total returns, including independently verified cash dividends, against the strategy's return after costs. Stress increases strategy costs while keeping the gross benchmark reference unchanged, so the increased costs cannot cancel out of excess return. These references are not independently funded executable benchmark portfolios. Base/stress scenarios, week bootstrap and leave-one-issuer-out diagnostics remain descriptive evidence.

Before the completed pilot's first price capture, independent documentation review found that pre-November 3, 2025 historical quote-size replay units were not established. The evaluator therefore accepts an unverified pre-switch size only if the same first sufficient quote qualifies under both raw-share and legacy-round-lot interpretations. An earlier ambiguous quote returns an explicit unknown; later quotes cannot replace it. Verified unit metadata and post-switch shares remain supported. The evidence and pre-price correction receipt are under `reviews/alpaca-quote-size-units.v1.json` and `reviews/pre-price-quote-size-correction.v1.json`.

Historical association dates can support descriptive manual-horizon diagnostics. They cannot establish when the bytes first became available, when an alert could actually have been ready or the first achievable trade. Recorded 2026 inference timing must never be backdated to a 2025 meeting. A 30-project/four-issuer census threshold does not make a sparse holdout statistically reliable. Return gates and bootstrap diagnostics do not prove profitable live execution.

## 5. Review whether Jev adds useful information

```powershell
python @procurementArgs decision-template --report data/procurement-v1/returns.json --quality data/procurement-v1/quality.v2.json --out data/procurement-v1/contribution-review-template.json
```

Complete a separate contribution review against the exact bound return, quality and profile artifacts. A Jev-specific positive finding needs evidence of additional correct useful projects or achievable entry opportunities. Cheaper inference or faster validated responses alone are insufficient. Historical replay cannot justify the achievable-entry basis. If conventional models find the same useful opportunities, retain any supported economic hypothesis while dropping the Jev-specific claim.

```powershell
python @procurementArgs decision --report data/procurement-v1/returns.json --quality data/procurement-v1/quality.v2.json --review data/procurement-v1/contribution-review.json --out data/procurement-v1/decision.json
```

This artifact records reject, inconclusive or promising research status, plus the independent contribution judgment. It does not authorize trades. Unknown or incomplete source, inference, review or market coverage must remain part of the conclusion.

## 6. Optional bounded prospective recorder

The following command is available only after a promising Jev return decision and an explicit bound contribution review resulting in `jev_contribution_verified: true`:

```powershell
python @procurementArgs checkpoint --decision data/procurement-v1/decision.json
python @procurementArgs checkpoint --decision data/procurement-v1/decision.json --live
```

Each live invocation captures one fresh snapshot of the frozen source roster, using a rolling collection window from 30 days before to 30 days after the poll date. It does not start a loop or install a scheduler. An external, explicitly arranged caller would be needed to request future polls. Calls made less than 300 seconds after the prior poll return `not_due`; elapsed gaps and missing coverage remain recorded. No recorder has been started as part of implementing these commands.

The first valid substantive capture for each source initializes that source's observed baseline even if inventory coverage is incomplete. Those bytes are excluded as startup candidates. A source that wholly fails at startup receives its baseline only on first successful recovery. Later previously unseen byte versions become **unclassified observations**, with `first_seen_at`, `inventory_coverage_complete`, and `may_preexist_observation`. A newly discovered URL, incomplete prior/current inventory or polling gap sets prior-existence uncertainty. A capture time is never labeled first publication. Reversions to already known bytes do not become new candidates.

Artifacts live under `root/prospective`: frozen configuration, separate snapshot archives and append-only hash-chained checkpoints. Failures stay visible. The recorder stops after 90 days regardless of candidate volume, or at 30 distinct **reviewed eligible events**. It does not stop merely because 30 unclassified documents were observed. Event eligibility requires separate reviewed records in `root/prospective/eligible-events/`, with schema `procurement-prospective-eligibility-v1`, `reviewed: true`, an existing `candidate_id`, a stable nonempty `event_id`, and `status` equal to `eligible`, `ineligible` or `unknown`. Previously observed eligibility records cannot be edited or removed. Review and deduplication must supply genuine project/issuer event identity; the recorder does not infer it.

Prospective model classification, alert readiness, event eligibility review and subsequent market evaluation are separate work. The current recorder leaves `ready_at` and `eligible_trading_event` unset. It does not prove an opportunity, classify an observation or trade. Incomplete HTML inventory remains a limitation even after a baseline is initialized.
