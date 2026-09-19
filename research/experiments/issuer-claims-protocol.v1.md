# Issuer-first cash claims pilot

The fixed pilot covers **10 current listed operating issuers and 20 financial reports**. It asks whether broad Jev screening can find **direct, quantified, currently outstanding cash claims with unresolved legal or administrative triggers**. It does not test prices or returns.

The machine-readable [protocol](issuer-claims-protocol.v1.json) records all 20 accession numbers, original SEC filing URLs, metadata hashes, selection rules and stop conditions. Registration is September 19, 2026 UTC, September 18 in the project timezone; the information cutoff is September 18 at 20:00 UTC. All existing filings are development evidence.

## Fixed cohort

- **Known examples:** Tenaris (TS), BlueLinx (BXC), and MasterBrand (MBC), the listed company following its American Woodmark merger. These three are convenience seeds and cannot estimate claim prevalence.
- **Industry comparison issuers:** Nucor (NUE), Steel Dynamics (STLD), Cleveland-Cliffs (CLF), Louisiana-Pacific (LPX), UFP Industries (UFPI), Boise Cascade (BCC), and Builders FirstSource (BLDR). They were selected for wood, building-materials or steel operations without screening current claim outcomes or prices. Their claim status is **unknown**, not presumed negative.

All 10 current SEC submissions records were downloaded successfully and confirm the proposed symbol/exchange mappings. Current membership supports this prospective research universe; it is not a historical survivorship-safe stock universe. [SEC submissions data](https://www.sec.gov/search-filings/edgar-application-programming-interfaces)

American Woodmark is historical context, not an independently tradable current target. MasterBrand's current issuer presentation reports the completed merger. A predecessor claim must not be assigned to MBC without explicit ownership/liability evidence, and the merger changes the financial-statement comparison perimeter. [MasterBrand issuer presentation](https://www.sec.gov/Archives/edgar/data/1941365/000194136526000070/q22026investorpresentati.htm)

## Fixed document rule

Use the latest original full financial report accepted by the cutoff and its immediately preceding financial-period report. All nine domestic issuers resolve to Q2 and Q1 2026 10-Qs. Tenaris resolves to full interim-financial-statement 6-Ks for June 30 and March 31, 2026, excluding earnings releases and management-only reports. Compare outstanding balances and legal states; do not compare unadjusted six-month versus three-month cash flows. [Tenaris current financial statements](https://www.sec.gov/Archives/edgar/data/1190723/000117184326005340/f6k_080526.htm)

No replacing an inaccessible, acquired, claim-negative or uninteresting issuer. No adding a third core document to resolve a promising case without a versioned evidence-manifest amendment. Needed follow-up documents are recorded explicitly. Prior seed research cannot silently supply a missing current amount.

## What qualifies

A case requires a sourced direct payer or beneficiary, an outstanding amount or defensible source bound, and an unresolved legal or administrative trigger. The scope includes customs/trade-duty balances and specific disputed tax, regulatory or litigation cash claims or obligations. Ordinary receivables, generic tariff effects, completed refunds, historical income and unquantified upside do not qualify.

Jev screens passages for ownership, balances, cash versus income, units, unresolved triggers, prior disclosures, exceptions and contradictory evidence. Code handles arithmetic. Deterministic retrieval/diffs and a cheap conventional model receive the same source corpus and total review budget. Claim-negative and model-rejected passages need review; answer confidence is not a probability of profit.

## Access and decision gates

The 20 exact original URLs are resolved, but full-document capture has not passed. Earlier direct SEC archive requests returned 403 despite working metadata and readable web research. Use the existing collector identity; this registration changes no identity or settings. Target at most one request per second, cache results, retain full provenance and stop on access denial. An explicitly linked official issuer copy is a possible alternative with separate provenance; search snippets do not establish full-document coverage. [SEC access requirements](https://www.sec.gov/search-filings/edgar-search-assistance/accessing-edgar-data)

Continue the claim-specific build only if at least eight issuer pairs are readable and the pilot finds at least three distinct qualifying cases across three issuers, including two industry comparisons. These are workload and breadth gates, not statistical evidence. All 10 issuers stay in the denominator. Separately report whether at least two comparisons are actually claim-negative; do not manufacture negative controls or replace firms to obtain them.

The known workload is **20 reports and 10 comparisons**. Roughly 40 periodic updates a year across this cohort is a source-volume planning scale if quarterly reporting continues. The number of eligible claims or new legal triggers is unknown and may be zero. Quarterly repetition of one balance does not create independent opportunities.

This registration makes no paid model calls, schedules no monitoring, inspects no prices and places no orders. A later model run needs a finite manifest using existing provider controls; a later price test needs frozen timing, decision, event-grouping and execution-cost rules.
