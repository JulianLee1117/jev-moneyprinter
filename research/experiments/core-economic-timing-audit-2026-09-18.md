# CORE economic and timing rejection audit

Research date: 2026-09-18 (America/Los_Angeles). Reviewer: Codex research agent. This is agent-reviewed development research, not independent human adjudication. No market prices or trading returns were requested or analyzed. The audit covers the eight 2024–2026 notices in the frozen 20-document sample; it does not claim that all 20 notices or all prior sources have been adjudicated.

## Decision

**Do not promote a Federal Register headline-to-NUE/STLD strategy into a return backtest yet. Continue a bounded, source-aware semantic-change experiment; pivot away from CORE as the sole family if it cannot produce several independently dated, economically meaningful increments.** There are useful difficult cases, but the audited sample does not establish a tradable surprise.

Four findings change the experiment:

1. Later notices can formalize a result announced months earlier. Testing their Federal Register dates as new events would manufacture a misleading opportunity set.
2. A single correction can raise one exporter's rate while reducing the residual rate for other exporters. A generic “higher duties help domestic producers” label is inadequate.
3. Published official documents can contradict their own “no changes” summary. The Taiwan example is a real source discrepancy, but discrepancy detection alone does not establish stock-price materiality.
4. The most recent preliminary proposal and the previously operative deposit rate are different reference states. A change relative to a proposal may leave the actual deposit rate unchanged.

Jev's relevant experiment is inexpensive exhaustive screening of **current passage × prior passage × affected exporter/product × issuer exposure**, with uncertain conflicts retained for review. Its probability outputs are screening scores until calibrated on independently reviewed labels. The mechanism is not exclusive to Jev; a Jev advantage must survive comparison against cheaper conventional models under the same total research budget and evidence access.

## Source and timing convention

The frozen selection is `research/experiments/core-pilot-v2.json`; locally collected text and inspection metadata are in `data/evidence-v2/` and `data/dossiers-v2/`. Public-inspection timestamps below are **candidate availability on one channel**, not verified earliest disclosure. Documents signed earlier, public ACCESS memoranda, parties' public submissions, agency press releases, issuer announcements, and newswires may precede them. A signature date does not establish a public timestamp.

The currently accessible web pages were read in September 2026. Their stated historical dates are evidence of prior announcements, but this audit has not obtained contemporaneous snapshots proving exactly which bytes were present at an intraday time. Do not assign an 08:45 timestamp to an unrelated later text version.

## Episode findings

### A. The September 2024 multi-country petition and its 2025 decisions

Selected notices: `2025-08163` (Türkiye correction), `2025-21761` (USITC determinations), and `2025-23430` (CVD orders). Treat these as linked stages of a common petition wave, preserving country and case-number subepisodes rather than counting each document as an independent market observation.

Steel Dynamics publicly announced the petitions on **September 5, 2024**. It identified the products, named the ten countries, described an increase from under 1.25 million to almost 2 million tons of subject imports between the first halves of 2023 and 2024, and described expected agency stages. It also described CORE as strategically important and linked it to galvanizing and paint-line investments. These are evidence of dated STLD product exposure and public investor attention, not proof of an incremental profit amount. [STLD original issuer announcement](https://ir.steeldynamics.com/steel-dynamics-along-with-several-other-organizations-files-trade-petitions-against-countries-on-corrosion-resistant-flat-rolled-steel/)

Commerce announced preliminary AD rates on **April 4, 2025**. Its final-rate announcement was **August 26, 2025**, before the August 29 Federal Register notices. Commerce described about **$2.9 billion of affected imports**. That figure is import value across countries; it is not revenue transferable to NUE/STLD or projected incremental earnings. [Preliminary rates](https://www.trade.gov/preliminary-determinations-antidumping-duty-investigations-corrosion-resistant-steel-products), [final-rate announcement](https://www.trade.gov/press-release/us-department-commerce-issues-affirmative-final-determinations-adcvd-investigations)

USITC's official press release is dated **September 25, 2025**, identifies the same investigation numbers, reports affirmative injury votes, and says orders will follow. The selected `2025-21761` was inspected **December 1 at 08:45 ET** and published December 2; it says the Commission completed and filed determinations on November 28 and explains shutdown-related tolling. Therefore December 2 is not the first public date of the affirmative injury conclusion. Exact differences between the vote, formal filing, report, and legal effect still need tracking. [USITC September 25 release](https://www.usitc.gov/press_room/news_release/2025/er0925_67620.htm), [November 28 determination notice](https://www.usitc.gov/sites/default/files/secretary/fed_reg_notices/701_731/701_733_notice11282025sgl.pdf)

The selected CVD orders were inspected **December 18 at 08:45 ET** and published December 19. They address suspension, deposits, and provisional-measures gaps; some obligations relate to the ITC publication date. They may contain economically relevant implementation details, but repeating the earlier injury/rate conclusion is not a new surprise. [Order text](https://www.govinfo.gov/content/pkg/FR-2025-12-19/pdf/2025-23430.pdf)

**Disposition:** reject late formalization as a fresh directional event without identifying a previously unavailable operative delta. Preserve parent-case clustering in validation. The three notices may contain distinct increments, but cannot be assumed statistically independent.

### B. Türkiye correction: a genuine mixed-direction legal change

`2025-08163` was inspected **May 8, 2025 at 08:45 ET**, published May 9, and signed May 5. It references a petitioners' ministerial-error allegation dated April 14. The official amendment corrects double currency conversion: Borcelik's rate increases **0.00% → 2.04%**, YDÇ remains **15.18%**, and the all-others rate becomes **8.61%**. Changes in deposits apply from publication. [Amended preliminary determination](https://www.govinfo.gov/content/pkg/FR-2025-05-09/pdf/2025-08163.pdf)

The earlier Commerce fact sheet gives the original all-others rate as **15.18%**. Thus the correction lowers that rate by **6.57 percentage points** while increasing Borcelik's. The same fact sheet reports 2023 Türkiye subject import value of **$12,079,582** and volume of **11,413,358 kg**. These old country totals are a scale warning, not a reliable May 2025 exposure estimate; current flows and exporter shares are still missing. [April 4 fact sheet](https://www.trade.gov/preliminary-determinations-antidumping-duty-investigations-corrosion-resistant-steel-products)

**Disposition:** keep as a development test of opposing channels. Do not infer the net NUE/STLD direction without exporter-weighted current exposure and expected pass-through. Check the April 14 public allegation and the May 5 memorandum before treating May 8 as first disclosure. The legal delta is established; surprise and stock-level magnitude are not.

### C. Taiwan review: official contradiction, not an established trade

`2026-00193` (preliminary) and `2026-09903` (final) are the same **A-583-856, July 2023–June 2024 review**. Their inspection times are January 7 and May 15, 2026, both 08:45 ET. The preliminary notice assigns **0.00%** to Prosperity, Sheng Yu, and Great Grandeul Steel Company Limited (Samoa), both in prose and in the table. [January 8 published notice, page 692](https://www.govinfo.gov/content/pkg/FR-2026-01-08/pdf/2026-00193.pdf)

The May notice says the only change is Prosperity's name, yet gives **Great Grandeul 0.99%**, while the two examined respondents remain zero. Its non-examined-company methodology also changes to a prior non-de-minimis rate. The table discrepancy exists in the published GovInfo PDF, not only our public-inspection text extraction. Its assessment paragraph distinguishes Great Grandeul from the zero-rate respondents. [May 18 published notice, pages 28564–28565](https://www.govinfo.gov/content/pkg/FR-2026-05-18/pdf/2026-09903.pdf)

**Disposition:** retain as an unresolved source contradiction. Retrieve the May 8 decision memorandum, any correction, and applicable CBP instructions before labeling the controlling result. Do not let “no changes” text erase conflicting numeric evidence. Even a confirmed 0.99-point exporter-specific change needs shipment volume, issuer competition, and prior-publication evidence before it becomes an economic signal. The same-country **A-583-878** new investigation in the 2024 petition wave is a different case; joining only on product/country would conflate them.

The root agent located a necessary second reference, `2025-07962`, which this reviewer checked in the newly archived source packet: the prior **final 2022–2023 review**, published May 7, 2025, set Great Grandeul and Prosperity at **0.99%**, with Sheng Yu at **0.00%**, and made future deposit requirements effective on publication. Consequently Great Grandeul's May 2026 0.99% is a change from the January 2026 preliminary proposal, but is **unchanged from that prior-final deposit reference**. Prosperity's May 2026 zero, by contrast, differs from the prior-final 0.99%. Any intervening operative instructions still require verification. [Prior final review](https://www.govinfo.gov/content/pkg/FR-2025-05-07/pdf/2025-07962.pdf)

This distinction is essential: store `proposal_delta` and `operative_delta` separately, along with the source-version conflict. Neither is automatically an expectations surprise. A pairwise “new rate minus most recent document rate” calculation would misstate the economic channel here.

Searches did not establish an earlier public release of the May result. That is not evidence of absence. ACCESS was not fully searched/authenticated in this audit. Published PDFs also contain fragments of adjacent notices; extraction must maintain document boundaries.

### D. Korea's 2026 review and Thailand circumvention initiation

`2026-00191` concerns **A-580-878**, the 2023–2024 administrative review. Inspected **January 7, 2026 at 08:45 ET**, it preliminarily assigns zero rates to the mandatory respondents and non-selected firms. It is not a new broad final tariff; final results determine future deposits. An unchanged historic all-others rate is not the rate for every reviewed exporter. Prior completed rates and current import shares are required to infer a meaningful change. [January 8 notice](https://www.govinfo.gov/content/pkg/FR-2026-01-08/pdf/2026-00191.pdf)

`2026-06449` was inspected **April 1, 2026 at 08:45 ET**. It follows NUE and STLD's **February 26** request concerning CORE completed in Thailand using Korean components. It initiates a country-wide inquiry; the suspension paragraph continues treatment for entries already subject to suspension and discusses future determinations. This is not an affirmative final circumvention ruling or a blanket immediate new deposit obligation on all Thai CORE. [April 2 notice](https://www.govinfo.gov/content/pkg/FR-2026-04-02/pdf/2026-06449.pdf)

**Disposition:** retain the circumvention case as a separate research candidate because origin, processing, scope, and importer certification could create non-obvious changes. First obtain the public February request/checklist, affected volumes, and an exact public timeline. Neither document currently passes the economic-and-timing gate.

### E. Korea 2024 corporate successor review

`2024-10643` was inspected **May 14, 2024 at 08:45 ET**, published May 15, and signed May 9. It initiates a successor-in-interest review after Dongkuk CM's February 9 request following a corporate restructuring. Old Dongkuk was excluded from the CVD order; Commerce requests additional ownership information rather than combining initiation with preliminary results. [May 15 notice](https://www.govinfo.gov/content/pkg/FR-2024-05-15/pdf/2024-10643.pdf)

**Disposition:** reject an immediate “duties imposed” interpretation. Any downstream economic change remains conditional on the successor determination and trade volume. Date of the request and restructuring are not proof of the first public content timestamp.

## Dated issuer exposure is necessary, but not sufficient

STLD's September 2024 release supports direct product relevance for later events. NUE's 2024 10-K, filed February 27, 2025, describes galvanized sheet production, a diverse product mix, and the steel-mills segment as 61% of external sales. It also discloses a 51% controlling stake in Nucor-JFE Steel Mexico. The annual report describes that venture's roughly 400,000-ton galvanized-sheet capacity serving Mexico's automotive market. Thus a geographic legal scope must be matched to operating assets; ownership of a Mexican plant alone does not prove U.S. tariff liability or an offsetting loss. [NUE 10-K](https://www.sec.gov/Archives/edgar/data/73309/000095017025028427/nue-20241231.htm), [annual-report detail](https://www.sec.gov/Archives/edgar/data/73309/000119312525060945/d895862dars.pdf)

These disclosures establish **exposure candidates**, not a numeric earnings sensitivity. They do not supply current affected-exporter share, substitution, domestic pricing pass-through, the issuer's covered tonnage, cost offsets, or expectations. Do not label absent quantified evidence as zero exposure. Do not use a February 2025 filing as a feature for a May 2024 event.

## What the next experiment must test

1. Freeze the current/prior document pairs and exporter identities before price access. Where applicable, use both the prior proposal and the previous operative final decision as distinct references. Annotate numeric deltas, opposing channels, uncertainty, and first-known source separately. The Türkiye and Taiwan cases are development examples now; they cannot later become untouched holdouts.
2. Build earliest-source candidates from Commerce announcements, USITC votes/releases, ACCESS public records, and issuer releases. Maintain `observed_at`, `claimed_published_at`, exact bytes, and case/stage identities separately.
3. Require a range of plausible issuer operating-profit impact using dated public exposure. Missing numbers keep a candidate unresolved. A giant country rate or a legal title does not replace magnitude.
4. Test Jev against a conventional low-cost model and deterministic numeric diff under matched source access, cost, and review budget. Measure economically relevant changes found, false positives, and review time; probability confidence is not the scoring label.
5. Only eligible, independently dated events may enter a preregistered price pilot, with parent-episode clustering and realistic post-alert entry. No eligible economic surprise has been fully established by this eight-document audit.

The price-blind evidence supports **pivoting the mechanism toward prior-state comparison and multi-channel timing immediately**. It does not yet select a profitable family or justify live trading. CORE remains a useful rejection fixture; continuing it indefinitely merely because the API is inexpensive would not advance the profit objective.
