# Finite tender filing cohort is available

Full official SEC quarterly master indexes enumerate 155 exact `SC TO-I` initiation filings from 153 issuer CIKs in the fixed June 21–September 18, 2026 window. Separately, that window contains 187 `SC TO-I/A` amendment filings from 146 issuer CIKs. These are filing counts, not counts of currently open offers, distinct payable opportunities or eligible common-share cash tenders.

Sources are the [2026 Q2 master ZIP](https://www.sec.gov/Archives/edgar/full-index/2026/QTR2/master.zip) and [2026 Q3 master ZIP](https://www.sec.gov/Archives/edgar/full-index/2026/QTR3/master.zip). Both returned HTTP 200. The capture verified the complete named index entries using lengths, ZIP CRCs, a repeated local read, final-record/newline checks and zero malformed records. Their published "Last Data Received" headers cover June 30 and September 18. Q2 contributes 21 initiations and 28 amendments within the cohort window; Q3 contributes 134 and 159.

The initial two plaintext captures hit our own 20 MB read limit; they were retained as incomplete. A separately recorded correction authorized exactly two compressed-index requests. This resolved the local capture limit, not an SEC access restriction. Raw bodies, headers, all selected rows, source URLs, hashes and the two probe versions remain in ignored `data/tender-census-v1/`.

The index can associate filings with an issuer, but it cannot prove which offer an amendment modifies or establish current operative terms. Initiations before the window are outside this cohort even if an offer remains open. Issuers with multiple filings require offer-level reconciliation. No offering documents, models, prices or account/order endpoints were requested during this enumeration.

Next apply the [frozen feasibility protocol](tender-feasibility-protocol.v1.json): screen the complete cohort, retain rejection reasons, and establish current common-share cash terms, odd-lot priority and conservative small-account economics before another Jev experiment. The previously screened ABUS candidate failed the conditional floor-spread rule. No qualified profitable opportunity has been established by this census.

Private v2 manifest canonical SHA256: `464a6eb34d5216a82c92a9e9d08d6faa2f0bfe1a96f5a792d2aee8801857d3a9`.
