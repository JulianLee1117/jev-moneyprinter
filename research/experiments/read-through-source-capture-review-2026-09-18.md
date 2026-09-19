# Micron financial-diagnostic source capture

All six Micron earnings episodes in the frozen 2025–September 18, 2026 census now have archived official prepared-remarks PDFs, together with the fiscal Q1 2025 predecessor. This closes prepared-remarks access; at the end of this capture step, intervening guidance, earnings Q&A, supplier exposure and historical version timing remained unresolved. No economic labels, model judgments, prices or trading signals were assigned in this work.

Six public downloads returned HTTP 200 without retries. Fiscal Q3 2026 reused the previously archived identical PDF and its original capture timestamp. Raw-body hashes, response headers, displayed release dates, observed capture timestamps and source-index provenance are retained in `data/readthrough-financial-v1/source-capture/micron-prepared-capture.v1.json`. Capture dates are September 19 UTC / September 18 Pacific; they are not historical first-public timestamps.

The seven PDFs contain 70 pages. Extraction used bundled pypdf 6.10.0 without adding a project dependency. The archive includes every page and 184 ordered, nonoverlapping passages that reconstruct every extracted character. A separate navigation index lists all 460 literal matches for capital-spending, equipment, construction/facility, gross/net/incentive and annual-period terms. These term matches are navigation aids, not an economic relevance filter; complete text remains available to every eventual model arm.

- Complete passage inventory: `data/readthrough-financial-v1/source-capture/micron-all-passages.v1.json`.
- Page and exact-span topic inventory: `data/readthrough-financial-v1/source-capture/micron-topic-navigation.v1.json`.
- Raw-body and independent extraction checks: `data/readthrough-financial-v1/source-capture/micron-extraction-audit.v1.json` and `micron-extraction-audit-notes.v1.json` in the same directory.
- Explicit open intervals: `data/readthrough-financial-v1/source-capture/micron-intervening-source-gaps.v1.json`.

Independent PDFium extraction confirmed the page counts and all 70 pages' text after removing whitespace and accounting for PDFium's U+FFFE hyphen glyph. Exact repeat pypdf extraction also matched. Five PDFs required cross-reference recovery warnings, retained in per-source logs. One numeric-spacing artifact deserves care: fiscal Q3 2026 page 7 contains whitespace within “22” in pypdf text; the independent source rendering has the same characters without that space. Raw extraction is preserved, not silently corrected.

For navigation, capital-spending passages occur on FY2025 Q1 pages 6/8/9/10, Q2 pages 3/6/8/9, Q3 pages 8/9, Q4 pages 7/9/10, FY2026 Q1 pages 5/6/8/9, Q2 pages 6/8/9 and Q3 pages 6/9/10. Equipment, construction, accounting-basis and annual-period references also appear elsewhere; use the complete inventory rather than only these pages.

At this capture step, each of the six historical predecessor-to-current intervals lacked a complete census of intervening issuer releases and investor-conference material. Prepared remarks also omit earnings Q&A. The earlier archived news/events HTML contains dynamic “Loading” placeholders and does not prove complete coverage. The prospective September 30 event has additional archived issuer releases, but the actual content of the previously identified investor conference remains a gap. These missing sources cannot be treated as evidence that guidance stayed unchanged.

Subsequent work on the same research date captured the [currently indexed Micron news bodies and event metadata](read-through-micron-interval-review-2026-09-18.md) and verified [dated qualitative Lam supplier relationships](read-through-lam-exposure.v1.json). Those additions close the specific feed-access and historical relationship gaps. They do not resolve conference discussion, earnings Q&A, event-specific spending attribution or exact historical document versions; the original capture findings above are preserved.

The official document links and byte hashes are available through the frozen event census and new capture manifest. Current-version PDFs, even with a displayed historical date, do not establish that those exact bytes were available before the registered delayed entry. No price collection is justified by this capture alone.
