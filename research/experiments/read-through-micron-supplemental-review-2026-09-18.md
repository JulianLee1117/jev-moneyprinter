# Micron supplemental source audit

Official earnings slide decks are now archived for all six historical events and their fiscal Q1 2025 predecessor: **seven decks, 299 pages and 328 complete text passages**. Six PDFs were downloaded successfully; fiscal Q3 2026 reused its previously archived PDF. Each raw hash, capture time, page extraction and source link is retained. These decks supplement the prepared remarks but do not supply earnings Q&A.

The machine-readable result is `data/readthrough-financial-v1/micron-supplemental/micron-supplemental.v1.json`. Complete slide text is in per-event `.pages.json` files and `micron-slides-all-passages.v1.json` in the same directory. Independent PDFium extraction confirms each page count and is separately preserved. No image-only chart values or table-cell reading order were visually certified; numeric interpretation should retain the original PDF context.

The official event feed lists no transcript attachment for any of the seven earnings calls. Two sampled earnings webcast links, fiscal Q1 2025 and fiscal Q2 2026, returned JavaScript application shells without substantive text. Other known earnings and post-earnings webcast links remain explicitly unrequested within this phase's request limit; they are not declared unavailable. The current feed has no main webcast URL for fiscal Q3 2026, although it lists a separate post-earnings call URL.

All six interim conference entries now have recorded access outcomes:

- February 12, 2025 Wolfe Research: the issuer event-detail page returned HTTP 403. The feed supplied no webcast URL.
- May 14, 2025 J.P. Morgan: the ordinary redirect led to a public page explicitly stating that the recording had expired and the conference website was no longer available.
- August 11, 2025 KeyBanc: the issuer-linked webcast redirected to registration. Registration was not followed or submitted.
- November 19, 2025 RBC: the public wrapper identifies the session and links an embedded KnowledgeVision JavaScript presentation. The embedded application remains unrequested under the phase cap; its content is unknown.
- February 11, 2026 Wolfe Research: the issuer-linked Summitcast resource returned HTTP 404.
- May 20, 2026 J.P. Morgan: the public page explicitly states that the recording had expired and the conference website was no longer available.

This phase used exactly **15 new public requests**, including one separately counted ordinary redirect follow-up. No denied resource was retried, no registration or authentication was submitted, and no model, transcription, price or order calls occurred. All failures and page bodies remain in the ignored local archive.

No Q&A or conference transcript was obtained. Every historical interval therefore retains its unresolved conference-content flag, and every earnings event retains its Q&A flag. Current source versions and current access failures are not proof of historical availability. The FY2026 Q1 deck's filename includes a revision suffix; none of the captured deck versions has a verified historical first-public timestamp. The new slide evidence is suitable for identical-input diagnostics, not a claim that historical prior guidance has been exhaustively reconciled.
