# Public snapshot and reproducibility

This repository publishes the project implementation, offline tests, authored research summaries, registered experiment protocols and synthetic examples. It is a research system, with no order-submission implementation and no demonstrated profitable strategy.

The synthetic read-through example works from a clean checkout without credentials, market data or downloaded documents. The offline test suite uses synthetic inputs and the public Federal Register question pack. Historical result summaries describe local experiments; the public checkout does not include everything needed to reproduce the original model runs.

The following material remains outside Git:

- `.env` and all credentials; `.env.example` contains names with blank values only.
- `data/` and `research/data/`, including downloaded documents, Scry/Hacker News/Reddit records, market quotes, SQLite databases and provider request/response archives.
- Detailed local evidence captures, some source-excerpt fixtures and raw machine result files referenced by the research notes.

Protocols and reports retain source URLs, hashes, methods and important negative results. A reference to an excluded local artifact identifies provenance; it is not a public download link. Frozen protocols are retained as written, including historical assumptions subsequently superseded by the current plan. Raw-source availability, provider access and data rights are separate from code availability; no rights to third-party source material are granted by publishing this repository.

Live operations are explicit and budgeted, preserve failures, and use credentials supplied by the operator. Dry runs make no model calls. Native model confidence is not a trade-win probability, and selected interpretation examples are not independent financial observations.
