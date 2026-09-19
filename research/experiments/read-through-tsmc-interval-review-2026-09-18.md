# TSMC intervening-disclosure coverage audit

The bounded audit preserves all seven predecessor-to-earnings intervals, 34 unique news-index entries and 29 calendar entries spanning the earnings boundaries. It used 20 public requests, with exact saved-body hashes and actual outcomes recorded privately in `data/readthrough-financial-v1/tsmc-intervening/manifest.v1.json`.

All seven intervals remain incomplete. The official news archive returned identical bodies for different pagination and date-filter requests, so successful HTTP responses did not establish interval coverage. The current investor-meetings page contained no historical entries, while the current events index listed future events. Conference or shareholder-meeting spoken content remains a gap. Empty search results cannot establish unchanged guidance; monthly sales notices are not capex revisions.

One useful original disclosure was captured: TSMC's [March 4, 2025 US investment announcement](https://pr.tsmc.com/schinese/news/3210), whose body is in English. It describes an additional $100 billion investment intention spanning fabs, advanced packaging and research. Its relationship to annual spending must be reconciled with earnings guidance. The amount cannot simply be added to the 2025 annual capex forecast.

The [financial calendar](https://investor.tsmc.com/english/financial-calendar), displayed release dates and present-day source bodies do not verify historical publication times. No model calls, prices, orders or financial labels were used. The resulting packet audit must preserve the missingness rather than permit historical executable-profit claims.
