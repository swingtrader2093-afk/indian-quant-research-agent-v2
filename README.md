# Indian Quant Research Agent V2.6.4

## What this version does

One Streamlit app for Indian-market swing-trading research:

1. Broad current NSE equity screening (up to 2,000 symbols).
2. Automatic candidate selection.
3. Deep confirmation using 8 swing strategies.
4. Markov + HMM + GARCH-style volatility + Monte Carlo context.
5. Signal-at-close / next-session-open execution convention.
6. Train-only walk-forward validation.
7. Parameter sensitivity, costs/slippage and bias warnings.
8. Cumulative research memory plus latest five experiment snapshots.
9. Classical current-chart pattern scanner.

## Important V2.6.4 fixes

- **Pattern scanning starts automatically in parallel with deep confirmation** after Stage A screening. It runs server-side, so the browser/iPhone does not sit on a blocking spinner or need to remain awake. Pattern output is saved separately as `pattern_results.csv` and shown in the Pattern Scanner tab.
- Quant research and pattern detection remain logically separate. Pattern results never change backtests, walk-forward folds, parameter selection, or research scores.
- Job status updates are merged so parallel pattern progress cannot overwrite quantitative progress (and vice versa).
- **First-run research-memory bug fixed:** an empty historical ledger no longer turns valid current-run results into an empty `results_summary.csv`. Current results are always retained; historical columns show `0`/blank and a first-run context label when no prior experiment exists.
- COMPLETE evidence bundles include `pattern_results.csv` when available.

## Recommended workflow

Open the app → configure universe/settings → press **Start Full Validation Agent**. Stage A screens the NSE universe, then the app runs deep confirmation on selected candidates while the current-pattern scan runs in parallel. You can lock the iPhone or close Chrome. Reopen the app later and refresh status.

Review the Quant Research tab first. Then review Pattern Scanner and the **Current pattern + historical strategy evidence** cross-reference. Treat all output as research/watchlist evidence, not automatic buy signals.

## Pattern scanner

Heuristic structural detection includes H&S/inverse H&S, double top/bottom, triangles, wedges, flags/pennants, cup & handle and VCP. Pattern score is structural fit only; it is not a probability of success.

## Research memory

The ledger is cumulative and the latest five complete experiment snapshots/reports are retained. Historical results are context only and do not alter untouched walk-forward test folds.

## Operational caveat

The background worker is designed for Streamlit Community Cloud and protects the UI from long-running work, but it is not a durable external job queue. A full container restart can still terminate a running job or lose local job files.

## Bias warning

The current NSE universe is not survivorship-bias-free historically. Point-in-time constituent/addition/removal/delisting data would be required for that claim.

## V2.6.4 reliability fix

- Fixed a real parallel-worker race in `status.json`: quant and pattern threads previously used the same `status.tmp` path, which could cause `FileNotFoundError` during `os.replace()` and leave Pattern status at a partial percentage (for example 63%). Status writes are now protected by a process-local lock and use unique temporary filenames.
- Background pattern data loading no longer uses Streamlit cache APIs; it uses a worker-safe yfinance loader.
- Job/research-memory paths are anchored to the application directory, and the worker subprocess is launched with the application directory as its working directory. This prevents relative-path `FileNotFoundError` after deployment/restarts.


V2.6.4 UI addition: Ranked Results remain pure quantitative ranking; when pattern results are available, the table shows the best current pattern per ticker as informational context. A separate Combined Opportunity View shows current patterns alongside historical strategy evidence without creating a composite buy score or changing the quant rank.
