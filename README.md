# Indian Quant Research Agent V2.5 — NSE Stock Screening & Confirmation

V2.5 is the main stock-screening/research pipeline. It preserves the V2.4 research engine and adds a broad NSE screening stage, automatic candidate selection, confirmation testing, and research memory.

## What it does
1. Pulls the current NSE equity universe (EQ/BE series) from NSE's published `EQUITY_L.csv`, with a fallback universe if NSE is unreachable.
2. Allows manual tickers and can combine them with the NSE universe.
3. Runs a cheap historical screen for all selected stocks/strategies.
4. Automatically selects candidates that deserve deeper attention for that specific stock/strategy.
5. Runs the expensive confirmation engine only on selected candidates:
   - next-session-open backtest
   - train-only expanding walk-forward
   - parameter sensitivity
   - transaction costs + slippage assumptions
   - Markov return-state model
   - HMM / K-Means fallback
   - GARCH-style volatility
   - Student-t / jump Monte Carlo
   - bias checks
6. Maintains a cumulative research ledger and the latest five experiment summary snapshots/reports.
7. Produces an automated research conclusion, master CSV, screening CSV, validation report, trade logs, walk-forward files, sensitivity files and evidence ZIP.
8. Runs through a server-side worker so the iPhone browser can sleep during long jobs.

## Important interpretation
"Strategy deserves attention" means that, **for that particular stock**, the strategy's historical signals showed stronger/repeated historical behavior relative to the other tested strategies. For example, if Volatility Contraction Breakout ranks highly for INFY, it means the VCP/contraction proxy historically behaved better on INFY in the tested sample. It is not a prediction that the next VCP setup will work.

The broad screen is intentionally cheaper than the confirmation stage. Confirmation is where walk-forward, sensitivity, quant models and deeper validation are applied.

## Research memory
Historical results are used as context and consistency evidence only. They are **not fed into the untouched holdout or walk-forward test to optimize parameters**. This prevents meta-overfitting. The cumulative ledger is retained; only the latest five full experiment summary snapshots and reports are kept.

## NSE universe caveat
The current NSE equity list is a current universe. It is **not survivorship-bias-free historically**. A point-in-time historical constituent/delisting dataset is required for that standard.

## Run
- Replace/add `app.py`, `worker.py`, and `requirements.txt` in the existing Streamlit repository.
- Commit and let Streamlit Community Cloud redeploy.
- Start V2.5 from the browser, then the iPhone can sleep.
