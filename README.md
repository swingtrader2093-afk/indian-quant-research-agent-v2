# Indian Quant Research Agent V2.4 — Run & Leave

This version keeps the V2.3 research methodology and adds a server-side worker plus performance optimizations.

## Preserved research layers
- 8 swing strategies
- Next-session-open execution
- Markov return-state model
- Gaussian HMM with K-Means fallback
- GARCH(1,1)-style volatility grid
- Student-t + jump Monte Carlo
- True train-only expanding walk-forward parameter selection
- Parameter sensitivity
- Explicit transaction/slippage assumptions
- Bias checks and research score/verdict
- Trade logs, walk-forward folds, sensitivity files, master CSV and complete evidence ZIP

## Run & Leave
The Streamlit UI launches a separate Python worker process. After pressing Start, the iPhone/iPad browser can sleep or be closed; the worker writes status and results under `jobs/<job_id>/`. Reopen the app later and refresh status.

**Important:** this improves behavior for browser sleep, but Streamlit Community Cloud is not a durable job-queue service. A full app/container restart can discard local job files. For critical long-running production jobs, use a persistent external worker/queue/storage service.

## Performance changes
- Indicator calculations are prepared once per stock/period and reused by backtests, walk-forward and sensitivity.
- Walk-forward test folds retain historical indicator warm-up while restricting trades to the unseen fold.
- Quant-model layer still runs once per stock/period, not once per strategy.
- The walk-forward parameter grid remains intentionally small.
- The parameter-sensitivity function is fixed to call `param_candidates(strategy, base)` correctly.

## Validation status
Static Python compile checks should be run before deployment. Numerical validity still requires actual Yahoo Finance/Streamlit execution and later independent holdout/paper-trading confirmation.
