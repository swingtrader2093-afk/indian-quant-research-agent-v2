
# Indian Quant Research Agent V2 — All-in-One

This is the automated research/validation layer for the Indian Quant Swing Lab.

## Included in V2

### Strategy engine
- 8 swing-strategy hypotheses
- multi-stock testing
- 5Y / 10Y testing
- CAGR, Sharpe, max drawdown, win rate, profit factor, trade count
- trade-level logs

### Quant models
- 4-state Markov return-state transition matrix
- horizon probabilities (1/3/5/10 sessions)
- Gaussian HMM with K-Means fallback
- GARCH(1,1)-style volatility calibration
- Student-t + jump Monte Carlo scenario distribution

### Validation
- Signal on day T and execution on next session open
- expanding walk-forward
- train-only parameter selection in each walk-forward window
- parameter sensitivity analysis
- trade-count guardrails
- basic bias checks
- explicit cost assumptions

### Evidence exports
Every completed run can export:
- `results_summary.csv`
- `validation_report.md`
- `experiment_config.json`
- individual trade logs
- walk-forward fold files
- sensitivity files
- one ZIP containing everything

Upload the ZIP back to ChatGPT for analysis.

## What is NOT magically solved

### Survivorship bias
A current list of NIFTY stocks is not a survivorship-bias-free historical universe.

To establish that properly, supply a point-in-time universe file containing historical membership changes and delisted names. V2 does not pretend otherwise.

### Indian transaction costs
The UI makes costs explicit and editable. They are research assumptions, not a live broker quote. Update them when needed.

### HMM labels
HMM states are statistical states. R0 does not automatically mean "bear" and R3 does not automatically mean "bull."

### Monte Carlo
Monte Carlo gives a scenario distribution, not a prediction.

## Hosting

Recommended: Streamlit Community Cloud.

1. Download/unzip this project.
2. Create a GitHub repository, e.g. `indian-quant-research-agent-v2`.
3. Upload `app.py`, `requirements.txt`, `README.md` to the repository root.
4. Commit to `main`.
5. Open https://share.streamlit.io/
6. Sign in with GitHub.
7. Create a new app.
8. Choose your repository, branch `main`, and main file `app.py`.
9. Deploy.

## First run

Use 10 liquid names and 10y:
RELIANCE.NS
TCS.NS
HDFCBANK.NS
ICICIBANK.NS
INFY.NS
SBIN.NS
BHARTIARTL.NS
LT.NS
ITC.NS
AXISBANK.NS

Benchmark: `^NSEI`.

Select all 8 strategies.

Click **Run Full Validation Agent**.

Then download **COMPLETE EVIDENCE BUNDLE** and upload it here.

## Intended workflow

1. Agent runs the experiment.
2. Agent creates evidence.
3. ChatGPT analyzes the evidence.
4. Only after that do we decide what needs further testing.
5. Later, we can add point-in-time universe ingestion and a truly independent holdout.

## Important research principle

Do not repeatedly tune parameters until the score looks good. That turns the validation process into another form of overfitting.

The agent's score is a ranking heuristic, not a probability of making money.
