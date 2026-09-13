# Indian Quant Research Agent V2.7 — Master Shortlist

V2.7 preserves the V2.6.4 research engine and adds a decision-oriented UI layer.

## UI tabs
- ⭐ Master — run the full agent and see the strict daily shortlist.
- 🧠 Quant Research — deep historical strategy evidence and model outputs.
- 🎯 Opportunity — historical quant evidence cross-referenced with current patterns.
- 🔎 Pattern Scanner — broad current classical pattern scan.
- 🗂 Research Memory — prior experiment context.

## Master shortlist logic
A stock/strategy row reaches the strict **MASTER** view only when:
1. The same strategy has a historical `PROMISING` verdict after walk-forward/robustness/integrity checks.
2. That same strategy's signal is active on the latest available bar.
3. A current classical chart pattern is detected for the ticker.
4. Pattern volume confirmation is `PASS`.

Rows with the same first three conditions but without volume `PASS` remain visible as `SETUP WATCH` so a potentially developing setup is not lost.

This is a gate-based research shortlist, not a composite buy score and not an order signal.

## Preserved engine
The existing V2.6.4 components remain intact: all 8 swing strategies, Markov, HMM, GARCH-style volatility, Student-t Monte Carlo, train-only walk-forward validation, parameter sensitivity, cost/slippage assumptions, research memory, NSE screening, parallel pattern scan, evidence bundle generation, and server-side worker execution.

## Important interpretation
Historical strategy evidence and current strategy state are separate concepts. A strategy can have strong historical evidence while being inactive today. V2.7 explicitly displays the current signal state so that a historical 52-week-high result, for example, cannot be mistaken for a current 52-week-high setup.

Research only. No order execution. No guarantee of profitability. The NSE universe is current and therefore does not by itself establish survivorship-bias-free historical results.
