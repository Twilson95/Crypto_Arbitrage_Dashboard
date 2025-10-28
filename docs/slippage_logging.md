# Interpreting slippage logs

Each triangular leg is planned using simulated order book depth (``TriangularOpportunity``) and then executed live. During execution we log how the realised fills differed from the simulated plan so operators can inspect the per-leg slippage. These logs come from ``TriangularArbitrageExecutor._log_slippage_effect`` and ``_log_route_slippage``.

## Per-leg entries

``Slippage effect for BUY USDC/USD [residual #1]: input +0.00012000 (+0.1200%) output -0.00009000 (-0.0900%)``

* **input** compares the amount of the leg's *source* currency the order actually consumed against what the plan expected (``actual_amount_in - amount_in``). The percentage normalises this delta by the planned amount.【F:cryptopy/src/trading/triangular_arbitrage/executor.py†L1442-L1492】
* **output** performs the same comparison for the leg's *destination* currency (``actual_amount_out - amount_out``).【F:cryptopy/src/trading/triangular_arbitrage/executor.py†L1442-L1492】
* For staggered fills, residual clean-up orders compare against the *expected residual* rather than the entire leg, so tiny follow-up trades no longer appear as ~100 % deviations—even when the primary plan information is missing from the runtime state.【F:cryptopy/src/trading/triangular_arbitrage/executor.py†L924-L1013】

Because both the spend and proceeds are calculated from live exchange data, the two values can diverge independently. Market orders walk the book: when the effective price (VWAP) shifts relative to the plan, the executor may need to spend more quote to reach the available depth and the fills may return more base than forecast (or vice versa). In the example above the buy leg hit slightly worse prices than simulated, consuming 0.4 % more USD, while the deeper fill also yielded 0.2004 % more USDC after fees.

Both deltas therefore describe how the live trade differed from the simulated trade—not a trade-off where one is fixed. Rounding to the exchange's precision and realised fee deductions are also applied to the *actual* amounts, so they can move in the same direction even when the ratio (effective price) stays close to the plan.【F:cryptopy/src/trading/triangular_arbitrage/slippage.py†L214-L320】

## Route summary

``Route execution slippage: final -0.34109896 (-0.3407% vs plan 100.34109896) without fees -0.54442599 (-0.5384% vs plan 100.54442599) profit (w/ fees) -0.2291% (w/o fees -0.1200%, fee impact +0.1123/+0.1123%)``

This aggregates the per-leg deviations by comparing the realised ending balances to the simulated ones. Positive numbers mean you ended with more than planned; negative values indicate shortfall. The extra fields show the realised profit without fees and how much value fees removed, both in absolute and percentage terms. The updated log now also reminds you which simulated totals we compare against, making it easier to reconcile the percentages when the "without fees" plan assumed a larger end balance than the fee-inclusive plan.【F:cryptopy/src/trading/triangular_arbitrage/executor.py†L1494-L1542】

Understanding both levels helps diagnose whether slippage is concentrated in a single leg or distributed across the cycle.
