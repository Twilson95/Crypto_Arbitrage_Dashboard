# Interpreting slippage logs

Each triangular leg is planned using simulated order book depth (``TriangularOpportunity``) and then executed live. During execution we log how the realised fills differed from the simulated plan so operators can inspect the per-leg slippage. These logs come from ``TriangularArbitrageExecutor._log_slippage_effect`` and ``_log_route_slippage``.

## Per-leg entries

``Slippage effect for BUY USDC/USD: input +0.40000000 (+0.4000%) output +0.20002000 (+0.2004%)``

* **input** compares the amount of the leg's *source* currency the order actually consumed against what the plan expected (``actual_amount_in - amount_in``). The percentage normalises this delta by the planned amount.【F:cryptopy/src/trading/triangular_arbitrage/executor.py†L545-L562】
* **output** performs the same comparison for the leg's *destination* currency (``actual_amount_out - amount_out``).【F:cryptopy/src/trading/triangular_arbitrage/executor.py†L545-L562】

Because both the spend and proceeds are calculated from live exchange data, the two values can diverge independently. Market orders walk the book: when the effective price (VWAP) shifts relative to the plan, the executor may need to spend more quote to reach the available depth and the fills may return more base than forecast (or vice versa). In the example above the buy leg hit slightly worse prices than simulated, consuming 0.4 % more USD, while the deeper fill also yielded 0.2004 % more USDC after fees.

Both deltas therefore describe how the live trade differed from the simulated trade—not a trade-off where one is fixed. Rounding to the exchange's precision and realised fee deductions are also applied to the *actual* amounts, so they can move in the same direction even when the ratio (effective price) stays close to the plan.【F:cryptopy/src/trading/triangular_arbitrage/slippage.py†L214-L320】

## Route summary

``Route execution slippage: final -0.34109896 (-0.3407%) without fees -0.54442599 (-0.5384%) profit -0.2291%``

This aggregates the per-leg deviations by comparing the realised ending balances to the simulated ones. Positive numbers mean you ended with more than planned; negative values indicate shortfall.【F:cryptopy/src/trading/triangular_arbitrage/executor.py†L564-L585】

Understanding both levels helps diagnose whether slippage is concentrated in a single leg or distributed across the cycle.
