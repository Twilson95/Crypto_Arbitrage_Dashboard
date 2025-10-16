# Triangular Arbitrage Runner

The `run_triangular_arbitrage.py` script monitors real-time ticker prices and evaluates triangular routes that it discovers automatically from the exchange markets. By default it only reports opportunities and does **not** attempt to place any orders.

At the top of the script you will find a handful of boolean defaults (for example `ENABLE_EXECUTION_DEFAULT`, `LIVE_TRADING_DEFAULT`, and `USE_TESTNET_DEFAULT`) along with `EXCHANGE_DEFAULT`. New helpers such as `STARTING_CURRENCY_DEFAULT`, `MAX_ROUTE_LENGTH_DEFAULT`, `MAX_EXECUTIONS_DEFAULT`, `EVALUATION_INTERVAL_DEFAULT`, and `TRADE_LOG_PATH_DEFAULT` make it easy to pin the seed currency to USD, control the longest route to evaluate, stop automatically after a certain number of executed opportunities, choose how infrequently the periodic safety check should run, and decide where executed legs should be logged. Toggle these values to quickly change the runner's behaviour without modifying your command line each time. CLI arguments still override the inline defaults when explicitly supplied.

## Usage

```
python -m cryptopy.scripts.trading.run_triangular_arbitrage \
  --starting-currency USD \
  --starting-amount 100 \
  --min-profit-percentage 0.2
```

Key options:

- `--starting-currency` restricts discovery to routes that both begin and end in the chosen currency (USD by default).
- `--enable-execution` toggles order placement. Omit the flag to keep execution disabled and simply review the logs.
- `--live-trading` switches from simulated test orders to real exchange orders. It requires `--enable-execution`.
- `--use-testnet` enables the sandbox/testnet for exchanges that provide one.
- `--max-executions` stops the runner after it successfully executes the requested number of opportunities so you can review the results before continuing.
- `--trade-log` stores each simulated or live leg in a CSV file for auditing (defaults to the `TRADE_LOG_PATH_DEFAULT` location if left unspecified).
- `--websocket-timeout` bounds how long the watcher will wait for a streamed ticker update before temporarily polling the REST API instead.
- `--price-refresh-interval` sets the target cadence (per symbol) for REST ticker refreshes so rate limits stay under control even on large markets.
- `--price-max-age` determines how long cached price snapshots remain valid before the evaluator ignores them.
- `--config` can point at a YAML file (defaults to `config/exchange_config.yaml` in the project root) that contains credentials. For Kraken, the loader expects an entry in the form:

  ```yaml
  kraken_websocket:
    api_key: YOUR_KEY
    api_secret: YOUR_SECRET
  ```

## Automatic route discovery

When the runner starts it queries the exchange metadata to build every distinct three-leg triangular route that begins with the configured starting currency (USD by default). It then primes each price snapshot and filters the routes using the arbitrage-friendly condition:

\[
\ln(p_1 f_1) + \ln(p_2 f_2) + \ln(p_3 f_3) < 0,
\]

where `p` is the effective conversion price for a leg (buy uses the best ask while sell uses the inverse of the best bid) and `f` reflects the taker fee. Routes that satisfy this negative log-sum constraint are passed to the simulator for a full best-bid/ask evaluation, ensuring that only promising opportunities are explored without requiring manual route definitions.

Many derivative venues, such as BitMEX, expose symbols like `ETH/USD:BTC-251226` where the currency after the colon represents the settlement asset for a quanto or futures contract. These instruments cannot be traded with spot USD balances, so the runner automatically filters them (along with inactive and non-spot markets) before attempting route discovery. Simply trimming the suffix (e.g. using `ETH/USD`) will not work—the exchange will either reject the symbol or settle the trade in the wrong currency, breaking the USD cashflow assumptions used by the simulator. The log entry starting with `Filtered ... market(s)` summarises how many symbols were skipped and why. If you notice recurring `Currency mismatch` errors, confirm that the exchange offers true spot markets for your starting currency or switch to a venue with native USD pairs.

## Sandbox and exchange support

Some ccxt exchanges expose official sandbox or testnet environments that can be toggled through `set_sandbox_mode(True)`. Examples include **Binance** (spot and futures testnet), **BitMEX**, **KuCoin**, and **OKX**. These are ideal for rehearsing end-to-end order placement without risking funds—set `USE_TESTNET_DEFAULT = True` or pass `--use-testnet` on the CLI to request the sandbox where it is available.

Kraken operates a **Spot Trading Sandbox** at <https://support.kraken.com/hc/en-us/articles/360022839011-Kraken-Spot-Trading-API-Sandbox>, but the ccxt library does not currently expose that sandbox through `set_sandbox_mode`. When you connect to Kraken through this toolkit, ccxt reports that sandbox mode is unsupported and requests fall back to the production API. The runner now detects this situation and automatically keeps the trading client in sandbox mode (when available) while sourcing public market data from the production endpoints so that price updates continue to flow. If you require a fully sandbox-backed workflow, consider testing with one of the exchanges above. For Kraken, plan to validate with small real-money trades once you are satisfied that the strategy behaves correctly.
Each evaluation cycle logs the trigger that woke it (market data refresh, periodic fallback, or a post-trade confirmation), the start time, and how long it took to review the available routes. The log message also reports how many of the discovered routes currently have complete price data and how many of those satisfy the negative log-sum filter, making it clear why a subset of routes advance to the profitability simulation. This makes it easy to confirm that the runner only reacts when something meaningful happens while still providing an infrequent periodic backstop.

Live market triggers now come from lightweight ticker streams (or REST price polling when websockets are absent) while a separate maintainer refreshes cached tickers sequentially at the configured cadence. This keeps bid/ask snapshots fresh for the simulator without hammering the exchange with full depth subscriptions. When websockets for tickers are unavailable or time out, the watcher automatically falls back to REST polling and resumes streaming once updates begin to flow again.

## Authenticated order smoke testing

Use `python -m cryptopy.scripts.trading.test_authenticated_order` when you want to
confirm that your API credentials work for private endpoints before letting the
full runner place trades. The helper loads the same `config/exchange_config.yaml`
credentials as the arbitrage script, verifies access to private endpoints such as
`fetch_balance`, optionally lists open orders, and can submit a minimal market
order when you pass `--execute-order --symbol BTC/USD --amount 0.001`. Omit
`--execute-order` to review the payload without placing the trade. This makes it
straightforward to troubleshoot authentication issues (for example, missing
Kraken API keys) without running the entire arbitrage workflow.
