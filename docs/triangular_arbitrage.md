# Triangular Arbitrage Runner

The `run_triangular_arbitrage.py` script monitors real-time order books and evaluates triangular routes that it discovers automatically from the exchange markets. By default it only reports opportunities and does **not** attempt to place any orders.

At the top of the script you will find a handful of boolean defaults (for example `ENABLE_EXECUTION_DEFAULT`, `LIVE_TRADING_DEFAULT`, and `USE_TESTNET_DEFAULT`) along with `EXCHANGE_DEFAULT`. Toggle these flags to quickly change the runner's behaviour without modifying your command line each time. CLI arguments still override the inline defaults when explicitly supplied.

## Usage

```
python -m cryptopy.scripts.trading.run_triangular_arbitrage \
  --starting-amount 100 \
  --min-profit-percentage 0.2
```

Key options:

- `--enable-execution` toggles order placement. Omit the flag to keep execution disabled and simply review the logs.
- `--live-trading` switches from simulated test orders to real exchange orders. It requires `--enable-execution`.
- `--use-testnet` enables the sandbox/testnet for exchanges that provide one.
- `--trade-log` stores each simulated or live leg in a CSV file for auditing.
- `--config` can point at a YAML file (defaults to `config.yaml` in the project root) that contains credentials. For Kraken, the loader expects an entry in the form:

  ```yaml
  kraken_websocket:
    api_key: YOUR_KEY
    api_secret: YOUR_SECRET
  ```

## Automatic route discovery

When the runner starts it queries the exchange metadata to build every distinct three-leg triangular route. It then primes each order book and filters the routes using the arbitrage-friendly condition:

\[
\ln(p_1 f_1) + \ln(p_2 f_2) + \ln(p_3 f_3) < 0,
\]

where `p` is the effective conversion price for a leg (buy uses the best ask while sell uses the inverse of the best bid) and `f` reflects the taker fee. Routes that satisfy this negative log-sum constraint are passed to the simulator for a full depth-aware evaluation, ensuring that only promising opportunities are explored without requiring manual route definitions.

## Kraken testing

Kraken offers a dedicated **Spot Trading Sandbox** that mirrors the production API with a separate base URL (`https://api.sandbox.kraken.com`). You can sign up for a sandbox account at <https://support.kraken.com/hc/en-us/articles/360022839011-Kraken-Spot-Trading-API-Sandbox> and generate API credentials that are safe for testing. When using ccxt, pass the sandbox API key/secret and enable sandbox mode either via the library or with the `--use-testnet` flag provided by this project.

Keep in mind that some Kraken features—particularly funding and staking—are not available in the sandbox, so final integration tests should still be performed with small real orders once you are confident in the strategy.
