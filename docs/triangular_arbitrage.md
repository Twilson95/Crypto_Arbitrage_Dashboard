# Triangular Arbitrage Runner

The `run_triangular_arbitrage.py` script monitors real-time order books and evaluates configured triangular routes. By default it only reports opportunities and does **not** attempt to place any orders.

## Usage

```
python -m cryptopy.scripts.trading.run_triangular_arbitrage \
  --exchange kraken \
  --route XBT/USD>ETH/XBT>ETH/USD:USD \
  --starting-amount 100 \
  --min-profit-percentage 0.2
```

Key options:

- `--enable-execution` toggles order placement. Omit the flag to keep execution disabled and simply review the logs.
- `--live-trading` switches from simulated test orders to real exchange orders. It requires `--enable-execution`.
- `--use-testnet` enables the sandbox/testnet for exchanges that provide one.
- `--trade-log` stores each simulated or live leg in a CSV file for auditing.

## Kraken testing

Kraken offers a dedicated **Spot Trading Sandbox** that mirrors the production API with a separate base URL (`https://api.sandbox.kraken.com`). You can sign up for a sandbox account at <https://support.kraken.com/hc/en-us/articles/360022839011-Kraken-Spot-Trading-API-Sandbox> and generate API credentials that are safe for testing. When using ccxt, pass the sandbox API key/secret and enable sandbox mode either via the library or with the `--use-testnet` flag provided by this project.

Keep in mind that some Kraken features—particularly funding and staking—are not available in the sandbox, so final integration tests should still be performed with small real orders once you are confident in the strategy.
