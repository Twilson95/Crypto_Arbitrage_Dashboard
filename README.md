# Crypto Arbitrage Dashboard

This repository contains the trading dashboard and live triangular arbitrage runner
for monitoring and executing opportunities across supported exchanges such as Kraken.

## Local setup

1. Create and activate a Python 3.11 virtual environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```
3. Copy `config/exchange_config.example.yaml` to `config/exchange_config.yaml` and populate
   your API credentials. The runner automatically searches for this file when
   authenticating.
4. Launch the arbitrage runner:
   ```bash
   python -m cryptopy.scripts.trading.run_triangular_arbitrage --help
   ```

## Docker deployment

The provided `Dockerfile` packages the runner together with its Python dependencies.

```bash
docker build -t triangular-arb .
```

Before running the container, create `config/exchange_config.yaml` (copy the example
file and fill in credentials) so it can be bundled into the image or mounted at
runtime.

Example run with the config mounted read-only and benchmarking enabled every 5 seconds:

```bash
docker run --rm \
  -v "$PWD/config/exchange_config.yaml:/app/config/exchange_config.yaml:ro" \
  triangular-arb \
  --enable-benchmarking \
  --benchmark-interval 5 \
  --starting-currency USD
```

You can override the default command to supply any of the runner's CLI arguments.
The container entrypoint executes `python -m cryptopy.scripts.trading.run_triangular_arbitrage`.

## Benchmarking

When launched with `--enable-benchmarking`, the runner logs CPU utilisation, RSS
memory, and thread statistics at shutdown. Use these summaries to right-size the
Azure VM (e.g., start with a `Standard_D4s_v5` and scale up if the average CPU
load approaches saturation).

## Security note

Never commit real API credentials. Keep `config/exchange_config.yaml` out of
source control and distribute it securely when building or deploying images.
