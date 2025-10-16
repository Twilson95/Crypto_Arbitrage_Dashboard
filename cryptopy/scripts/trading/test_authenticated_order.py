"""Utility script for validating authenticated ccxt calls and market orders."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from cryptopy.src.trading.triangular_arbitrage.exchange import (
    CcxtAuthenticationError,
    ExchangeConnection,
)

# Defaults mirror the triangular arbitrage runner so operators can reuse configs.
EXCHANGE_DEFAULT = "kraken"
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[3] / "config.yaml"
CONFIG_SECTION_BY_EXCHANGE = {
    "kraken": "kraken_websocket",
}


logger = logging.getLogger(__name__)


def load_credentials_from_config(exchange: str, config_path: Optional[str]) -> Dict[str, str]:
    """Load API credentials for ``exchange`` from ``config.yaml`` style files."""

    raw_path = config_path if config_path else DEFAULT_CONFIG_PATH
    try:
        config_file = Path(raw_path).expanduser()
    except TypeError:
        logger.warning(f"Invalid config path {raw_path!r} supplied; skipping credential load.")
        return {}

    if isinstance(config_file, str):  # pragma: no cover - defensive guard
        config_file = Path(config_file)

    if not config_file.exists():
        logger.debug(f"Config path {config_file} does not exist; skipping credential load")
        return {}

    try:
        data = yaml.safe_load(config_file.read_text()) or {}
    except FileNotFoundError:  # pragma: no cover - race condition guard
        return {}
    except yaml.YAMLError:  # pragma: no cover - malformed config
        logger.warning(f"Unable to parse credentials from {config_file}")
        return {}

    section_name = CONFIG_SECTION_BY_EXCHANGE.get(exchange.lower())
    if not section_name:
        return {}

    section: Dict[str, Any] = {}
    if isinstance(data, dict):
        for key, candidate in data.items():
            if isinstance(key, str) and key.lower() == section_name.lower():
                section = candidate or {}
                break

    if not isinstance(section, dict) or not section:
        logger.debug(f"No credentials found for {exchange} in {config_file}")
        return {}

    credentials: Dict[str, str] = {}
    for raw_key, raw_value in section.items():
        if raw_value in (None, ""):
            continue
        key = str(raw_key).strip().lower()
        if key in {"apikey", "api_key", "key"} and "apiKey" not in credentials:
            credentials["apiKey"] = str(raw_value)
        elif key in {"secret", "api_secret", "secretkey"} and "secret" not in credentials:
            credentials["secret"] = str(raw_value)
        elif key in {"password", "passphrase"} and "password" not in credentials:
            credentials["password"] = str(raw_value)

    if not credentials:
        logger.debug(f"No credentials found for {exchange} in {config_file}")
        return {}

    masked = ", ".join(
        f"{key}={'***' if key != 'password' else '***'}" for key in sorted(credentials)
    )
    logger.debug(f"Loaded credentials for {exchange} from {config_file}: {masked}")
    return credentials


def merge_credentials(
    *,
    config_credentials: Dict[str, str],
    cli_api_key: Optional[str],
    cli_secret: Optional[str],
    cli_password: Optional[str],
) -> Dict[str, str]:
    """Combine config and CLI credentials, with CLI values taking precedence."""

    credentials = dict(config_credentials)
    if cli_api_key:
        credentials["apiKey"] = cli_api_key
    if cli_secret:
        credentials["secret"] = cli_secret
    if cli_password:
        credentials["password"] = cli_password
    return credentials


def parse_order_params(raw_params: Optional[list[str]]) -> Dict[str, Any]:
    """Parse ``key=value`` CLI values into a ccxt params dictionary."""

    params: Dict[str, Any] = {}
    if not raw_params:
        return params

    for item in raw_params:
        if "=" not in item:
            raise ValueError(f"Order parameter {item!r} is not in key=value format")
        key, value = item.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            raise ValueError(f"Order parameter {item!r} is missing a key")
        # Attempt to decode JSON for structured values; fallback to raw strings.
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            decoded = value
        params[key] = decoded
    return params


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate authenticated ccxt interactions by fetching balances, "
            "open orders, and optionally submitting a market order."
        )
    )
    parser.add_argument(
        "--exchange",
        default=EXCHANGE_DEFAULT,
        help=f"Exchange identifier supported by ccxt (default: {EXCHANGE_DEFAULT})",
    )
    parser.add_argument(
        "--config",
        default=None,
        help=f"Path to config.yaml containing credentials (default: {DEFAULT_CONFIG_PATH})",
    )
    parser.add_argument("--api-key", dest="api_key", default=None, help="Override API key")
    parser.add_argument("--secret", dest="secret", default=None, help="Override API secret")
    parser.add_argument(
        "--password",
        dest="password",
        default=None,
        help="Optional API password/passphrase override",
    )
    parser.add_argument(
        "--use-testnet",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Toggle ccxt sandbox mode when supported",
    )
    parser.add_argument(
        "--check-balance",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Fetch account balances to verify private API access",
    )
    parser.add_argument(
        "--check-open-orders",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Fetch open orders to validate authenticated requests",
    )
    parser.add_argument(
        "--symbol",
        default=None,
        help="Symbol to use when placing a market order (e.g. BTC/USD)",
    )
    parser.add_argument(
        "--side",
        choices=("buy", "sell"),
        default="buy",
        help="Market order side when submitting an order",
    )
    parser.add_argument(
        "--amount",
        type=float,
        default=None,
        help="Order size to submit when testing market order creation",
    )
    parser.add_argument(
        "--order-param",
        action="append",
        dest="order_params",
        default=None,
        help="Additional order parameters in key=value format (repeatable)",
    )
    parser.add_argument(
        "--execute-order",
        default=False,
        action=argparse.BooleanOptionalAction,
        help=(
            "Actually submit the market order to the exchange. By default the script "
            "performs a dry run and only prints the request payload."
        ),
    )
    parser.add_argument(
        "--log-level",
        default="DEBUG",
        help="Python logging level (default: DEBUG)",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.DEBUG))

    logger.debug(f"Loading credentials for {args.exchange} using config {args.config or DEFAULT_CONFIG_PATH}")
    config_credentials = load_credentials_from_config(args.exchange, args.config)
    credentials = merge_credentials(
        config_credentials=config_credentials,
        cli_api_key=args.api_key,
        cli_secret=args.secret,
        cli_password=args.password,
    )

    if not credentials:
        logger.warning(
            "No credentials supplied; only public endpoints will be available. "
            "Authenticated tests and order submission will fail."
        )

    try:
        exchange = ExchangeConnection(
            args.exchange,
            credentials=credentials,
            use_testnet=args.use_testnet,
            enable_websocket=False,
            make_trades=args.execute_order,
        )
    except CcxtAuthenticationError as exc:
        logger.error(f"Failed to authenticate with {args.exchange}: {exc}")
        raise

    logger.info(
        f"Initialised ExchangeConnection for {args.exchange} (use_testnet={args.use_testnet}, "
        f"credentials={'provided' if credentials else 'missing'})"
    )

    if args.check_balance:
        fetch_balance = getattr(exchange.rest_client, "fetch_balance", None)
        if callable(fetch_balance):
            try:
                balance = fetch_balance()
            except CcxtAuthenticationError as exc:
                logger.error(f"Authentication failed when fetching balances: {exc}")
            except Exception as exc:  # pragma: no cover - network dependent
                logger.error(f"Fetching balances failed with {exc.__class__.__name__}: {exc}")
            else:
                free = balance.get("free") if isinstance(balance, dict) else None
                summary = ", ".join(sorted(free)) if isinstance(free, dict) else "unknown"
                logger.info(f"Fetched balances; currencies with 'free' amounts: {summary}")
        else:
            logger.info("Exchange does not support fetch_balance; skipping balance check")

    if args.check_open_orders:
        fetch_open_orders = getattr(exchange.rest_client, "fetch_open_orders", None)
        if callable(fetch_open_orders):
            try:
                open_orders = fetch_open_orders(args.symbol)
            except CcxtAuthenticationError as exc:
                logger.error(f"Authentication failed when fetching open orders: {exc}")
            except Exception as exc:  # pragma: no cover - network dependent
                logger.error(f"Fetching open orders failed with {exc.__class__.__name__}: {exc}")
            else:
                count = len(open_orders) if isinstance(open_orders, list) else "unknown"
                logger.info(f"Fetched open orders for {args.symbol or 'all symbols'}; count={count}")
        else:
            logger.info("Exchange does not support fetch_open_orders; skipping order check")

    if args.symbol and args.amount:
        try:
            params = parse_order_params(args.order_params)
        except ValueError as exc:
            logger.error(str(exc))
            raise SystemExit(2) from exc

        try:
            result = exchange.create_market_order(
                args.symbol,
                args.side,
                args.amount,
                params=params,
                test_order=not args.execute_order,
            )
        except CcxtAuthenticationError as exc:
            logger.error(f"Authentication failed when creating market order: {exc}")
            raise
        except Exception as exc:  # pragma: no cover - network dependent
            logger.error(f"Order creation failed with {exc.__class__.__name__}: {exc}")
            raise
        else:
            if args.execute_order:
                logger.info(f"Submitted market order on {args.exchange}: {result}")
            else:
                logger.info(
                    "Dry-run market order payload (submit with --execute-order to place it): %s",
                    result,
                )
    else:
        if args.symbol or args.amount:
            logger.warning(
                "Both --symbol and --amount are required to test market order creation; skipping."
            )
        else:
            logger.debug("No market order parameters provided; skipping order creation test")


if __name__ == "__main__":
    main()
