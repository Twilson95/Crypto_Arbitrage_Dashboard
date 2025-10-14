"""Command line entry point for the triangular arbitrage toolkit."""
from __future__ import annotations

import argparse
import asyncio
import logging
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import yaml

from cryptopy.src.trading.triangular_arbitrage import (
    ExchangeConnection,
    InsufficientLiquidityError,
    OrderBookSnapshot,
    TriangularArbitrageCalculator,
    TriangularArbitrageExecutor,
    TriangularRoute,
)


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Runtime defaults
# ---------------------------------------------------------------------------
# These flags can be adjusted directly in the source file to quickly change the
# runner behaviour without editing the CLI invocation. Command-line arguments
# still take precedence when explicitly supplied.
EXCHANGE_DEFAULT = "kraken"
ENABLE_EXECUTION_DEFAULT = False
LIVE_TRADING_DEFAULT = False
USE_TESTNET_DEFAULT = True
DISABLE_WEBSOCKET_DEFAULT = False
LOG_LEVEL_DEFAULT = "INFO"

# Optional configuration file used to source API credentials when present.
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[3] / "config.yaml"
CONFIG_SECTION_BY_EXCHANGE = {
    "kraken": "kraken_websocket",
}


def generate_triangular_routes(markets: Dict[str, Dict[str, object]]) -> List[TriangularRoute]:
    """Construct all distinct three-leg triangular routes from exchange markets."""

    currency_graph: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    for symbol, metadata in markets.items():
        if not isinstance(metadata, dict):
            continue
        if metadata.get("active") is False:
            continue
        base = metadata.get("base")
        quote = metadata.get("quote")
        if not base or not quote:
            continue
        currency_graph[str(quote)].append((symbol, str(base)))
        currency_graph[str(base)].append((symbol, str(quote)))

    routes: List[TriangularRoute] = []
    seen: set[Tuple[str, Tuple[str, str, str]]] = set()

    for start_currency, first_edges in currency_graph.items():
        for symbol_a, currency_b in first_edges:
            for symbol_b, currency_c in currency_graph.get(currency_b, []):
                if symbol_b == symbol_a:
                    continue
                for symbol_c, currency_d in currency_graph.get(currency_c, []):
                    if symbol_c in (symbol_a, symbol_b):
                        continue
                    if currency_d != start_currency:
                        continue
                    route_symbols = (symbol_a, symbol_b, symbol_c)
                    key = (start_currency, route_symbols)
                    if key in seen:
                        continue
                    seen.add(key)
                    routes.append(TriangularRoute(route_symbols, start_currency))

    routes.sort(key=lambda route: (route.starting_currency, route.symbols))
    return routes


def compute_route_log_cost(
    route: TriangularRoute,
    order_books: Dict[str, OrderBookSnapshot],
    markets: Dict[str, Dict[str, object]],
    fee_lookup: Dict[str, float],
) -> Optional[float]:
    """Return the logarithmic cost of a route; negative implies arbitrage potential."""

    current_currency = route.starting_currency
    log_sum = 0.0

    for symbol in route.symbols:
        market = markets.get(symbol)
        order_book = order_books.get(symbol)
        if market is None or order_book is None:
            return None

        base = market.get("base")
        quote = market.get("quote")
        if not base or not quote:
            return None

        fee = fee_lookup.get(symbol, 0.0)
        fee_factor = 1.0 - float(fee)
        if fee_factor <= 0:
            return None

        if current_currency == quote:
            best_level = order_book.best_ask()
            if not best_level:
                return None
            price = float(best_level[0])
            if price <= 0:
                return None
            cost = price / fee_factor
            next_currency = str(base)
        elif current_currency == base:
            best_level = order_book.best_bid()
            if not best_level:
                return None
            price = float(best_level[0])
            if price <= 0:
                return None
            cost = 1.0 / (price * fee_factor)
            next_currency = str(quote)
        else:
            return None

        log_sum += math.log(cost)
        current_currency = next_currency

    if current_currency != route.starting_currency:
        return None

    return log_sum


def select_routes_with_negative_log_sum(
    routes: Sequence[TriangularRoute],
    order_books: Dict[str, OrderBookSnapshot],
    markets: Dict[str, Dict[str, object]],
    fee_lookup: Dict[str, float],
) -> List[TriangularRoute]:
    """Return only the routes whose cumulative log cost is negative."""

    selected: List[TriangularRoute] = []
    for route in routes:
        log_cost = compute_route_log_cost(route, order_books, markets, fee_lookup)
        if log_cost is None:
            continue
        if log_cost < 0:
            selected.append(route)
    return selected


@dataclass
class OpportunityExecution:
    """Records the most recent execution to avoid duplicate orders."""

    route: TriangularRoute
    profit_signature: Tuple[float, float]


def load_credentials_from_config(exchange: str, config_path: Optional[str]) -> Dict[str, str]:
    """Load API credentials for ``exchange`` from a YAML configuration file."""

    raw_path = config_path if config_path else DEFAULT_CONFIG_PATH
    try:
        config_file = Path(raw_path).expanduser()
    except TypeError:
        logger.warning("Invalid config path %r supplied; skipping credential load.", raw_path)
        return {}

    if isinstance(config_file, str):  # pragma: no cover - defensive for unexpected types
        config_file = Path(config_file)

    if not config_file.exists():
        return {}

    try:
        data = yaml.safe_load(config_file.read_text()) or {}
    except FileNotFoundError:  # pragma: no cover - race condition guard
        return {}
    except yaml.YAMLError:  # pragma: no cover - malformed config
        logger.warning("Unable to parse credentials from %s", config_file)
        return {}

    section_name = CONFIG_SECTION_BY_EXCHANGE.get(exchange.lower())
    if not section_name:
        return {}

    section = data.get(section_name) or {}
    credentials: Dict[str, str] = {}
    api_key = section.get("api_key")
    api_secret = section.get("api_secret")
    if api_key:
        credentials["apiKey"] = api_key
    if api_secret:
        credentials["secret"] = api_secret
    return credentials


async def watch_order_books(
    exchange: ExchangeConnection,
    symbols: Sequence[str],
    *,
    limit: int,
    poll_interval: float,
    cache: Dict[str, OrderBookSnapshot],
    wake_event: asyncio.Event,
    stop_event: asyncio.Event,
) -> None:
    """Continuously populate ``cache`` with the latest order book snapshots."""

    async def _watch_symbol(symbol: str) -> None:
        try:
            async for snapshot in exchange.watch_order_book(
                symbol,
                limit=limit,
                poll_interval=poll_interval,
            ):
                cache[symbol] = snapshot
                wake_event.set()
                if stop_event.is_set():
                    break
        except asyncio.CancelledError:
            raise
        except Exception:  # pragma: no cover - network failure path
            logger.exception("Order book watcher for %s stopped unexpectedly", symbol)
            stop_event.set()

    tasks = [asyncio.create_task(_watch_symbol(symbol), name=f"watch-{symbol}") for symbol in symbols]

    try:
        await stop_event.wait()
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


async def evaluate_and_execute(
    calculator: TriangularArbitrageCalculator,
    executor: Optional[TriangularArbitrageExecutor],
    routes: Sequence[TriangularRoute],
    *,
    order_books: Dict[str, OrderBookSnapshot],
    markets: Dict[str, Dict[str, object]],
    fee_lookup: Dict[str, float],
    starting_amount: float,
    min_profit_percentage: float,
    max_route_length: Optional[int],
    evaluation_interval: float,
    enable_execution: bool,
    wake_event: asyncio.Event,
    stop_event: asyncio.Event,
) -> None:
    """Evaluate cached order books and execute profitable opportunities."""

    last_execution: Optional[OpportunityExecution] = None

    while not stop_event.is_set():
        try:
            await asyncio.wait_for(wake_event.wait(), timeout=evaluation_interval)
        except asyncio.TimeoutError:
            pass
        wake_event.clear()

        if stop_event.is_set():
            break
        if not order_books:
            continue

        candidate_routes = select_routes_with_negative_log_sum(
            routes,
            order_books,
            markets,
            fee_lookup,
        )
        if not candidate_routes:
            logger.debug("No routes satisfied the negative log-sum arbitrage condition.")
            continue

        try:
            opportunities = calculator.find_profitable_routes(
                candidate_routes,
                order_books,
                starting_amount=starting_amount,
                min_profit_percentage=min_profit_percentage,
                max_route_length=max_route_length,
            )
        except (InsufficientLiquidityError, KeyError, ValueError) as exc:
            logger.debug("Skipping evaluation due to error: %s", exc)
            continue

        if not opportunities:
            continue

        best = opportunities[0]
        logger.info(
            "Best opportunity: route=%s profit=%.6f (%.4f%%)",
            " -> ".join(best.route.symbols),
            best.profit,
            best.profit_percentage,
        )

        profit_signature = (round(best.final_amount, 8), round(best.profit_percentage, 4))
        if last_execution and last_execution.route == best.route and last_execution.profit_signature == profit_signature:
            logger.debug("Opportunity already executed recently; skipping duplicate execution")
            continue

        if not enable_execution or executor is None:
            logger.info(
                "Execution disabled. Opportunity would yield %.6f (%.4f%%) if executed.",
                best.profit,
                best.profit_percentage,
            )
            last_execution = OpportunityExecution(best.route, profit_signature)
            continue

        try:
            orders = await executor.execute_async(best)
        except ValueError as exc:
            logger.debug("Skipping execution: %s", exc)
            continue

        last_execution = OpportunityExecution(best.route, profit_signature)
        if orders:
            logger.info("Placed %d order(s) for opportunity.", len(orders))


async def run_from_args(args: argparse.Namespace) -> None:
    exchange_name = args.exchange or EXCHANGE_DEFAULT
    if not exchange_name:
        raise SystemExit("An exchange must be specified either inline or via --exchange.")

    if args.live_trading and not args.enable_execution:
        raise SystemExit("--live-trading requires --enable-execution to be set.")

    credentials = load_credentials_from_config(exchange_name, args.config)
    if args.api_key:
        credentials["apiKey"] = args.api_key
    if args.secret:
        credentials["secret"] = args.secret
    if args.password:
        credentials["password"] = args.password

    exchange = ExchangeConnection(
        exchange_name,
        credentials=credentials or None,
        use_testnet=args.use_testnet,
        enable_websocket=not args.disable_websocket,
        make_trades=args.live_trading,
    )

    if args.use_testnet and not exchange.sandbox_supported:
        logger.warning(
            "Sandbox mode is not available for %s via ccxt; requests will hit the production API.",
            exchange_name,
        )

    markets = exchange.get_markets()
    routes = generate_triangular_routes(markets)
    if not routes:
        raise SystemExit(f"No triangular routes discovered for {exchange_name}.")
    logger.info("Discovered %d triangular routes on %s", len(routes), exchange_name)

    calculator = TriangularArbitrageCalculator(
        exchange,
        slippage_buffer=args.slippage_buffer,
    )
    executor: Optional[TriangularArbitrageExecutor] = None
    if args.enable_execution:
        executor = TriangularArbitrageExecutor(
            exchange,
            dry_run=not args.live_trading,
            trade_log_path=args.trade_log,
        )

    symbols = sorted({symbol for route in routes for symbol in route.symbols})
    fee_lookup = {symbol: float(exchange.get_taker_fee(symbol)) for symbol in symbols}
    order_books: Dict[str, OrderBookSnapshot] = {}
    wake_event = asyncio.Event()
    stop_event = asyncio.Event()

    if symbols:
        for symbol in symbols:
            try:
                snapshot = await asyncio.to_thread(
                    exchange.get_order_book,
                    symbol,
                    limit=args.order_book_depth,
                )
            except Exception:
                logger.debug("Initial order book fetch failed for %s", symbol, exc_info=True)
            else:
                order_books[symbol] = snapshot
        if order_books:
            wake_event.set()

    watcher = asyncio.create_task(
        watch_order_books(
            exchange,
            symbols,
            limit=args.order_book_depth,
            poll_interval=args.poll_interval,
            cache=order_books,
            wake_event=wake_event,
            stop_event=stop_event,
        ),
        name="order-book-watcher",
    )

    evaluator = asyncio.create_task(
        evaluate_and_execute(
            calculator,
            executor,
            routes,
            order_books=order_books,
            markets=markets,
            fee_lookup=fee_lookup,
            starting_amount=args.starting_amount,
            min_profit_percentage=args.min_profit_percentage,
            max_route_length=args.max_route_length,
            evaluation_interval=args.evaluation_interval,
            enable_execution=args.enable_execution,
            wake_event=wake_event,
            stop_event=stop_event,
        ),
        name="arbitrage-evaluator",
    )

    try:
        await asyncio.gather(watcher, evaluator)
    except asyncio.CancelledError:  # pragma: no cover - cancellation path
        pass
    except KeyboardInterrupt:
        logger.info("Interrupted by user. Shutting down...")
    finally:
        stop_event.set()
        watcher.cancel()
        evaluator.cancel()
        await asyncio.gather(watcher, evaluator, return_exceptions=True)
        await exchange.aclose()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Monitor and execute triangular arbitrage opportunities in real time.",
    )
    parser.add_argument(
        "--exchange",
        default=EXCHANGE_DEFAULT,
        help="Name of the exchange supported by ccxt (e.g. binance, kraken).",
    )
    parser.add_argument(
        "--starting-amount",
        type=float,
        default=100.0,
        help="Amount of the starting currency used for simulations and orders.",
    )
    parser.add_argument(
        "--min-profit-percentage",
        type=float,
        default=0.0,
        help="Minimum profit percentage required before executing an opportunity.",
    )
    parser.add_argument(
        "--max-route-length",
        type=int,
        default=None,
        help="Optional maximum number of legs per route to evaluate.",
    )
    parser.add_argument(
        "--slippage-buffer",
        type=float,
        default=0.0,
        help="Fractional buffer applied after fees to account for slippage (e.g. 0.01 for 1%%).",
    )
    parser.add_argument(
        "--order-book-depth",
        type=int,
        default=10,
        help="Number of price levels to request for each order book.",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=2.0,
        help="Fallback polling interval (seconds) when websockets are unavailable.",
    )
    parser.add_argument(
        "--evaluation-interval",
        type=float,
        default=1.0,
        help="Minimum interval in seconds between opportunity evaluations.",
    )
    parser.add_argument(
        "--trade-log",
        default=None,
        help="Optional path to a CSV file where executed legs will be appended.",
    )
    parser.add_argument(
        "--api-key", dest="api_key", default=None, help="API key for authenticated requests.")
    parser.add_argument(
        "--secret", dest="secret", default=None, help="API secret for authenticated requests."
    )
    parser.add_argument(
        "--password", dest="password", default=None, help="Exchange password if required."
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional path to a YAML file containing API credentials.",
    )
    parser.add_argument(
        "--use-testnet",
        action=argparse.BooleanOptionalAction,
        default=USE_TESTNET_DEFAULT,
        help="Enable or disable the exchange sandbox/testnet when supported.",
    )
    parser.add_argument(
        "--disable-websocket",
        action=argparse.BooleanOptionalAction,
        default=DISABLE_WEBSOCKET_DEFAULT,
        help="Disable websocket usage and rely on REST polling for order books.",
    )
    parser.add_argument(
        "--enable-execution",
        action=argparse.BooleanOptionalAction,
        default=ENABLE_EXECUTION_DEFAULT,
        help="Allow the executor to place simulated or live orders.",
    )
    parser.add_argument(
        "--live-trading",
        action=argparse.BooleanOptionalAction,
        default=LIVE_TRADING_DEFAULT,
        help="Execute real trades instead of running in dry-run mode.",
    )
    parser.add_argument(
        "--log-level",
        default=LOG_LEVEL_DEFAULT,
        help="Configure the logging level (e.g. DEBUG, INFO).",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    default_level = getattr(logging, LOG_LEVEL_DEFAULT.upper(), logging.INFO)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), default_level))
    try:
        asyncio.run(run_from_args(args))
    except KeyboardInterrupt:  # pragma: no cover - outer signal handler
        logger.info("Interrupted by user. Goodbye!")


if __name__ == "__main__":  # pragma: no cover - script entry point
    main()

