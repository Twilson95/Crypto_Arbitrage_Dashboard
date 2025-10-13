"""Command line entry point for the triangular arbitrage toolkit."""
from __future__ import annotations

import argparse
import asyncio
import logging
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

from cryptopy.src.trading.triangular_arbitrage import (
    ExchangeConnection,
    InsufficientLiquidityError,
    OrderBookSnapshot,
    TriangularArbitrageCalculator,
    TriangularArbitrageExecutor,
    TriangularRoute,
)


logger = logging.getLogger(__name__)


def parse_route(route_definition: str) -> TriangularRoute:
    """Parse a CLI route definition into a :class:`TriangularRoute` instance.

    The expected format is ``PAIR1>PAIR2>PAIR3:START_CURRENCY``.
    """

    try:
        symbols_part, starting_currency = route_definition.split(":", maxsplit=1)
    except ValueError as exc:  # pragma: no cover - defensive guard
        raise argparse.ArgumentTypeError(
            "Routes must follow the format 'PAIR1>PAIR2>PAIR3:START'."
        ) from exc
    symbols = tuple(symbol.strip().upper() for symbol in symbols_part.split(">") if symbol.strip())
    if len(symbols) < 3:
        raise argparse.ArgumentTypeError(
            "A triangular route must contain at least three trading pairs."
        )
    return TriangularRoute(symbols, starting_currency.upper())


@dataclass
class OpportunityExecution:
    """Records the most recent execution to avoid duplicate orders."""

    route: TriangularRoute
    profit_signature: Tuple[float, float]


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
    required_symbols: Sequence[str],
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
        if any(symbol not in order_books for symbol in required_symbols):
            continue

        try:
            opportunities = calculator.find_profitable_routes(
                routes,
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
    if args.live_trading and not args.enable_execution:
        raise SystemExit("--live-trading requires --enable-execution to be set.")

    credentials: Dict[str, str] = {}
    if args.api_key:
        credentials["apiKey"] = args.api_key
    if args.secret:
        credentials["secret"] = args.secret
    if args.password:
        credentials["password"] = args.password

    exchange = ExchangeConnection(
        args.exchange,
        credentials=credentials or None,
        use_testnet=args.use_testnet,
        enable_websocket=not args.disable_websocket,
        make_trades=args.live_trading,
    )

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

    symbols = sorted({symbol for route in args.routes for symbol in route.symbols})
    order_books: Dict[str, OrderBookSnapshot] = {}
    wake_event = asyncio.Event()
    stop_event = asyncio.Event()

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
            args.routes,
            order_books=order_books,
            required_symbols=symbols,
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
        required=True,
        help="Name of the exchange supported by ccxt (e.g. binance, kraken).",
    )
    parser.add_argument(
        "--route",
        dest="routes",
        action="append",
        type=parse_route,
        required=True,
        help="Triangular route definition formatted as PAIR1>PAIR2>PAIR3:START_CURRENCY. Repeat for multiple routes.",
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
        "--use-testnet",
        action="store_true",
        default=False,
        help="Enable the exchange sandbox/testnet when supported.",
    )
    parser.add_argument(
        "--disable-websocket",
        action="store_true",
        help="Disable websocket usage and rely on REST polling for order books.",
    )
    parser.add_argument(
        "--enable-execution",
        action="store_true",
        help="Allow the executor to place simulated or live orders.",
    )
    parser.add_argument(
        "--live-trading",
        action="store_true",
        help="Execute real trades instead of running in dry-run mode.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Configure the logging level (e.g. DEBUG, INFO).",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    try:
        asyncio.run(run_from_args(args))
    except KeyboardInterrupt:  # pragma: no cover - outer signal handler
        logger.info("Interrupted by user. Goodbye!")


if __name__ == "__main__":  # pragma: no cover - script entry point
    main()

