"""Realtime slippage estimation helpers backed by websocket order books."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import AsyncIterator, Dict, Mapping, Optional, Tuple

from .exchange import ExchangeConnection
from .exceptions import InsufficientLiquidityError
from .models import OrderBookSnapshot, TriangularOpportunity
from .slippage import PrecisionAdapter, SlippageSimulation, simulate_opportunity_with_order_books


@dataclass(frozen=True)
class OpportunitySlippageUpdate:
    """Container emitted when recalculating slippage for an opportunity."""

    simulation: SlippageSimulation
    order_books: Mapping[str, OrderBookSnapshot]


async def stream_opportunity_slippage(
    exchange: ExchangeConnection,
    opportunity: TriangularOpportunity,
    *,
    starting_amount: Optional[float] = None,
    depth: int = 10,
    websocket_timeout: Optional[float] = 10.0,
    throttle: Optional[float] = None,
    require_websocket: bool = True,
) -> AsyncIterator[OpportunitySlippageUpdate]:
    """Yield live slippage estimates for ``opportunity`` using websocket order books.

    The coroutine starts a background watcher for every symbol present in
    ``opportunity``. Whenever one of those order books updates, a new
    :class:`OpportunitySlippageUpdate` is produced after replaying the
    opportunity against the refreshed order book snapshots.

    Parameters
    ----------
    exchange:
        Exchange connection used to access ``watch_order_book`` and precision
        helpers.
    opportunity:
        Planned triangular trade to revalue against the streamed order books.
    starting_amount:
        Amount of the starting currency to simulate. When omitted the
        opportunity's ``starting_amount`` is used.
    depth:
        Order book depth requested from the exchange websocket.
    websocket_timeout:
        Timeout passed to :meth:`ExchangeConnection.watch_order_book`.
    throttle:
        Minimum number of seconds between emitted updates. When set to ``None``
        (the default) every order book update results in a recalculation.
    require_websocket:
        When ``True`` (the default) an exception is raised if websocket data is
        unavailable, preventing a fallback to REST polling.

    Yields
    ------
    OpportunitySlippageUpdate
        Contains the recomputed simulation and the latest order books used for
        the calculation.

    Raises
    ------
    InsufficientLiquidityError
        If the streamed order book does not contain enough depth for the
        planned trade size. Consumers can catch the exception and continue the
        stream if desired.
    Exception
        Any unexpected exception raised by ``watch_order_book`` propagates to
        the caller so it can decide how to recover.
    """

    if starting_amount is None:
        starting_amount = opportunity.starting_amount

    if starting_amount <= 0:
        raise ValueError("starting_amount must be positive")

    precision = PrecisionAdapter(
        amount_to_precision=exchange.amount_to_precision,
        cost_to_precision=exchange.cost_to_precision,
    )

    symbols: Tuple[str, ...] = tuple(dict.fromkeys(opportunity.route.symbols))
    if not symbols:
        raise ValueError("opportunity.route.symbols must contain at least one symbol")

    latest_books: Dict[str, OrderBookSnapshot] = {}
    loop = asyncio.get_running_loop()
    last_emit = 0.0

    streams: Dict[str, AsyncIterator[OrderBookSnapshot]] = {}
    task_to_symbol: Dict[asyncio.Task, str] = {}

    for symbol in symbols:
        stream = exchange.watch_order_book(
            symbol,
            limit=depth,
            websocket_timeout=websocket_timeout,
            require_websocket=require_websocket,
        )
        streams[symbol] = stream
        task = asyncio.create_task(stream.__anext__())
        task_to_symbol[task] = symbol

    try:
        while True:
            if not task_to_symbol:
                raise RuntimeError("watch_order_book produced no active tasks")

            done, _ = await asyncio.wait(
                list(task_to_symbol.keys()),
                return_when=asyncio.FIRST_COMPLETED,
            )

            for finished in done:
                symbol = task_to_symbol.pop(finished)

                try:
                    payload = finished.result()
                except asyncio.CancelledError:
                    raise
                except StopAsyncIteration:
                    raise RuntimeError(
                        f"watch_order_book stream for {symbol} terminated unexpectedly"
                    )
                except Exception as exc:
                    raise exc

                latest_books[symbol] = payload

                stream = streams.get(symbol)
                if stream is not None:
                    next_task = asyncio.create_task(stream.__anext__())
                    task_to_symbol[next_task] = symbol

                if len(latest_books) < len(symbols):
                    continue

                now = loop.time()
                if throttle is not None and throttle > 0 and (now - last_emit) < throttle:
                    continue

                try:
                    simulation = simulate_opportunity_with_order_books(
                        opportunity,
                        latest_books,
                        starting_amount=starting_amount,
                        precision=precision,
                    )
                except InsufficientLiquidityError:
                    raise

                last_emit = now
                yield OpportunitySlippageUpdate(simulation, dict(latest_books))
    finally:
        for task in list(task_to_symbol.keys()):
            task.cancel()
        if task_to_symbol:
            await asyncio.gather(*task_to_symbol.keys(), return_exceptions=True)

        for stream in streams.values():
            close = getattr(stream, "aclose", None)
            if callable(close):
                try:
                    await close()
                except Exception:  # pragma: no cover - best effort cleanup
                    pass

