import asyncio
import pytest

from cryptopy.src.trading.triangular_arbitrage.models import (
    OrderBookSnapshot,
    TriangularOpportunity,
    TriangularRoute,
    TriangularTradeLeg,
)
from cryptopy.src.trading.triangular_arbitrage.slippage_stream import (
    OpportunitySlippageUpdate,
    stream_opportunity_slippage,
)


class _FakeExchange:
    def __init__(self, symbols):
        self._queues = {symbol: asyncio.Queue() for symbol in symbols}
        self.watch_calls = []

    def amount_to_precision(self, symbol, value):  # pragma: no cover - helper
        return float(value)

    def cost_to_precision(self, symbol, value):  # pragma: no cover - helper
        return float(value)

    async def watch_order_book(
        self,
        symbol,
        *,
        limit=10,
        websocket_timeout=None,
        require_websocket=True,
    ):
        self.watch_calls.append((symbol, limit, require_websocket))
        try:
            while True:
                snapshot = await self._queues[symbol].get()
                yield snapshot
        except asyncio.CancelledError:
            raise

    async def push(self, symbol, snapshot):
        await self._queues[symbol].put(snapshot)


def _make_order_book(symbol, *, bid, bid_qty, ask, ask_qty):
    return OrderBookSnapshot(
        symbol=symbol,
        bids=[(bid, bid_qty)],
        asks=[(ask, ask_qty)],
    )


def _make_buy_pressure(symbol):
    # Preferential ask depth at 10, with an additional level at 10.5.
    return OrderBookSnapshot(
        symbol=symbol,
        bids=[(9.5, 100.0)],
        asks=[(10.0, 5.0), (10.5, 20.0)],
    )


def test_stream_opportunity_slippage_emits_updates():
    opportunity = TriangularOpportunity(
        route=TriangularRoute(
            symbols=("AAA/BBB", "AAA/CCC", "CCC/BBB"),
            starting_currency="BBB",
        ),
        starting_amount=100.0,
        final_amount=102.0,
        final_amount_without_fees=102.0,
        trades=[
            TriangularTradeLeg(
                symbol="AAA/BBB",
                side="buy",
                amount_in=100.0,
                amount_out=10.0,
                amount_out_without_fee=10.0,
                average_price=10.0,
                fee_rate=0.0,
                fee_paid=0.0,
                traded_quantity=10.0,
            ),
            TriangularTradeLeg(
                symbol="AAA/CCC",
                side="sell",
                amount_in=10.0,
                amount_out=200.0,
                amount_out_without_fee=200.0,
                average_price=20.0,
                fee_rate=0.0,
                fee_paid=0.0,
                traded_quantity=10.0,
            ),
            TriangularTradeLeg(
                symbol="CCC/BBB",
                side="sell",
                amount_in=200.0,
                amount_out=102.0,
                amount_out_without_fee=102.0,
                average_price=0.51,
                fee_rate=0.0,
                fee_paid=0.0,
                traded_quantity=200.0,
            ),
        ],
    )

    exchange = _FakeExchange(opportunity.route.symbols)

    async def _collect():
        updates = []
        async for update in stream_opportunity_slippage(
            exchange,
            opportunity,
            starting_amount=100.0,
            depth=5,
            throttle=None,
        ):
            updates.append(update)
            if len(updates) == 2:
                return updates

    async def _run_test():
        task = asyncio.create_task(_collect())

        await exchange.push(
            "AAA/BBB",
            OrderBookSnapshot(
                symbol="AAA/BBB",
                bids=[(9.9, 100.0)],
                asks=[(10.0, 20.0)],
            ),
        )
        await exchange.push(
            "AAA/CCC",
            _make_order_book("AAA/CCC", bid=20.0, bid_qty=50.0, ask=20.1, ask_qty=50.0),
        )
        await exchange.push(
            "CCC/BBB",
            _make_order_book("CCC/BBB", bid=0.51, bid_qty=400.0, ask=0.52, ask_qty=400.0),
        )

        await exchange.push("AAA/BBB", _make_buy_pressure("AAA/BBB"))

        return await asyncio.wait_for(task, timeout=1.0)

    updates: list[OpportunitySlippageUpdate] = asyncio.run(_run_test())

    assert len(updates) == 2

    first, second = updates

    assert first.simulation.opportunity.final_amount == pytest.approx(102.0)
    assert all(leg.slippage_pct == pytest.approx(0.0) for leg in first.simulation.legs)

    assert second.simulation.opportunity.final_amount < 102.0
    first_leg_slippage = next(leg for leg in second.simulation.legs if leg.symbol == "AAA/BBB")
    assert first_leg_slippage.slippage_pct > 0.0

    assert all(call[2] is True for call in exchange.watch_calls)

