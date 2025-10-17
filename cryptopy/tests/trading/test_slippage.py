from __future__ import annotations

import math

import pytest

from cryptopy.src.trading.triangular_arbitrage.models import (
    OrderBookSnapshot,
    TriangularOpportunity,
    TriangularRoute,
    TriangularTradeLeg,
)
from cryptopy.src.trading.triangular_arbitrage.slippage import (
    simulate_opportunity_with_order_books,
)
from cryptopy.src.trading.triangular_arbitrage.exceptions import InsufficientLiquidityError


def _make_opportunity() -> TriangularOpportunity:
    route = TriangularRoute(("ETH/USD", "ETH/BTC", "BTC/USD"), "USD")
    trades = [
        TriangularTradeLeg(
            symbol="ETH/USD",
            side="buy",
            amount_in=100.0,
            amount_out=0.05,
            amount_out_without_fee=0.05,
            average_price=2000.0,
            fee_rate=0.0,
            fee_paid=0.0,
            traded_quantity=0.05,
        ),
        TriangularTradeLeg(
            symbol="ETH/BTC",
            side="sell",
            amount_in=0.05,
            amount_out=0.0035,
            amount_out_without_fee=0.0035,
            average_price=0.07,
            fee_rate=0.0,
            fee_paid=0.0,
            traded_quantity=0.05,
        ),
        TriangularTradeLeg(
            symbol="BTC/USD",
            side="sell",
            amount_in=0.0035,
            amount_out=105.0,
            amount_out_without_fee=105.0,
            average_price=30_000.0,
            fee_rate=0.0,
            fee_paid=0.0,
            traded_quantity=0.0035,
        ),
    ]
    return TriangularOpportunity(
        route=route,
        starting_amount=100.0,
        final_amount=105.0,
        final_amount_without_fees=105.0,
        trades=trades,
    )


def test_simulation_without_slippage_matches_plan() -> None:
    opportunity = _make_opportunity()
    order_books = {
        "ETH/USD": OrderBookSnapshot("ETH/USD", bids=[(1999.0, 1.0)], asks=[(2000.0, 1.0)]),
        "ETH/BTC": OrderBookSnapshot("ETH/BTC", bids=[(0.07, 1.0)], asks=[(0.071, 1.0)]),
        "BTC/USD": OrderBookSnapshot("BTC/USD", bids=[(30_000.0, 1.0)], asks=[(30_100.0, 1.0)]),
    }

    simulation = simulate_opportunity_with_order_books(
        opportunity,
        order_books,
        starting_amount=opportunity.starting_amount,
    )

    assert math.isclose(simulation.opportunity.final_amount, 105.0, rel_tol=0, abs_tol=1e-9)
    assert all(math.isclose(leg.slippage_pct, 0.0, abs_tol=1e-9) for leg in simulation.legs)


def test_simulation_with_slippage_reduces_profit() -> None:
    opportunity = _make_opportunity()
    order_books = {
        "ETH/USD": OrderBookSnapshot(
            "ETH/USD",
            bids=[(1999.0, 1.0)],
            asks=[(2000.0, 0.02), (2005.0, 0.08)],
        ),
        "ETH/BTC": OrderBookSnapshot(
            "ETH/BTC",
            bids=[(0.07, 0.02), (0.0695, 1.0)],
            asks=[(0.071, 1.0)],
        ),
        "BTC/USD": OrderBookSnapshot(
            "BTC/USD",
            bids=[(30_000.0, 0.002), (29_900.0, 1.0)],
            asks=[(30_100.0, 1.0)],
        ),
    }

    simulation = simulate_opportunity_with_order_books(
        opportunity,
        order_books,
        starting_amount=opportunity.starting_amount,
    )

    assert simulation.opportunity.final_amount < opportunity.final_amount
    assert any(leg.slippage_pct > 0 for leg in simulation.legs)


def test_simulation_raises_when_depth_insufficient() -> None:
    opportunity = _make_opportunity()
    order_books = {
        "ETH/USD": OrderBookSnapshot("ETH/USD", bids=[(1999.0, 1.0)], asks=[(2000.0, 0.01)]),
        "ETH/BTC": OrderBookSnapshot("ETH/BTC", bids=[(0.07, 1.0)], asks=[(0.071, 1.0)]),
        "BTC/USD": OrderBookSnapshot("BTC/USD", bids=[(30_000.0, 1.0)], asks=[(30_100.0, 1.0)]),
    }

    with pytest.raises(InsufficientLiquidityError):
        simulate_opportunity_with_order_books(
            opportunity,
            order_books,
            starting_amount=opportunity.starting_amount,
        )
