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
    PrecisionAdapter,
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
    for plan_leg, leg in zip(opportunity.trades, simulation.legs):
        assert math.isclose(leg.slippage_pct, 0.0, abs_tol=1e-9)
        assert math.isclose(leg.output_slippage_pct, 0.0, abs_tol=1e-9)
        assert math.isclose(leg.input_slippage_pct, 0.0, abs_tol=1e-9)
        assert math.isclose(leg.expected_amount_in, plan_leg.amount_in, abs_tol=1e-9)
        assert math.isclose(leg.actual_amount_in, plan_leg.amount_in, abs_tol=1e-9)
        assert math.isclose(leg.expected_amount_out, plan_leg.amount_out, abs_tol=1e-9)
        assert math.isclose(leg.actual_amount_out, plan_leg.amount_out, abs_tol=1e-9)


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
    assert any(leg.output_slippage_pct > 0 for leg in simulation.legs)


def test_simulation_scales_expected_amounts() -> None:
    opportunity = _make_opportunity()
    order_books = {
        "ETH/USD": OrderBookSnapshot("ETH/USD", bids=[(1999.0, 1.0)], asks=[(2000.0, 1.0)]),
        "ETH/BTC": OrderBookSnapshot("ETH/BTC", bids=[(0.07, 1.0)], asks=[(0.071, 1.0)]),
        "BTC/USD": OrderBookSnapshot("BTC/USD", bids=[(30_000.0, 1.0)], asks=[(30_100.0, 1.0)]),
    }

    scale = 0.4
    simulation = simulate_opportunity_with_order_books(
        opportunity,
        order_books,
        starting_amount=opportunity.starting_amount * scale,
    )

    for plan_leg, leg in zip(opportunity.trades, simulation.legs):
        assert math.isclose(leg.expected_amount_in, plan_leg.amount_in * scale, rel_tol=0, abs_tol=1e-12)
        assert math.isclose(leg.expected_amount_out, plan_leg.amount_out * scale, rel_tol=0, abs_tol=1e-12)
        assert math.isclose(leg.actual_amount_in, plan_leg.amount_in * scale, rel_tol=0, abs_tol=1e-12)
        assert math.isclose(leg.actual_amount_out, plan_leg.amount_out * scale, rel_tol=0, abs_tol=1e-12)


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


def test_simulation_respects_precision_adapter() -> None:
    opportunity = _make_opportunity()
    order_books = {
        "ETH/USD": OrderBookSnapshot("ETH/USD", bids=[(1999.0, 1.0)], asks=[(2000.0, 1.0)]),
        "ETH/BTC": OrderBookSnapshot("ETH/BTC", bids=[(0.07, 1.0)], asks=[(0.071, 1.0)]),
        "BTC/USD": OrderBookSnapshot("BTC/USD", bids=[(30_000.0, 1.0)], asks=[(30_100.0, 1.0)]),
    }

    amount_decimals = {"ETH/USD": 3, "ETH/BTC": 5, "BTC/USD": 4}
    cost_decimals = {"ETH/USD": 2, "ETH/BTC": 8, "BTC/USD": 2}

    def _floor(value: float, decimals: int) -> float:
        factor = 10 ** decimals
        return math.floor(value * factor) / factor

    precision = PrecisionAdapter(
        amount_to_precision=lambda symbol, amount: _floor(amount, amount_decimals[symbol]),
        cost_to_precision=lambda symbol, cost: _floor(cost, cost_decimals[symbol]),
    )

    scale = 0.3333333
    simulation = simulate_opportunity_with_order_books(
        opportunity,
        order_books,
        starting_amount=opportunity.starting_amount * scale,
        precision=precision,
    )

    for plan_leg, leg in zip(opportunity.trades, simulation.legs):
        if plan_leg.side == "buy":
            expected_in = _floor(plan_leg.amount_in * scale, cost_decimals[leg.symbol])
            expected_out = _floor(plan_leg.amount_out * scale, amount_decimals[leg.symbol])
        else:
            expected_in = _floor(plan_leg.amount_in * scale, amount_decimals[leg.symbol])
            expected_out = _floor(plan_leg.amount_out * scale, cost_decimals[leg.symbol])

        assert math.isclose(leg.expected_amount_in, expected_in, rel_tol=0, abs_tol=1e-12)
        assert math.isclose(leg.expected_amount_out, expected_out, rel_tol=0, abs_tol=1e-12)
        assert math.isclose(leg.actual_amount_in, expected_in, rel_tol=0, abs_tol=1e-12)
        assert math.isclose(leg.actual_amount_out, expected_out, rel_tol=0, abs_tol=1e-12)
