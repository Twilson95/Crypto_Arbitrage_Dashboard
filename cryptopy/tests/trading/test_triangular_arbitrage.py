from dataclasses import dataclass
import importlib.util
import pathlib
import sys

import csv
import pytest

MODULE_PATH = pathlib.Path(__file__).resolve().parents[2] / "src" / "trading" / "triangular_arbitrage" / "__init__.py"
spec = importlib.util.spec_from_file_location("triangular_arbitrage", MODULE_PATH)
triangular_arbitrage = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = triangular_arbitrage
assert spec.loader is not None
spec.loader.exec_module(triangular_arbitrage)

InsufficientLiquidityError = triangular_arbitrage.InsufficientLiquidityError
PriceSnapshot = triangular_arbitrage.PriceSnapshot
TriangularArbitrageCalculator = triangular_arbitrage.TriangularArbitrageCalculator
TriangularArbitrageExecutor = triangular_arbitrage.TriangularArbitrageExecutor
TriangularOpportunity = triangular_arbitrage.TriangularOpportunity
TriangularRoute = triangular_arbitrage.TriangularRoute


@dataclass
class MockExchange:
    fee: float = 0.001
    orders: list = None

    def __post_init__(self):
        self.orders = []

    def get_taker_fee(self, symbol: str) -> float:
        return self.fee

    def create_market_order(self, symbol, side, amount, **kwargs):
        order = {"symbol": symbol, "side": side, "amount": amount, **kwargs}
        self.orders.append(order)
        return order


def make_price_snapshot(symbol, bid, ask):
    return PriceSnapshot(symbol=symbol, bid=bid, ask=ask)


def test_profitable_route_detected():
    exchange = MockExchange()
    calculator = TriangularArbitrageCalculator(exchange)  # type: ignore[arg-type]
    route = TriangularRoute(("BTC/USD", "ETH/BTC", "ETH/USD"), "USD")

    prices = {
        "BTC/USD": make_price_snapshot("BTC/USD", bid=101.0, ask=100.0),
        "ETH/BTC": make_price_snapshot("ETH/BTC", bid=0.55, ask=0.5),
        "ETH/USD": make_price_snapshot("ETH/USD", bid=0.55, ask=0.56),
    }

    opportunity = calculator.evaluate_route(route, prices, starting_amount=1000.0)

    assert opportunity is not None
    assert opportunity.final_amount > opportunity.starting_amount
    assert pytest.approx(opportunity.profit_percentage, rel=1e-3) > 5.0


def test_route_filtered_when_below_threshold():
    exchange = MockExchange(fee=0.01)
    calculator = TriangularArbitrageCalculator(exchange)  # type: ignore[arg-type]
    route = TriangularRoute(("BTC/USD", "ETH/BTC", "ETH/USD"), "USD")

    prices = {
        "BTC/USD": make_price_snapshot("BTC/USD", bid=101.0, ask=100.0),
        "ETH/BTC": make_price_snapshot("ETH/BTC", bid=0.5, ask=0.5),
        "ETH/USD": make_price_snapshot("ETH/USD", bid=0.5, ask=0.5),
    }

    opportunity = calculator.evaluate_route(
        route, prices, starting_amount=1000.0, min_profit_percentage=0.1
    )

    assert opportunity is None


def test_insufficient_liquidity_raises():
    exchange = MockExchange()
    calculator = TriangularArbitrageCalculator(exchange)  # type: ignore[arg-type]
    route = TriangularRoute(("BTC/USD", "ETH/BTC", "ETH/USD"), "USD")

    prices = {
        "BTC/USD": make_price_snapshot("BTC/USD", bid=101.0, ask=None),
        "ETH/BTC": make_price_snapshot("ETH/BTC", bid=0.55, ask=0.5),
        "ETH/USD": make_price_snapshot("ETH/USD", bid=0.55, ask=0.56),
    }

    with pytest.raises(InsufficientLiquidityError):
        calculator.evaluate_route(route, prices, starting_amount=1000.0)


def test_executor_places_orders_in_sequence(tmp_path):
    exchange = MockExchange()
    calculator = TriangularArbitrageCalculator(exchange)  # type: ignore[arg-type]
    log_path = tmp_path / "trades.csv"
    executor = TriangularArbitrageExecutor(exchange, dry_run=True, trade_log_path=log_path)  # type: ignore[arg-type]
    route = TriangularRoute(("BTC/USD", "ETH/BTC", "ETH/USD"), "USD")

    prices = {
        "BTC/USD": make_price_snapshot("BTC/USD", bid=101.0, ask=100.0),
        "ETH/BTC": make_price_snapshot("ETH/BTC", bid=0.55, ask=0.5),
        "ETH/USD": make_price_snapshot("ETH/USD", bid=0.55, ask=0.56),
    }

    opportunity = calculator.evaluate_route(route, prices, starting_amount=100.0)
    assert opportunity is not None
    orders = executor.execute(opportunity)

    assert len(orders) == 3
    assert all(order["test_order"] for order in orders)
    assert [order["side"] for order in orders] == ["buy", "buy", "sell"]

    with log_path.open() as csvfile:
        reader = csv.DictReader(csvfile)
        rows = list(reader)

    assert len(rows) == 3
    assert {row["symbol"] for row in rows} == {"BTC/USD", "ETH/BTC", "ETH/USD"}
    assert all(row["dry_run"] == "True" for row in rows)


def test_find_routes_respects_max_route_length():
    exchange = MockExchange()
    calculator = TriangularArbitrageCalculator(exchange)  # type: ignore[arg-type]
    long_route = TriangularRoute(
        ("BTC/USD", "ETH/BTC", "XRP/ETH", "XRP/USD"),
        "USD",
    )
    short_route = TriangularRoute(("BTC/USD", "ETH/BTC", "ETH/USD"), "USD")

    prices = {
        "BTC/USD": make_price_snapshot("BTC/USD", bid=101.0, ask=100.0),
        "ETH/BTC": make_price_snapshot("ETH/BTC", bid=0.55, ask=0.5),
        "XRP/ETH": make_price_snapshot("XRP/ETH", bid=0.002, ask=0.0018),
        "XRP/USD": make_price_snapshot("XRP/USD", bid=0.55, ask=0.56),
        "ETH/USD": make_price_snapshot("ETH/USD", bid=0.55, ask=0.56),
    }

    opportunities, stats = calculator.find_profitable_routes(
        [long_route, short_route],
        prices,
        starting_amount=100.0,
        max_route_length=3,
    )

    assert all(len(opp.route.symbols) <= 3 for opp in opportunities)
    assert all(opp.route != long_route for opp in opportunities)
    assert stats.filtered_by_length == 1
    assert stats.considered == 1
    assert stats.evaluation_error_reasons == {}


def test_error_reasons_tracked_for_missing_prices():
    exchange = MockExchange()
    calculator = TriangularArbitrageCalculator(exchange)  # type: ignore[arg-type]
    route = TriangularRoute(("BTC/USD", "ETH/BTC", "ETH/USD"), "USD")

    prices = {
        "BTC/USD": make_price_snapshot("BTC/USD", bid=101.0, ask=100.0),
        # Intentionally omit ETH/BTC snapshot to trigger KeyError
        "ETH/USD": make_price_snapshot("ETH/USD", bid=0.55, ask=0.56),
    }

    opportunities, stats = calculator.find_profitable_routes(
        [route],
        prices,
        starting_amount=100.0,
    )

    assert opportunities == []
    assert stats.evaluation_errors == 1
    assert stats.rejected_by_profit == 0
    assert stats.evaluation_error_reasons == {
        "KeyError: Missing price snapshot for ETH/BTC": 1
    }
