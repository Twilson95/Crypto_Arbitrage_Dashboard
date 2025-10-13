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
OrderBookSnapshot = triangular_arbitrage.OrderBookSnapshot
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


def make_order_book(symbol, bids, asks):
    return OrderBookSnapshot(symbol=symbol, bids=bids, asks=asks)


def test_profitable_route_detected():
    exchange = MockExchange()
    calculator = TriangularArbitrageCalculator(exchange)  # type: ignore[arg-type]
    route = TriangularRoute(("BTC/USD", "ETH/BTC", "ETH/USD"), "USD")

    order_books = {
        "BTC/USD": make_order_book("BTC/USD", bids=[(101.0, 50)], asks=[(100.0, 50)]),
        "ETH/BTC": make_order_book("ETH/BTC", bids=[(0.55, 3000)], asks=[(0.5, 3000)]),
        "ETH/USD": make_order_book("ETH/USD", bids=[(0.55, 3000)], asks=[(0.56, 3000)]),
    }

    opportunity = calculator.evaluate_route(route, order_books, starting_amount=1000.0)

    assert opportunity is not None
    assert opportunity.final_amount > opportunity.starting_amount
    assert pytest.approx(opportunity.profit_percentage, rel=1e-3) > 5.0


def test_route_filtered_when_below_threshold():
    exchange = MockExchange(fee=0.01)
    calculator = TriangularArbitrageCalculator(exchange)  # type: ignore[arg-type]
    route = TriangularRoute(("BTC/USD", "ETH/BTC", "ETH/USD"), "USD")

    order_books = {
        "BTC/USD": make_order_book("BTC/USD", bids=[(101.0, 50)], asks=[(100.0, 50)]),
        "ETH/BTC": make_order_book("ETH/BTC", bids=[(0.5, 3000)], asks=[(0.5, 3000)]),
        "ETH/USD": make_order_book("ETH/USD", bids=[(0.5, 3000)], asks=[(0.5, 3000)]),
    }

    opportunity = calculator.evaluate_route(
        route, order_books, starting_amount=1000.0, min_profit_percentage=0.1
    )

    assert opportunity is None


def test_insufficient_liquidity_raises():
    exchange = MockExchange()
    calculator = TriangularArbitrageCalculator(exchange)  # type: ignore[arg-type]
    route = TriangularRoute(("BTC/USD", "ETH/BTC", "ETH/USD"), "USD")

    order_books = {
        "BTC/USD": make_order_book("BTC/USD", bids=[(101.0, 1)], asks=[(100.0, 1)]),
        "ETH/BTC": make_order_book("ETH/BTC", bids=[(0.55, 1)], asks=[(0.5, 1)]),
        "ETH/USD": make_order_book("ETH/USD", bids=[(0.55, 1)], asks=[(0.56, 1)]),
    }

    with pytest.raises(InsufficientLiquidityError):
        calculator.evaluate_route(route, order_books, starting_amount=1000.0)


def test_executor_places_orders_in_sequence(tmp_path):
    exchange = MockExchange()
    calculator = TriangularArbitrageCalculator(exchange)  # type: ignore[arg-type]
    log_path = tmp_path / "trades.csv"
    executor = TriangularArbitrageExecutor(exchange, dry_run=True, trade_log_path=log_path)  # type: ignore[arg-type]
    route = TriangularRoute(("BTC/USD", "ETH/BTC", "ETH/USD"), "USD")

    order_books = {
        "BTC/USD": make_order_book("BTC/USD", bids=[(101.0, 50)], asks=[(100.0, 50)]),
        "ETH/BTC": make_order_book("ETH/BTC", bids=[(0.55, 3000)], asks=[(0.5, 3000)]),
        "ETH/USD": make_order_book("ETH/USD", bids=[(0.55, 3000)], asks=[(0.56, 3000)]),
    }

    opportunity = calculator.evaluate_route(route, order_books, starting_amount=100.0)
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

    order_books = {
        "BTC/USD": make_order_book("BTC/USD", bids=[(101.0, 50)], asks=[(100.0, 50)]),
        "ETH/BTC": make_order_book("ETH/BTC", bids=[(0.55, 3000)], asks=[(0.5, 3000)]),
        "XRP/ETH": make_order_book("XRP/ETH", bids=[(0.002, 100000)], asks=[(0.0018, 100000)]),
        "XRP/USD": make_order_book("XRP/USD", bids=[(0.55, 100000)], asks=[(0.56, 100000)]),
        "ETH/USD": make_order_book("ETH/USD", bids=[(0.55, 3000)], asks=[(0.56, 3000)]),
    }

    opportunities = calculator.find_profitable_routes(
        [long_route, short_route],
        order_books,
        starting_amount=100.0,
        max_route_length=3,
    )

    assert all(len(opp.route.symbols) <= 3 for opp in opportunities)
    assert all(opp.route != long_route for opp in opportunities)
