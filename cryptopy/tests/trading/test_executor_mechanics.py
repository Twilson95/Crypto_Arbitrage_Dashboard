import math

from cryptopy.src.trading.triangular_arbitrage.executor import TriangularArbitrageExecutor
from cryptopy.src.trading.triangular_arbitrage.models import TriangularTradeLeg


class _DummyExchange:
    def create_market_order(self, *args, **kwargs):  # pragma: no cover - not used in tests
        raise AssertionError("create_market_order should not be called during mechanics tests")

    def amount_to_precision(self, symbol: str, amount: float) -> float:
        return float(amount)


def _make_leg(
    *,
    symbol: str,
    side: str,
    amount_in: float,
    amount_out: float,
    traded_quantity: float,
    fee_rate: float = 0.0,
) -> TriangularTradeLeg:
    return TriangularTradeLeg(
        symbol=symbol,
        side=side,
        amount_in=amount_in,
        amount_out=amount_out,
        amount_out_without_fee=amount_out,
        average_price=amount_out / amount_in if amount_in else 0.0,
        fee_rate=fee_rate,
        fee_paid=0.0,
        traded_quantity=traded_quantity,
    )


def test_sell_leg_respects_available_balance() -> None:
    executor = TriangularArbitrageExecutor(_DummyExchange(), dry_run=False)
    leg = _make_leg(
        symbol="ETH/USD",
        side="sell",
        amount_in=1.0,
        amount_out=2000.0,
        traded_quantity=1.0,
    )

    amount, scale = executor._determine_order_amount(leg, available_amount=0.4)

    assert math.isclose(amount, 0.4, rel_tol=0, abs_tol=1e-12)
    assert math.isclose(scale, 0.4, rel_tol=0, abs_tol=1e-12)


def test_buy_leg_consumes_quote_fee_currency() -> None:
    executor = TriangularArbitrageExecutor(_DummyExchange(), dry_run=False)
    leg = _make_leg(
        symbol="ETH/USD",
        side="buy",
        amount_in=100.0,
        amount_out=0.05,
        traded_quantity=0.05,
    )
    order = {
        "filled": 0.05,
        "cost": 100.0,
        "fees": [
            {"currency": "USD", "cost": 0.2},
        ],
    }

    metrics = executor._extract_execution_metrics(order, leg)

    assert math.isclose(metrics["amount_in"], 100.2, rel_tol=0, abs_tol=1e-12)
    assert math.isclose(metrics["amount_out"], 0.05, rel_tol=0, abs_tol=1e-12)
    assert math.isclose(metrics["traded_quantity"], 0.05, rel_tol=0, abs_tol=1e-12)


def test_extract_realised_amount_respects_base_fee() -> None:
    executor = TriangularArbitrageExecutor(_DummyExchange(), dry_run=False)
    leg = _make_leg(
        symbol="ETH/USD",
        side="buy",
        amount_in=100.0,
        amount_out=0.05,
        traded_quantity=0.05,
    )
    order = {
        "filled": 0.05,
        "fee": {"currency": "ETH", "cost": 0.0005},
    }

    realised = executor._extract_realised_amount(order, leg)

    assert realised is not None
    assert math.isclose(realised, 0.0495, rel_tol=0, abs_tol=1e-12)


def test_order_amount_quantised_to_precision() -> None:
    class _PrecisionExchange(_DummyExchange):
        def amount_to_precision(self, symbol: str, amount: float) -> float:
            return math.floor(amount * 1000) / 1000

    exchange = _PrecisionExchange()
    executor = TriangularArbitrageExecutor(exchange, dry_run=False)
    leg = _make_leg(
        symbol="ETH/USD",
        side="buy",
        amount_in=100.0,
        amount_out=0.05,
        traded_quantity=0.0123456,
    )

    amount, scale = executor._determine_order_amount(leg, available_amount=100.0)

    assert math.isclose(amount, 0.012, rel_tol=0, abs_tol=1e-12)
    assert math.isclose(scale, 0.012 / 0.0123456, rel_tol=0, abs_tol=1e-12)
