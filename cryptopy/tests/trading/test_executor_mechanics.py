import math
from typing import Any, Dict, List, Tuple

import pytest

from cryptopy.src.trading.triangular_arbitrage.executor import (
    MinimumTradeSizeError,
    TriangularArbitrageExecutor,
)
from cryptopy.src.trading.triangular_arbitrage.models import (
    TriangularOpportunity,
    TriangularRoute,
    TriangularTradeLeg,
)


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


def _make_staggered_opportunity() -> TriangularOpportunity:
    route = TriangularRoute(("USDC/USD", "ALGO/USDC", "ALGO/USD"), "USD")
    trades = [
        _make_leg(
            symbol="USDC/USD",
            side="buy",
            amount_in=100.0,
            amount_out=100.0,
            traded_quantity=100.0,
        ),
        _make_leg(
            symbol="ALGO/USDC",
            side="buy",
            amount_in=100.0,
            amount_out=200.0,
            traded_quantity=200.0,
        ),
        _make_leg(
            symbol="ALGO/USD",
            side="sell",
            amount_in=200.0,
            amount_out=101.0,
            traded_quantity=200.0,
        ),
    ]
    return TriangularOpportunity(
        route=route,
        starting_amount=100.0,
        final_amount=101.0,
        final_amount_without_fees=101.0,
        trades=trades,
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


def test_submit_leg_order_retries_after_insufficient_funds() -> None:
    class InsufficientFunds(Exception):
        pass

    class _RetryExchange:
        def __init__(self) -> None:
            self.calls: List[float] = []
            self.failures = 0

        def create_market_order(self, symbol: str, side: str, amount: float, **_: Any) -> Dict[str, Any]:
            self.calls.append(float(amount))
            if self.failures == 0:
                self.failures += 1
                raise InsufficientFunds("Insufficient funds")
            return {"id": "order", "symbol": symbol, "side": side, "amount": float(amount)}

        def amount_to_precision(self, symbol: str, amount: float) -> float:
            return float(amount)

    exchange = _RetryExchange()
    executor = TriangularArbitrageExecutor(exchange, dry_run=False)
    leg = _make_leg(
        symbol="PAXG/ETH",
        side="buy",
        amount_in=100.0,
        amount_out=1.0,
        traded_quantity=1.0,
    )

    order, amount, scale = executor._submit_leg_order(leg, available_amount=100.0)

    assert len(exchange.calls) >= 2
    assert exchange.calls[0] > exchange.calls[-1]
    assert math.isclose(exchange.calls[0], 1.0, rel_tol=0, abs_tol=1e-12)
    assert math.isclose(exchange.calls[-1], amount, rel_tol=0, abs_tol=1e-12)
    assert amount < 1.0
    assert scale < 1.0
    assert order["amount"] == amount


def test_submit_leg_order_caps_to_exchange_balance() -> None:
    class _BalanceExchange:
        def __init__(self) -> None:
            self.balance_calls = 0
            self.orders: List[float] = []

        def fetch_balance(self) -> Dict[str, Any]:
            self.balance_calls += 1
            return {"free": {"USD": 50.0}}

        def create_market_order(self, symbol: str, side: str, amount: float, **_: Any) -> Dict[str, Any]:
            self.orders.append(float(amount))
            return {"id": "order", "symbol": symbol, "side": side, "amount": float(amount)}

        def amount_to_precision(self, symbol: str, amount: float) -> float:
            return float(amount)

    exchange = _BalanceExchange()
    executor = TriangularArbitrageExecutor(exchange, dry_run=False)
    leg = _make_leg(
        symbol="ETH/USD",
        side="buy",
        amount_in=100.0,
        amount_out=0.05,
        traded_quantity=0.05,
    )

    order, amount, scale = executor._submit_leg_order(leg, available_amount=100.0)

    assert exchange.balance_calls >= 1
    assert math.isclose(amount, 0.025, rel_tol=0, abs_tol=1e-12)
    assert math.isclose(scale, 0.5, rel_tol=0, abs_tol=1e-12)
    assert order["amount"] == amount
    assert exchange.orders == [amount]


def test_submit_leg_order_can_skip_initial_balance_sync() -> None:
    class _Exchange:
        def __init__(self) -> None:
            self.balance_calls = 0

        def fetch_balance(self) -> Dict[str, Any]:
            self.balance_calls += 1
            return {"free": {"USD": 100.0}}

        def create_market_order(
            self, symbol: str, side: str, amount: float, **_: Any
        ) -> Dict[str, Any]:
            return {"id": "order", "symbol": symbol, "side": side, "amount": float(amount)}

        def amount_to_precision(self, symbol: str, amount: float) -> float:
            return float(amount)

    exchange = _Exchange()
    executor = TriangularArbitrageExecutor(exchange, dry_run=False)
    leg = _make_leg(
        symbol="ETH/USD",
        side="buy",
        amount_in=100.0,
        amount_out=0.05,
        traded_quantity=0.05,
    )

    executor._submit_leg_order(leg, available_amount=100.0, sync_balance=False)

    assert exchange.balance_calls == 0


def test_submit_leg_order_enables_balance_sync_after_insufficient_funds() -> None:
    class InsufficientFunds(Exception):
        pass

    class _Exchange:
        def __init__(self) -> None:
            self.balance_calls = 0
            self.calls = 0

        def fetch_balance(self) -> Dict[str, Any]:
            self.balance_calls += 1
            return {"free": {"USD": 50.0}}

        def create_market_order(
            self, symbol: str, side: str, amount: float, **_: Any
        ) -> Dict[str, Any]:
            self.calls += 1
            if self.calls == 1:
                raise InsufficientFunds("Insufficient funds")
            return {"id": "order", "symbol": symbol, "side": side, "amount": float(amount)}

        def amount_to_precision(self, symbol: str, amount: float) -> float:
            return float(amount)

    exchange = _Exchange()
    executor = TriangularArbitrageExecutor(exchange, dry_run=False)
    leg = _make_leg(
        symbol="ETH/USD",
        side="buy",
        amount_in=100.0,
        amount_out=0.05,
        traded_quantity=0.05,
    )

    executor._submit_leg_order(leg, available_amount=100.0, sync_balance=False)

    assert exchange.balance_calls >= 1


def test_submit_leg_order_enforces_minimum_trade_size() -> None:
    class _MinExchange(_DummyExchange):
        def __init__(self) -> None:
            self.orders: List[float] = []

        def get_min_trade_values(self, symbol: str) -> Dict[str, float]:
            return {"amount": 0.5, "cost": 20.0}

        def create_market_order(self, symbol: str, side: str, amount: float, **_: Any) -> Dict[str, Any]:
            self.orders.append(float(amount))
            return {"id": "order", "symbol": symbol, "side": side, "amount": float(amount)}

    exchange = _MinExchange()
    executor = TriangularArbitrageExecutor(exchange, dry_run=False)
    leg = _make_leg(
        symbol="AAVE/USD",
        side="buy",
        amount_in=10.0,
        amount_out=0.2,
        traded_quantity=0.2,
    )

    with pytest.raises(MinimumTradeSizeError):
        executor._submit_leg_order(leg, available_amount=10.0)

    assert exchange.orders == []


def test_watch_order_update_without_websocket_hook() -> None:
    executor = TriangularArbitrageExecutor(_DummyExchange(), dry_run=False)

    payload, used = executor._watch_order_update("order", "ETH/USD", 0.1)

    assert payload is None
    assert used is False


def test_watch_order_update_uses_exchange_hook() -> None:
    class _WatchExchange(_DummyExchange):
        def __init__(self) -> None:
            self.calls: List[Tuple[str, str, float]] = []

        def watch_order_via_websocket(self, order_id: str, symbol: str, timeout: float):
            self.calls.append((order_id, symbol, timeout))
            return {"id": order_id, "status": "closed"}

    exchange = _WatchExchange()
    executor = TriangularArbitrageExecutor(exchange, dry_run=False)

    payload, used = executor._watch_order_update("abc", "ETH/USD", 0.2)

    assert used is True
    assert payload == {"id": "abc", "status": "closed"}
    assert exchange.calls == [("abc", "ETH/USD", 0.2)]


def test_speculative_submit_retries_until_success() -> None:
    class InsufficientFunds(Exception):
        pass

    class _SpeculativeExchange:
        def __init__(self) -> None:
            self.calls = 0

        def create_market_order(self, symbol: str, side: str, amount: float, **_: Any) -> Dict[str, Any]:
            self.calls += 1
            if self.calls < 3:
                raise InsufficientFunds("insufficient funds")
            return {"id": f"order-{self.calls}", "symbol": symbol, "side": side, "amount": float(amount)}

        def amount_to_precision(self, symbol: str, amount: float) -> float:
            return float(amount)

    exchange = _SpeculativeExchange()
    executor = TriangularArbitrageExecutor(
        exchange,
        dry_run=False,
        staggered_leg_delay=0.0,
        staggered_slippage_assumption=[0.0],
    )
    leg = _make_leg(
        symbol="ETH/USD",
        side="buy",
        amount_in=100.0,
        amount_out=1.0,
        traded_quantity=1.0,
    )

    order, amount, scale = executor._submit_leg_order(
        leg,
        available_amount=100.0,
        speculative=True,
    )

    assert exchange.calls == 3
    assert order["id"] == "order-3"
    assert math.isclose(amount, 1.0, rel_tol=0, abs_tol=1e-12)
    assert math.isclose(scale, 1.0, rel_tol=0, abs_tol=1e-12)


def test_speculative_submit_scales_to_available_balance() -> None:
    class InsufficientFunds(Exception):
        pass

    class _PartialExchange:
        def __init__(self) -> None:
            self.calls = 0
            self.balance = 4.0

        def create_market_order(
            self, symbol: str, side: str, amount: float, **_: Any
        ) -> Dict[str, Any]:
            self.calls += 1
            if self.calls == 1:
                raise InsufficientFunds("insufficient funds")
            return {
                "id": f"order-{self.calls}",
                "symbol": symbol,
                "side": side,
                "amount": float(amount),
            }

        def fetch_balance(self) -> Dict[str, Dict[str, float]]:
            return {"free": {"ALGO": self.balance}}

        def amount_to_precision(self, symbol: str, amount: float) -> float:
            return float(amount)

    exchange = _PartialExchange()
    executor = TriangularArbitrageExecutor(
        exchange,
        dry_run=False,
        staggered_leg_delay=0.0,
        staggered_slippage_assumption=[0.0],
    )

    leg = _make_leg(
        symbol="ALGO/USD",
        side="sell",
        amount_in=100.0,
        amount_out=100.0,
        traded_quantity=100.0,
    )

    order, amount, scale = executor._submit_leg_order(
        leg,
        available_amount=100.0,
        speculative=True,
    )

    assert exchange.calls == 2
    assert order["id"] == "order-2"
    assert math.isclose(amount, 4.0, rel_tol=0, abs_tol=1e-12)
    assert math.isclose(scale, 0.04, rel_tol=0, abs_tol=1e-12)


def test_reconcile_staggered_gap_executes_residual() -> None:
    class _StaggeredExchange(_DummyExchange):
        def __init__(self) -> None:
            self.orders: List[float] = []

        def create_market_order(self, symbol: str, side: str, amount: float, **_: Any) -> Dict[str, Any]:
            self.orders.append(float(amount))
            return {"id": f"order-{len(self.orders)}", "symbol": symbol, "side": side, "amount": float(amount)}

    class _StaggeredExecutor(TriangularArbitrageExecutor):
        def __init__(self) -> None:
            super().__init__(
                _StaggeredExchange(),
                dry_run=False,
                staggered_leg_delay=0.0,
                staggered_slippage_assumption=[0.0],
            )
            self._submitted: List[float] = []

        def _submit_leg_order(
            self,
            leg: TriangularTradeLeg,
            available_amount: float,
            *,
            speculative: bool = False,
        ) -> Tuple[Dict[str, Any], float, float]:
            self._submitted.append(available_amount)
            return super()._submit_leg_order(leg, available_amount, speculative=speculative)

        def _finalise_order_execution(
            self, order: Dict[str, Any], leg: TriangularTradeLeg
        ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
            amount = float(order["amount"])
            metrics = {
                "amount_in": amount,
                "amount_out": amount,
                "amount_out_without_fee": amount,
                "traded_quantity": amount,
                "fee_breakdown": {},
            }
            return order, metrics

    executor = _StaggeredExecutor()
    leg = _make_leg(
        symbol="ALGO/USD",
        side="sell",
        amount_in=100.0,
        amount_out=100.0,
        traded_quantity=100.0,
    )

    previous_state = {"aggregated_metrics": {"amount_out": 100.0}}
    current_state = {
        "leg": leg,
        "orders": [],
        "metrics": [
            {
                "amount_in": 90.0,
                "amount_out": 90.0,
                "amount_out_without_fee": 90.0,
                "traded_quantity": 90.0,
                "fee_breakdown": {},
            }
        ],
        "submit_durations": [],
        "fill_durations": [],
    }
    current_state["aggregated_metrics"] = executor._combine_execution_metrics(
        current_state["metrics"]
    )

    reconciled = executor._reconcile_staggered_gap(previous_state, current_state)

    assert executor._submitted[-1] == pytest.approx(10.0)
    assert math.isclose(reconciled["amount_in"], 100.0, rel_tol=0, abs_tol=1e-12)
    assert math.isclose(reconciled["amount_out"], 100.0, rel_tol=0, abs_tol=1e-12)
    assert len(current_state["orders"]) == 1


def test_reconcile_staggered_gap_skips_residual_below_minimum() -> None:
    class _MinResidualExchange(_DummyExchange):
        def __init__(self) -> None:
            self.orders: List[float] = []

        def get_min_trade_values(self, symbol: str) -> Dict[str, float]:
            return {"amount": 0.5}

        def create_market_order(self, symbol: str, side: str, amount: float, **_: Any) -> Dict[str, Any]:
            self.orders.append(float(amount))
            return {"id": "order", "symbol": symbol, "side": side, "amount": float(amount)}

    executor = TriangularArbitrageExecutor(
        _MinResidualExchange(),
        dry_run=False,
        staggered_leg_delay=0.0,
        staggered_slippage_assumption=[0.0],
    )
    exchange = executor.exchange

    leg = _make_leg(
        symbol="ALGO/USD",
        side="sell",
        amount_in=1.0,
        amount_out=1.0,
        traded_quantity=1.0,
    )

    previous_state = {"aggregated_metrics": {"amount_out": 1.0}}
    current_state = {
        "leg": leg,
        "orders": [],
        "metrics": [
            {
                "amount_in": 0.51,
                "amount_out": 0.51,
                "amount_out_without_fee": 0.51,
                "traded_quantity": 0.51,
                "fee_breakdown": {},
            }
        ],
        "submit_durations": [],
        "fill_durations": [],
    }
    current_state["aggregated_metrics"] = executor._combine_execution_metrics(
        current_state["metrics"]
    )

    reconciled = executor._reconcile_staggered_gap(previous_state, current_state)

    assert exchange.orders == []
    assert math.isclose(reconciled["amount_in"], 0.51, rel_tol=0, abs_tol=1e-12)
    assert math.isclose(reconciled["amount_out"], 0.51, rel_tol=0, abs_tol=1e-12)
    assert current_state["orders"] == []


def test_prepare_staggered_factors_disable_when_residual_below_min() -> None:
    class _MinExchange(_DummyExchange):
        def get_min_trade_values(self, symbol: str) -> Dict[str, float]:
            return {"amount": 5.0, "cost": 20.0}

    executor = TriangularArbitrageExecutor(
        _MinExchange(),
        dry_run=False,
        staggered_leg_delay=0.0,
        staggered_slippage_assumption=[0.1],
    )

    opportunity = _make_staggered_opportunity()
    overrides, expected = executor._prepare_staggered_factors(opportunity)

    assert overrides.get(1) == pytest.approx(1.0)
    assert expected[1]["assumption"] == pytest.approx(0.0)
    assert "residual_disabled_reason" in expected[1]


def test_prepare_staggered_factors_retains_underfill_above_min() -> None:
    class _LooseExchange(_DummyExchange):
        def get_min_trade_values(self, symbol: str) -> Dict[str, float]:
            return {"amount": 0.01, "cost": 0.01}

    executor = TriangularArbitrageExecutor(
        _LooseExchange(),
        dry_run=False,
        staggered_leg_delay=0.0,
        staggered_slippage_assumption=[0.1],
    )

    opportunity = _make_staggered_opportunity()
    overrides, expected = executor._prepare_staggered_factors(opportunity)

    assert 1 not in overrides
    assert expected[1]["assumption"] == pytest.approx(0.1)
    assert expected[1]["input"] == pytest.approx(10.0)
