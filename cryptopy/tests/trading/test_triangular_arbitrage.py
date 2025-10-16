from dataclasses import dataclass
import importlib.util
import pathlib
import sys
from types import SimpleNamespace

import csv
import pytest

MODULE_PATH = pathlib.Path(__file__).resolve().parents[2] / "src" / "trading" / "triangular_arbitrage" / "__init__.py"
spec = importlib.util.spec_from_file_location("triangular_arbitrage", MODULE_PATH)
triangular_arbitrage = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = triangular_arbitrage
assert spec.loader is not None
spec.loader.exec_module(triangular_arbitrage)

RUNNER_MODULE_PATH = (
    pathlib.Path(__file__).resolve().parents[2]
    / "scripts"
    / "trading"
    / "run_triangular_arbitrage.py"
)
runner_spec = importlib.util.spec_from_file_location(
    "triangular_arbitrage_runner",
    RUNNER_MODULE_PATH,
)
runner_module = importlib.util.module_from_spec(runner_spec)
sys.modules[runner_spec.name] = runner_module
assert runner_spec.loader is not None
runner_spec.loader.exec_module(runner_module)

InsufficientLiquidityError = triangular_arbitrage.InsufficientLiquidityError
PriceSnapshot = triangular_arbitrage.PriceSnapshot
TriangularArbitrageCalculator = triangular_arbitrage.TriangularArbitrageCalculator
TriangularArbitrageExecutor = triangular_arbitrage.TriangularArbitrageExecutor
TriangularOpportunity = triangular_arbitrage.TriangularOpportunity
TriangularRoute = triangular_arbitrage.TriangularRoute

filter_markets_for_triangular_routes = (
    runner_module.filter_markets_for_triangular_routes
)
generate_triangular_routes = runner_module.generate_triangular_routes


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
    assert opportunity.final_amount_without_fees >= opportunity.final_amount
    assert pytest.approx(opportunity.fee_impact, rel=1e-9) == pytest.approx(
        opportunity.profit_without_fees - opportunity.profit, rel=1e-9
    )
    assert pytest.approx(opportunity.trades[0].fee_rate, rel=1e-9) == exchange.fee


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
    assert all("fee_rate" in row for row in rows)
    assert all(
        pytest.approx(float(row["fee_rate"]), rel=1e-9) == exchange.fee for row in rows
    )
    assert all(float(row["fee_impact"]) >= 0 for row in rows)


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
    assert stats.best_opportunity is not None


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
    assert stats.best_opportunity is None


def test_best_opportunity_reported_when_below_threshold():
    exchange = MockExchange()
    calculator = TriangularArbitrageCalculator(exchange)  # type: ignore[arg-type]
    route = TriangularRoute(("BTC/USD", "ETH/BTC", "ETH/USD"), "USD")

    prices = {
        "BTC/USD": make_price_snapshot("BTC/USD", bid=101.0, ask=100.0),
        "ETH/BTC": make_price_snapshot("ETH/BTC", bid=0.55, ask=0.5),
        "ETH/USD": make_price_snapshot("ETH/USD", bid=0.55, ask=0.56),
    }

    opportunities, stats = calculator.find_profitable_routes(
        [route],
        prices,
        starting_amount=1000.0,
        min_profit_percentage=0.1,
    )

    assert opportunities == []
    assert stats.rejected_by_profit == 1
    assert stats.best_opportunity is not None
    assert stats.best_opportunity.route == route


def test_market_filter_allows_matching_settle_currency():
    markets = {
        "ETH/USDT": {
            "active": True,
            "base": "ETH",
            "quote": "USDT",
            "settle": "USDT",
        }
    }

    filtered, stats = filter_markets_for_triangular_routes(
        markets,
        starting_currencies=["USDT"],
    )

    assert "ETH/USDT" in filtered
    assert stats.total == 1
    assert stats.retained == 1
    assert stats.skipped == 0


def test_market_filter_rejects_unmatched_settle_currency():
    markets = {
        "ETH/USD": {
            "active": True,
            "base": "ETH",
            "quote": "USD",
            "settle": "BTC",
        }
    }

    filtered_usdt, stats_usdt = filter_markets_for_triangular_routes(
        markets,
        starting_currencies=["USDT"],
    )

    assert "ETH/USD" not in filtered_usdt
    assert stats_usdt.skipped_by_reason.get("derivative_settlement") == 1

    filtered_btc, stats_btc = filter_markets_for_triangular_routes(
        markets,
        starting_currencies=["BTC"],
    )

    assert "ETH/USD" in filtered_btc
    assert stats_btc.retained == 1


def test_market_filter_respects_asset_filter():
    markets = {
        "ETH/USD": {
            "active": True,
            "base": "ETH",
            "quote": "USD",
        },
        "ETH/BTC": {
            "active": True,
            "base": "ETH",
            "quote": "BTC",
        },
        "LTC/USD": {
            "active": True,
            "base": "LTC",
            "quote": "USD",
        },
    }

    filtered, stats = filter_markets_for_triangular_routes(
        markets,
        starting_currencies=["USD"],
        asset_filter=["USD", "ETH"],
    )

    assert "ETH/USD" in filtered
    assert "ETH/BTC" not in filtered
    assert "LTC/USD" not in filtered
    assert stats.retained == 1
    assert stats.skipped_by_reason.get("asset_filter") == 2


def test_generate_routes_respects_asset_filter():
    markets = {
        "BTC/USD": {"active": True, "base": "BTC", "quote": "USD"},
        "ETH/BTC": {"active": True, "base": "ETH", "quote": "BTC"},
        "ETH/USD": {"active": True, "base": "ETH", "quote": "USD"},
        "LTC/USD": {"active": True, "base": "LTC", "quote": "USD"},
    }

    all_routes = generate_triangular_routes(
        markets,
        starting_currencies=["USD"],
    )

    filtered_routes = generate_triangular_routes(
        markets,
        starting_currencies=["USD"],
        allowed_assets=["USD", "ETH"],
    )

    assert any(route.symbols == ("BTC/USD", "ETH/BTC", "ETH/USD") for route in all_routes)
    assert filtered_routes == []


def test_generate_routes_include_reverse_cycles():
    markets = {
        "BTC/USD": {"active": True, "base": "BTC", "quote": "USD"},
        "ETH/BTC": {"active": True, "base": "ETH", "quote": "BTC"},
        "ETH/USD": {"active": True, "base": "ETH", "quote": "USD"},
    }

    routes = generate_triangular_routes(markets, starting_currencies=["USD"])

    assert any(route.symbols == ("BTC/USD", "ETH/BTC", "ETH/USD") for route in routes)
    assert any(route.symbols == ("ETH/USD", "ETH/BTC", "BTC/USD") for route in routes)


def test_load_credentials_from_config_supports_snake_case(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text(
        """
kraken_websocket:
  api_key: KEY123
  api_secret: SECRET456
        """.strip()
    )

    credentials = runner_module.load_credentials_from_config("kraken", str(config))

    assert credentials["apiKey"] == "KEY123"
    assert credentials["secret"] == "SECRET456"


def test_exchange_connection_applies_credentials_for_orders(monkeypatch):
    from cryptopy.src.trading.triangular_arbitrage import exchange as exchange_module

    class DummyExchange:
        def __init__(self, config):
            self.config = config
            self.apiKey = config.get("apiKey")
            self.secret = config.get("secret")
            self.options = {}
            self.has = {"fetchTradingFees": False, "fetchTradingFee": False}
            self.fees = {"trading": {"taker": 0.001}}
            self.last_order = None
            self.balance_calls = 0

        def load_markets(self):
            return {}

        def fetch_trading_fees(self):
            return {}

        def fetch_balance(self):
            self.balance_calls += 1
            return {"total": {}}

        def create_order(self, symbol, order_type, side, amount, params=None):
            if not getattr(self, "apiKey", None):
                raise AssertionError("apiKey not set on client")
            if not getattr(self, "secret", None):
                raise AssertionError("secret not set on client")
            self.last_order = {
                "symbol": symbol,
                "type": order_type,
                "side": side,
                "amount": amount,
                "apiKey": self.apiKey,
                "secret": self.secret,
            }
            return {"id": "dummy-order"}

        def close(self):
            pass

    monkeypatch.setattr(exchange_module, "ccxt", SimpleNamespace(dummy=DummyExchange))
    monkeypatch.setattr(exchange_module, "ccxtpro", None)
    monkeypatch.setattr(exchange_module, "CcxtNotSupported", Exception)

    connection = exchange_module.ExchangeConnection(
        "dummy",
        credentials={"apiKey": "KEY123", "secret": "SECRET456"},
        use_testnet=False,
        enable_websocket=False,
        make_trades=True,
    )

    result = connection.create_market_order("BTC/USD", "buy", 1.0)

    assert result == {"id": "dummy-order"}
    assert connection.rest_client.last_order is not None
    assert connection.rest_client.last_order["apiKey"] == "KEY123"
    assert connection.rest_client.last_order["secret"] == "SECRET456"
    assert connection.rest_client.balance_calls == 1


def test_exchange_connection_raises_when_credentials_invalid(monkeypatch):
    from cryptopy.src.trading.triangular_arbitrage import exchange as exchange_module

    class DummyAuthError(Exception):
        pass

    class DummyExchange:
        def __init__(self, config):
            self.config = config
            self.apiKey = config.get("apiKey")
            self.secret = config.get("secret")
            self.options = {}
            self.has = {"fetchTradingFees": False, "fetchTradingFee": False}
            self.fees = {"trading": {"taker": 0.001}}

        def load_markets(self):
            return {}

        def fetch_trading_fees(self):
            return {}

        def fetch_balance(self):
            raise DummyAuthError("invalid credentials")

    monkeypatch.setattr(exchange_module, "ccxt", SimpleNamespace(dummy=DummyExchange))
    monkeypatch.setattr(exchange_module, "ccxtpro", None)
    monkeypatch.setattr(exchange_module, "CcxtNotSupported", Exception)
    monkeypatch.setattr(exchange_module, "CcxtAuthenticationError", DummyAuthError)

    with pytest.raises(DummyAuthError, match="Authentication failed when calling fetch_balance"):
        exchange_module.ExchangeConnection(
            "dummy",
            credentials={"apiKey": "BAD", "secret": "CREDENTIALS"},
            use_testnet=False,
            enable_websocket=False,
            make_trades=True,
        )

