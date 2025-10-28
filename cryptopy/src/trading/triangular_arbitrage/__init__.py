"""Triangular arbitrage toolkit for the crypto trading simulator."""
from cryptopy.src.trading.triangular_arbitrage.calculator import TriangularArbitrageCalculator
from cryptopy.src.trading.triangular_arbitrage.exchange import ExchangeConnection
from cryptopy.src.trading.triangular_arbitrage.exceptions import (
    ExchangeRequestTimeout,
    InsufficientLiquidityError,
)
from cryptopy.src.trading.triangular_arbitrage.executor import TriangularArbitrageExecutor
from cryptopy.src.trading.triangular_arbitrage.models import (
    OrderBookSnapshot,
    PriceSnapshot,
    RouteEvaluationStats,
    TriangularOpportunity,
    TriangularRoute,
    TriangularTradeLeg,
)
from cryptopy.src.trading.triangular_arbitrage.slippage import (
    LegSlippage,
    PrecisionAdapter,
    SlippageSimulation,
    simulate_opportunity_with_order_books,
)

__all__ = [
    "ExchangeConnection",
    "ExchangeRequestTimeout",
    "InsufficientLiquidityError",
    "OrderBookSnapshot",
    "PriceSnapshot",
    "RouteEvaluationStats",
    "TriangularArbitrageCalculator",
    "TriangularArbitrageExecutor",
    "TriangularOpportunity",
    "TriangularRoute",
    "TriangularTradeLeg",
    "LegSlippage",
    "PrecisionAdapter",
    "SlippageSimulation",
    "simulate_opportunity_with_order_books",
]
