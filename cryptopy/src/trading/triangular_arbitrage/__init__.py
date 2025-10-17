"""Triangular arbitrage toolkit for the crypto trading simulator."""
from cryptopy.src.trading.triangular_arbitrage.calculator import TriangularArbitrageCalculator
from cryptopy.src.trading.triangular_arbitrage.exchange import ExchangeConnection
from cryptopy.src.trading.triangular_arbitrage.exceptions import InsufficientLiquidityError
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
    SlippageSimulation,
    simulate_opportunity_with_order_books,
)

__all__ = [
    "ExchangeConnection",
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
    "SlippageSimulation",
    "simulate_opportunity_with_order_books",
]
