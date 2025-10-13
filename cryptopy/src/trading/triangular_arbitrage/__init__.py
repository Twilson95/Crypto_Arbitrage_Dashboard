"""Triangular arbitrage toolkit for the crypto trading simulator."""
from .calculator import TriangularArbitrageCalculator
from .exchange import ExchangeConnection
from .exceptions import InsufficientLiquidityError
from .executor import TriangularArbitrageExecutor
from .models import (
    OrderBookSnapshot,
    TriangularOpportunity,
    TriangularRoute,
    TriangularTradeLeg,
)

__all__ = [
    "ExchangeConnection",
    "InsufficientLiquidityError",
    "OrderBookSnapshot",
    "TriangularArbitrageCalculator",
    "TriangularArbitrageExecutor",
    "TriangularOpportunity",
    "TriangularRoute",
    "TriangularTradeLeg",
]
