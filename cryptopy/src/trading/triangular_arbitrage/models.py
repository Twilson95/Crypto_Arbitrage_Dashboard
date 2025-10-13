"""Data models used by the triangular arbitrage utilities."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class OrderBookSnapshot:
    """Normalised order book representation used by the simulator."""

    symbol: str
    bids: List[Tuple[float, float]]
    asks: List[Tuple[float, float]]
    timestamp: float = field(default_factory=lambda: time.time())

    @classmethod
    def from_ccxt(cls, symbol: str, order_book: Dict[str, Any]) -> "OrderBookSnapshot":
        """Create an :class:`OrderBookSnapshot` from raw CCXT data."""
        bids = sorted(order_book.get("bids", []), key=lambda level: level[0], reverse=True)
        asks = sorted(order_book.get("asks", []), key=lambda level: level[0])
        timestamp = order_book.get("timestamp")
        if timestamp:
            timestamp = float(timestamp) / 1000.0
        else:
            timestamp = time.time()
        return cls(symbol=symbol, bids=bids, asks=asks, timestamp=timestamp)

    def best_bid(self) -> Optional[Tuple[float, float]]:
        return self.bids[0] if self.bids else None

    def best_ask(self) -> Optional[Tuple[float, float]]:
        return self.asks[0] if self.asks else None


@dataclass(frozen=True)
class TriangularRoute:
    """Represents a trading route that can contain multiple legs."""

    symbols: Tuple[str, ...]
    starting_currency: str


@dataclass
class TriangularTradeLeg:
    """Details about a single executed leg in an arbitrage cycle."""

    symbol: str
    side: str
    amount_in: float
    amount_out: float
    average_price: float
    fee_paid: float
    traded_quantity: float


@dataclass
class TriangularOpportunity:
    """A potential (or simulated) triangular arbitrage opportunity."""

    route: TriangularRoute
    starting_amount: float
    final_amount: float
    trades: List[TriangularTradeLeg]

    @property
    def profit(self) -> float:
        return self.final_amount - self.starting_amount

    @property
    def profit_percentage(self) -> float:
        if self.starting_amount == 0:
            return 0.0
        return (self.final_amount / self.starting_amount - 1.0) * 100.0
