"""Helpers to act upon profitable arbitrage opportunities."""
from __future__ import annotations

import asyncio
import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from cryptopy.src.trading.triangular_arbitrage.models import TriangularOpportunity


class TriangularArbitrageExecutor:
    """Executes arbitrage opportunities by placing orders on the exchange."""

    def __init__(
        self,
        exchange: Any,
        *,
        dry_run: bool = True,
        trade_log_path: Optional[Union[str, Path]] = None,
    ) -> None:
        self.exchange = exchange
        self.dry_run = dry_run
        self.trade_log_path = Path(trade_log_path) if trade_log_path else None

    def execute(self, opportunity: TriangularOpportunity) -> List[Dict[str, Any]]:
        if opportunity.profit <= 0:
            raise ValueError("Cannot execute an unprofitable opportunity")

        orders: List[Dict[str, Any]] = []
        for leg in opportunity.trades:
            order = self.exchange.create_market_order(
                leg.symbol,
                leg.side,
                leg.traded_quantity,
                test_order=self.dry_run,
            )
            orders.append(order)

        if self.trade_log_path is not None:
            self._log_trades(opportunity, orders)

        return orders

    async def execute_async(self, opportunity: TriangularOpportunity) -> List[Dict[str, Any]]:
        return await asyncio.to_thread(self.execute, opportunity)

    def _log_trades(
        self,
        opportunity: TriangularOpportunity,
        orders: Sequence[Dict[str, Any]],
    ) -> None:
        assert self.trade_log_path is not None
        log_path = self.trade_log_path
        log_path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "timestamp",
            "symbol",
            "side",
            "amount_in",
            "amount_out",
            "average_price",
            "fee_paid",
            "traded_quantity",
            "starting_amount",
            "final_amount",
            "profit",
            "profit_percentage",
            "dry_run",
            "order_id",
        ]

        row_time = datetime.now(timezone.utc).isoformat()
        file_exists = log_path.exists()
        with log_path.open("a", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()

            for leg, order in zip(opportunity.trades, orders):
                writer.writerow(
                    {
                        "timestamp": row_time,
                        "symbol": leg.symbol,
                        "side": leg.side,
                        "amount_in": leg.amount_in,
                        "amount_out": leg.amount_out,
                        "average_price": leg.average_price,
                        "fee_paid": leg.fee_paid,
                        "traded_quantity": leg.traded_quantity,
                        "starting_amount": opportunity.starting_amount,
                        "final_amount": opportunity.final_amount,
                        "profit": opportunity.profit,
                        "profit_percentage": opportunity.profit_percentage,
                        "dry_run": self.dry_run,
                        "order_id": order.get("id"),
                    }
                )
