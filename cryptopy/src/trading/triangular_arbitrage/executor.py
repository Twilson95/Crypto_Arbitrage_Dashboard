"""Helpers to act upon profitable arbitrage opportunities."""
from __future__ import annotations

import asyncio
import csv
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from cryptopy.src.trading.triangular_arbitrage.models import (
    TriangularOpportunity,
    TriangularTradeLeg,
)


logger = logging.getLogger(__name__)


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
        available_amount = opportunity.starting_amount

        for leg in opportunity.trades:
            amount, scale = self._determine_order_amount(leg, available_amount)

            if amount <= 0:
                raise ValueError(
                    f"Calculated non-positive trade amount ({amount}) for leg {leg.symbol}"
                )

            order = self.exchange.create_market_order(
                leg.symbol,
                leg.side,
                amount,
                test_order=self.dry_run,
            )
            orders.append(order)

            if self.dry_run:
                available_amount = leg.amount_out * scale
                continue

            realised_amount = self._extract_realised_amount(order, leg)
            if realised_amount is None:
                realised_amount = leg.amount_out * scale
                logger.debug(
                    "Falling back to simulated amount for %s leg %s due to incomplete order payload",
                    leg.side,
                    leg.symbol,
                )

            if leg.side == "buy":
                # We spent the previously held currency; the new balance is whatever was acquired.
                available_amount = realised_amount
            else:
                available_amount = realised_amount

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
            "amount_out_without_fee",
            "average_price",
            "fee_rate",
            "fee_paid",
            "traded_quantity",
            "starting_amount",
            "final_amount",
            "final_amount_without_fees",
            "profit",
            "profit_without_fees",
            "fee_impact",
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
                        "amount_out_without_fee": leg.amount_out_without_fee,
                        "average_price": leg.average_price,
                        "fee_rate": leg.fee_rate,
                        "fee_paid": leg.fee_paid,
                        "traded_quantity": leg.traded_quantity,
                        "starting_amount": opportunity.starting_amount,
                        "final_amount": opportunity.final_amount,
                        "final_amount_without_fees": opportunity.final_amount_without_fees,
                        "profit": opportunity.profit,
                        "profit_without_fees": opportunity.profit_without_fees,
                        "fee_impact": opportunity.fee_impact,
                        "profit_percentage": opportunity.profit_percentage,
                        "dry_run": self.dry_run,
                        "order_id": order.get("id"),
                    }
                )

    def _determine_order_amount(
        self,
        leg: TriangularTradeLeg,
        available_amount: float,
    ) -> Tuple[float, float]:
        """Return the amount to submit for the current leg.

        ``available_amount`` represents the quantity of the currency currently held
        before executing ``leg``. This value is derived from live order execution
        whenever trading is enabled. The calculator's simulated amounts are used as
        a fallback when the exchange response does not expose the realised volume.
        """

        desired_amount = float(leg.traded_quantity)

        if self.dry_run:
            return desired_amount, 1.0

        base, quote = leg.symbol.split("/")
        if leg.side == "sell":
            # We are selling the currency currently held (the base asset).
            if available_amount <= 0:
                logger.debug("No %s available for sell leg %s", base, leg.symbol)
                return 0.0, 0.0
            if available_amount < desired_amount:
                logger.debug(
                    "Reducing sell amount for %s from %s to available balance %s",  # noqa: G200
                    leg.symbol,
                    desired_amount,
                    available_amount,
                )
                scale = available_amount / desired_amount if desired_amount else 0.0
                return available_amount, scale
            return desired_amount, 1.0

        # Buying the base asset using the currently held quote currency.
        required_quote = float(leg.amount_in)
        if available_amount <= 0:
            logger.debug("No %s available to buy %s", quote, leg.symbol)
            return 0.0, 0.0
        if required_quote <= 0:
            return desired_amount, 1.0

        if available_amount < required_quote:
            scale = available_amount / required_quote
            adjusted_amount = desired_amount * max(scale, 0.0)
            logger.debug(
                "Reducing buy amount for %s from %s to %s due to limited %s",
                leg.symbol,
                desired_amount,
                adjusted_amount,
                quote,
            )
            return adjusted_amount, max(scale, 0.0)

        return desired_amount, 1.0

    def _extract_realised_amount(
        self,
        order: Dict[str, Any],
        leg: TriangularTradeLeg,
    ) -> Optional[float]:
        """Extract the realised output amount from an executed order."""

        if self.dry_run:
            return leg.amount_out

        base, quote = leg.symbol.split("/")

        if leg.side == "buy":
            filled = self._safe_float(order.get("filled"))
            if filled is None:
                filled = self._safe_float(order.get("amount"))

            if filled is None:
                return None

            fees = self._normalised_fees(order)
            base_fee = sum(
                self._safe_float(fee.get("cost")) or 0.0
                for fee in fees
                if fee.get("currency") == base
            )
            realised = max(filled - base_fee, 0.0)
            return realised

        cost = self._safe_float(order.get("cost"))
        if cost is None:
            average_price = self._safe_float(order.get("average") or order.get("price"))
            filled = self._safe_float(order.get("filled"))
            if average_price is not None and filled is not None:
                cost = average_price * filled

        if cost is None:
            return None

        fees = self._normalised_fees(order)
        quote_fee = sum(
            self._safe_float(fee.get("cost")) or 0.0
            for fee in fees
            if fee.get("currency") == quote
        )
        realised = max(cost - quote_fee, 0.0)
        return realised

    @staticmethod
    def _normalised_fees(order: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
        """Return the order's fees as an iterable of dictionaries."""

        fees: List[Dict[str, Any]] = []
        fee_field = order.get("fee")
        if isinstance(fee_field, dict):
            fees.append(fee_field)
        fees_field = order.get("fees")
        if isinstance(fees_field, list):
            fees.extend(f for f in fees_field if isinstance(f, dict))
        return fees

    @staticmethod
    def _safe_float(value: Any) -> Optional[float]:
        try:
            if value is None:
                return None
            return float(value)
        except (TypeError, ValueError):
            return None
