"""Helpers to act upon profitable arbitrage opportunities."""
from __future__ import annotations

import asyncio
import csv
import json
import logging
import time
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

    _ORDER_STATUS_ATTEMPTS = 5
    _ORDER_STATUS_DELAY = 0.2
    _ORDER_COMPLETION_TIMEOUT = 15.0
    _ORDER_POLL_INTERVAL = 0.5
    _TRADE_HISTORY_LIMIT = 10

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
        execution_records: List[Tuple[TriangularTradeLeg, Dict[str, Any], Dict[str, Any]]] = []
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
            if self.dry_run:
                orders.append(order)
                metrics = self._planned_execution_metrics(leg)
                execution_records.append((leg, order, metrics))
                available_amount = leg.amount_out * scale
                continue

            finalised_order, metrics = self._finalise_order_execution(order, leg)
            orders.append(finalised_order)
            execution_records.append((leg, finalised_order, metrics))

            available_amount = metrics["amount_out"]

        actual_final_amount = (
            opportunity.final_amount if self.dry_run else available_amount
        )
        actual_final_without_fees = (
            opportunity.final_amount_without_fees
            if self.dry_run
            else (
                execution_records[-1][2]["amount_out_without_fee"]
                if execution_records
                else opportunity.starting_amount
            )
        )
        actual_profit = actual_final_amount - opportunity.starting_amount
        actual_profit_without_fees = (
            actual_final_without_fees - opportunity.starting_amount
        )
        actual_profit_percentage = (
            (actual_final_amount / opportunity.starting_amount - 1.0) * 100.0
            if opportunity.starting_amount
            else 0.0
        )
        actual_fee_impact = actual_final_without_fees - actual_final_amount

        if self.trade_log_path is not None:
            self._log_trades(
                opportunity,
                execution_records,
                actual_final_amount,
                actual_final_without_fees,
                actual_profit,
                actual_profit_without_fees,
                actual_profit_percentage,
                actual_fee_impact,
            )

        return orders

    async def execute_async(self, opportunity: TriangularOpportunity) -> List[Dict[str, Any]]:
        return await asyncio.to_thread(self.execute, opportunity)

    def _log_trades(
        self,
        opportunity: TriangularOpportunity,
        execution_records: Sequence[Tuple[TriangularTradeLeg, Dict[str, Any], Dict[str, Any]]],
        actual_final_amount: float,
        actual_final_without_fees: float,
        actual_profit: float,
        actual_profit_without_fees: float,
        actual_profit_percentage: float,
        actual_fee_impact: float,
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
            "actual_amount_in",
            "actual_amount_out",
            "actual_amount_out_without_fee",
            "average_price",
            "fee_rate",
            "fee_paid",
            "traded_quantity",
            "actual_traded_quantity",
            "actual_fee_breakdown",
            "starting_amount",
            "final_amount",
            "final_amount_without_fees",
            "profit",
            "profit_without_fees",
            "fee_impact",
            "profit_percentage",
            "actual_final_amount",
            "actual_final_amount_without_fees",
            "actual_profit",
            "actual_profit_without_fees",
            "actual_fee_impact",
            "actual_profit_percentage",
            "dry_run",
            "order_id",
        ]

        row_time = datetime.now(timezone.utc).isoformat()
        file_exists = log_path.exists()
        with log_path.open("a", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()

            for leg, order, metrics in execution_records:
                writer.writerow(
                    {
                        "timestamp": row_time,
                        "symbol": leg.symbol,
                        "side": leg.side,
                        "amount_in": leg.amount_in,
                        "amount_out": leg.amount_out,
                        "amount_out_without_fee": leg.amount_out_without_fee,
                        "actual_amount_in": metrics.get("amount_in", leg.amount_in),
                        "actual_amount_out": metrics.get("amount_out", leg.amount_out),
                        "actual_amount_out_without_fee": metrics.get(
                            "amount_out_without_fee", leg.amount_out_without_fee
                        ),
                        "average_price": leg.average_price,
                        "fee_rate": leg.fee_rate,
                        "fee_paid": leg.fee_paid,
                        "traded_quantity": leg.traded_quantity,
                        "actual_traded_quantity": metrics.get(
                            "traded_quantity", leg.traded_quantity
                        ),
                        "actual_fee_breakdown": json.dumps(
                            metrics.get("fee_breakdown", {}), sort_keys=True
                        ),
                        "starting_amount": opportunity.starting_amount,
                        "final_amount": opportunity.final_amount,
                        "final_amount_without_fees": opportunity.final_amount_without_fees,
                        "profit": opportunity.profit,
                        "profit_without_fees": opportunity.profit_without_fees,
                        "fee_impact": opportunity.fee_impact,
                        "profit_percentage": opportunity.profit_percentage,
                        "actual_final_amount": actual_final_amount,
                        "actual_final_amount_without_fees": actual_final_without_fees,
                        "actual_profit": actual_profit,
                        "actual_profit_without_fees": actual_profit_without_fees,
                        "actual_fee_impact": actual_fee_impact,
                        "actual_profit_percentage": actual_profit_percentage,
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

    @staticmethod
    def _planned_execution_metrics(leg: TriangularTradeLeg) -> Dict[str, float]:
        return {
            "amount_in": float(leg.amount_in),
            "amount_out": float(leg.amount_out),
            "amount_out_without_fee": float(leg.amount_out_without_fee),
            "traded_quantity": float(leg.traded_quantity),
        }

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

    def _extract_execution_metrics(
        self,
        order: Dict[str, Any],
        leg: TriangularTradeLeg,
    ) -> Dict[str, Any]:
        metrics = self._planned_execution_metrics(leg)

        base, quote = leg.symbol.split("/")
        filled = self._safe_float(order.get("filled"))
        if filled is None:
            filled = self._safe_float(order.get("amount"))

        average_price = self._safe_float(order.get("average") or order.get("price"))
        cost = self._safe_float(order.get("cost"))
        if cost is None and filled is not None and average_price is not None:
            cost = average_price * filled

        fee_totals: Dict[str, float] = {}
        for fee in self._normalised_fees(order):
            currency = fee.get("currency")
            cost_value = self._safe_float(fee.get("cost"))
            if not currency or cost_value is None:
                continue
            currency_code = str(currency).upper()
            fee_totals[currency_code] = fee_totals.get(currency_code, 0.0) + cost_value

        base_fee = fee_totals.get(base) or fee_totals.get(base.upper(), 0.0)
        quote_fee = fee_totals.get(quote) or fee_totals.get(quote.upper(), 0.0)
        base_fee = float(base_fee or 0.0)
        quote_fee = float(quote_fee or 0.0)

        traded_quantity = filled if filled is not None else metrics["traded_quantity"]
        if traded_quantity is None:  # pragma: no cover - defensive guard
            traded_quantity = metrics["traded_quantity"]

        if leg.side == "buy":
            amount_in = (cost if cost is not None else metrics["amount_in"]) + quote_fee
            amount_out_without_fee = traded_quantity
            amount_out = max(traded_quantity - base_fee, 0.0)
        else:
            amount_in = traded_quantity + base_fee
            amount_out_without_fee = cost if cost is not None else metrics["amount_out_without_fee"]
            amount_out = max(amount_out_without_fee - quote_fee, 0.0)

        metrics.update(
            {
                "amount_in": float(amount_in),
                "amount_out": float(amount_out),
                "amount_out_without_fee": float(amount_out_without_fee),
                "traded_quantity": float(traded_quantity),
                "fee_breakdown": {k: float(v) for k, v in fee_totals.items()},
            }
        )

        return metrics

    def _finalise_order_execution(
        self,
        order: Dict[str, Any],
        leg: TriangularTradeLeg,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Ensure an order is fully executed and return its realised metrics."""

        order_id = self._extract_order_id(order)
        if not order_id:
            raise RuntimeError(
                f"Exchange response for {leg.side} {leg.symbol} order is missing an id"
            )

        merged_order = {**order}
        completion = self._wait_for_order_completion(order_id, leg.symbol)
        if completion:
            merged_order.update(completion)

        resolved = self._resolve_order_payload(merged_order, leg, order_id)
        metrics = self._extract_execution_metrics(resolved, leg)
        realised_amount = metrics.get("amount_out")
        if realised_amount is None:
            raise RuntimeError(
                f"Unable to determine realised amount for order {order_id} on {leg.symbol}"
            )

        return resolved, metrics

    def _wait_for_order_completion(self, order_id: str, symbol: str) -> Dict[str, Any]:
        deadline = time.time() + self._ORDER_COMPLETION_TIMEOUT
        latest_payload: Dict[str, Any] = {}

        while time.time() < deadline:
            try:
                latest = self.exchange.fetch_order(order_id, symbol)
            except Exception as exc:  # pragma: no cover - ccxt/network failure
                logger.debug(
                    "Unable to poll order %s for %s: %s",
                    order_id,
                    symbol,
                    exc,
                )
                break

            if latest:
                latest_payload.update(latest)

            status = str(latest_payload.get("status") or "").lower()
            filled = self._safe_float(latest_payload.get("filled")) or 0.0
            remaining = self._safe_float(latest_payload.get("remaining"))

            if status in {"closed", "canceled", "cancelled", "rejected"}:
                break
            if remaining is not None and remaining <= 0:
                break
            if status == "open" and filled > 0 and remaining is None:
                # Some exchanges omit ``remaining`` when fully filled.
                break

            time.sleep(self._ORDER_POLL_INTERVAL)

        return latest_payload

    def _resolve_order_payload(
        self,
        order: Dict[str, Any],
        leg: TriangularTradeLeg,
        order_id: str,
    ) -> Dict[str, Any]:
        """Enrich an order payload with realised execution details when missing."""

        if self._extract_realised_amount(order, leg) is not None:
            return order

        refreshed = order
        for attempt in range(self._ORDER_STATUS_ATTEMPTS):
            try:
                latest = self.exchange.fetch_order(order_id, leg.symbol)
            except Exception as exc:  # pragma: no cover - ccxt/network failure
                logger.debug(
                    "Unable to fetch order %s for %s on attempt %s: %s",
                    order_id,
                    leg.symbol,
                    attempt + 1,
                    exc,
                )
                break
            if latest:
                refreshed = {**refreshed, **latest}
            if self._extract_realised_amount(refreshed, leg) is not None:
                return refreshed
            time.sleep(self._ORDER_STATUS_DELAY)

        try:
            trades = self.exchange.fetch_my_trades(
                leg.symbol,
                limit=self._TRADE_HISTORY_LIMIT,
            )
        except Exception as exc:  # pragma: no cover - ccxt/network failure
            logger.debug(
                "Unable to fetch trades for %s when resolving order %s: %s",
                leg.symbol,
                order_id,
                exc,
            )
            return refreshed

        matched_trades = [
            trade
            for trade in trades
            if self._extract_order_id(trade) == order_id or trade.get("order") == order_id
        ]
        if not matched_trades:
            return refreshed

        augmented = self._augment_order_with_trades(refreshed, matched_trades, leg)
        return augmented

    def _augment_order_with_trades(
        self,
        order: Dict[str, Any],
        trades: Sequence[Dict[str, Any]],
        leg: TriangularTradeLeg,
    ) -> Dict[str, Any]:
        total_filled = sum(self._safe_float(trade.get("amount")) or 0.0 for trade in trades)
        total_cost = sum(self._safe_float(trade.get("cost")) or 0.0 for trade in trades)

        fees: List[Dict[str, Any]] = []
        for trade in trades:
            fee_field = trade.get("fee")
            if isinstance(fee_field, dict):
                fees.append(fee_field)
            elif isinstance(fee_field, list):
                fees.extend(f for f in fee_field if isinstance(f, dict))

        existing_fees = list(self._normalised_fees(order))
        augmented: Dict[str, Any] = {**order}
        if total_filled and not self._safe_float(augmented.get("filled")):
            augmented["filled"] = total_filled
        if total_cost and not self._safe_float(augmented.get("cost")):
            augmented["cost"] = total_cost

        if fees:
            merged_fees = list(existing_fees)
            for fee in fees:
                if fee not in merged_fees:
                    merged_fees.append(fee)
            if len(merged_fees) == 1:
                augmented["fee"] = merged_fees[0]
                augmented.pop("fees", None)
            else:
                augmented["fees"] = merged_fees

        augmented.setdefault("trades", trades)

        # When trades expose the quantities in the quote currency for buy orders,
        # ensure ``amount`` mirrors the filled base asset volume.
        if leg.side == "buy" and total_filled:
            augmented.setdefault("amount", total_filled)
        elif leg.side == "sell" and total_filled:
            augmented.setdefault("amount", total_filled)

        return augmented

    @staticmethod
    def _extract_order_id(payload: Dict[str, Any]) -> Optional[str]:
        order_id = payload.get("id") or payload.get("order")
        if order_id:
            return str(order_id)
        info = payload.get("info")
        if isinstance(info, dict):
            nested_id = info.get("id") or info.get("orderId") or info.get("order_id")
            if nested_id:
                return str(nested_id)
        return None

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
