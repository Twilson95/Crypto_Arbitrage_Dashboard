"""Helpers to act upon profitable arbitrage opportunities."""
from __future__ import annotations

import asyncio
import csv
import json
import logging
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

from cryptopy.src.trading.triangular_arbitrage.models import (
    TriangularOpportunity,
    TriangularTradeLeg,
)

try:  # pragma: no cover - optional dependency when ccxt is unavailable in tests
    from ccxt.base.errors import InsufficientFunds as CcxtInsufficientFunds  # type: ignore
except Exception:  # pragma: no cover - fallback when ccxt is not installed
    CcxtInsufficientFunds = None  # type: ignore[assignment]


logger = logging.getLogger(__name__)


class _ProgressiveExecutionState:
    """Tracks orders and realised totals when streaming partial fills."""

    def __init__(self, opportunity: TriangularOpportunity) -> None:
        self.opportunity = opportunity
        self.orders: List[Dict[str, Any]] = []
        self.execution_records: List[
            Tuple[TriangularTradeLeg, Dict[str, Any], Dict[str, Any]]
        ] = []
        self.final_amount: float = 0.0
        self.final_amount_without_fees: float = 0.0


class TriangularArbitrageExecutor:
    """Executes arbitrage opportunities by placing orders on the exchange."""

    _ORDER_STATUS_ATTEMPTS = 5
    _ORDER_STATUS_DELAY = 0.2
    _ORDER_COMPLETION_TIMEOUT = 15.0
    _ORDER_POLL_INTERVAL_FAST = 0.1
    _ORDER_POLL_INTERVAL_SLOW = 0.5
    _ORDER_POLL_FAST_WINDOW = 3.0
    _REST_POLL_INTERVAL = 0.2
    _REST_POLL_INTERVAL_WEBSOCKET = 0.5
    _WEBSOCKET_IDLE_SLEEP = 0.02
    _TRADE_HISTORY_LIMIT = 10
    _INSUFFICIENT_FUNDS_RETRIES = 10
    PARTIAL_FILL_BEHAVIOURS = {"wait", "progressive"}

    def __init__(
        self,
        exchange: Any,
        *,
        dry_run: bool = True,
        trade_log_path: Optional[Union[str, Path]] = None,
        partial_fill_mode: str = "wait",
    ) -> None:
        if partial_fill_mode not in self.PARTIAL_FILL_BEHAVIOURS:
            raise ValueError(
                f"partial_fill_mode must be one of {sorted(self.PARTIAL_FILL_BEHAVIOURS)}"
            )

        self.exchange = exchange
        self.dry_run = dry_run
        self.trade_log_path = Path(trade_log_path) if trade_log_path else None
        self.partial_fill_mode = partial_fill_mode

    def execute(self, opportunity: TriangularOpportunity) -> List[Dict[str, Any]]:
        if opportunity.profit <= 0:
            raise ValueError("Cannot execute an unprofitable opportunity")

        execution_started = time.perf_counter()
        leg_timings: List[Dict[str, float]] = []

        if self.partial_fill_mode == "progressive" and not self.dry_run:
            orders, execution_records, final_amounts = self._execute_progressive(
                opportunity,
                leg_timings,
            )
        else:
            orders, execution_records, final_amounts = self._execute_serial(
                opportunity,
                leg_timings,
            )

        actual_final_amount, actual_final_without_fees = final_amounts
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

        self._log_route_slippage(
            opportunity,
            actual_final_amount,
            actual_final_without_fees,
            actual_profit_percentage,
        )

        total_execution_time = time.perf_counter() - execution_started
        if leg_timings:
            for entry in leg_timings:
                logger.info(
                    "Execution timing for %s %s: submit %.3fs fill %.3fs total %.3fs",
                    entry["side"],
                    entry["symbol"],
                    entry["submit_duration"],
                    entry["fill_duration"],
                    entry["total_duration"],
                )
            logger.info(
                "Execution timing summary for %s: %.3fs across %d leg(s)",
                " -> ".join(opportunity.route.symbols),
                total_execution_time,
                len(leg_timings),
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

    def _execute_serial(
        self,
        opportunity: TriangularOpportunity,
        timings: Optional[List[Dict[str, float]]] = None,
    ) -> Tuple[
        List[Dict[str, Any]],
        List[Tuple[TriangularTradeLeg, Dict[str, Any], Dict[str, Any]]],
        Tuple[float, float],
    ]:
        orders: List[Dict[str, Any]] = []
        execution_records: List[Tuple[TriangularTradeLeg, Dict[str, Any], Dict[str, Any]]] = []
        available_amount = opportunity.starting_amount

        for leg in opportunity.trades:
            leg_started_perf = time.perf_counter()
            submit_start = time.perf_counter()
            order, amount, scale = self._submit_leg_order(leg, available_amount)
            submit_duration = time.perf_counter() - submit_start
            if self.dry_run:
                orders.append(order)
                metrics = self._planned_execution_metrics(leg)
                execution_records.append((leg, order, metrics))
                available_amount = leg.amount_out * scale
                self._log_slippage_effect(leg, metrics)
                if timings is not None:
                    total_duration = time.perf_counter() - leg_started_perf
                    timings.append(
                        {
                            "symbol": leg.symbol,
                            "side": leg.side,
                            "submit_duration": submit_duration,
                            "fill_duration": 0.0,
                            "total_duration": total_duration,
                        }
                    )
                continue

            finalise_start = time.perf_counter()
            finalised_order, metrics = self._finalise_order_execution(order, leg)
            fill_duration = time.perf_counter() - finalise_start
            orders.append(finalised_order)
            execution_records.append((leg, finalised_order, metrics))
            self._log_slippage_effect(leg, metrics)

            available_amount = metrics["amount_out"]
            if timings is not None:
                total_duration = time.perf_counter() - leg_started_perf
                timings.append(
                    {
                        "symbol": leg.symbol,
                        "side": leg.side,
                        "submit_duration": submit_duration,
                        "fill_duration": fill_duration,
                        "total_duration": total_duration,
                    }
                )

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

        return orders, execution_records, (
            actual_final_amount,
            actual_final_without_fees,
        )

    def _execute_progressive(
        self,
        opportunity: TriangularOpportunity,
        timings: Optional[List[Dict[str, float]]] = None,
    ) -> Tuple[
        List[Dict[str, Any]],
        List[Tuple[TriangularTradeLeg, Dict[str, Any], Dict[str, Any]]],
        Tuple[float, float],
    ]:
        state = _ProgressiveExecutionState(opportunity)

        if not opportunity.trades:
            return [], [], (opportunity.starting_amount, opportunity.starting_amount)

        self._execute_progressive_leg(
            opportunity,
            state,
            0,
            opportunity.starting_amount,
            timings,
        )

        final_amount = (
            state.final_amount
            if state.final_amount > 0
            else (
                state.execution_records[-1][2]["amount_out"]
                if state.execution_records
                else opportunity.starting_amount
            )
        )
        final_without_fee = (
            state.final_amount_without_fees
            if state.final_amount_without_fees > 0
            else (
                state.execution_records[-1][2]["amount_out_without_fee"]
                if state.execution_records
                else opportunity.starting_amount
            )
        )

        return state.orders, state.execution_records, (
            final_amount,
            final_without_fee,
        )

    def _execute_progressive_leg(
        self,
        opportunity: TriangularOpportunity,
        state: _ProgressiveExecutionState,
        index: int,
        available_amount: float,
        timings: Optional[List[Dict[str, float]]] = None,
    ) -> None:
        if index >= len(opportunity.trades):
            return

        leg = opportunity.trades[index]
        leg_started_perf = time.perf_counter()
        try:
            order, amount, scale = self._submit_leg_order(leg, available_amount)
        except ValueError:
            logger.debug(
                "Skipping leg %s because rounded order amount dropped below precision (available=%s)",
                leg.symbol,
                available_amount,
            )
            if timings is not None:
                total_duration = time.perf_counter() - leg_started_perf
                timings.append(
                    {
                        "symbol": leg.symbol,
                        "side": leg.side,
                        "submit_duration": total_duration,
                        "fill_duration": 0.0,
                        "total_duration": total_duration,
                    }
                )
            return

        submit_duration = time.perf_counter() - leg_started_perf

        if self.dry_run:
            metrics = self._planned_execution_metrics(leg)
            state.orders.append(order)
            state.execution_records.append((leg, order, metrics))
            self._log_slippage_effect(leg, metrics)
            if index + 1 < len(opportunity.trades):
                self._execute_progressive_leg(
                    opportunity,
                    state,
                    index + 1,
                    leg.amount_out * scale,
                    timings,
                )
            else:
                state.final_amount += metrics["amount_out"]
                state.final_amount_without_fees += metrics["amount_out_without_fee"]
            if timings is not None:
                total_duration = time.perf_counter() - leg_started_perf
                timings.append(
                    {
                        "symbol": leg.symbol,
                        "side": leg.side,
                        "submit_duration": total_duration,
                        "fill_duration": 0.0,
                        "total_duration": total_duration,
                    }
                )
            return

        order_id = self._extract_order_id(order)
        if not order_id:
            raise RuntimeError(
                f"Exchange response for {leg.side} {leg.symbol} order is missing an id"
            )

        latest_payload: Dict[str, Any] = {**order}
        seen_trades: set[str] = set()
        order_poll_started = time.perf_counter()
        last_rest_poll = time.perf_counter()

        while True:
            new_trades = self._fetch_new_trades_for_order(
                order_id,
                leg.symbol,
                seen_trades,
            )
            for trade in new_trades:
                chunk_metrics = self._extract_trade_metrics(trade, leg)
                if chunk_metrics["amount_out"] <= 0:
                    continue
                if index + 1 < len(opportunity.trades):
                    self._execute_progressive_leg(
                        opportunity,
                        state,
                        index + 1,
                        chunk_metrics["amount_out"],
                        timings,
                    )

            poll_interval = self._order_poll_interval(order_poll_started)
            ws_update, used_websocket = self._watch_order_update(
                order_id,
                leg.symbol,
                poll_interval,
            )
            if ws_update:
                latest_payload.update(ws_update)
                if self._order_complete(latest_payload):
                    break

            now = time.perf_counter()
            should_poll_rest = (not used_websocket) or (
                now - last_rest_poll >= self._rest_poll_interval(used_websocket)
            )
            try:
                latest = (
                    self.exchange.fetch_order(order_id, leg.symbol)
                    if should_poll_rest
                    else None
                )
                if should_poll_rest:
                    last_rest_poll = now
            except Exception as exc:  # pragma: no cover - ccxt/network failure
                logger.debug(
                    "Unable to poll order %s for %s while streaming fills: %s",
                    order_id,
                    leg.symbol,
                    exc,
                )
                latest = None

            if latest:
                latest_payload.update(latest)

            if self._order_complete(latest_payload):
                break

            if not used_websocket and latest is None and ws_update is None:
                time.sleep(poll_interval)
            elif used_websocket and latest is None and ws_update is None:
                time.sleep(min(self._WEBSOCKET_IDLE_SLEEP, poll_interval))

        residual_trades = self._fetch_new_trades_for_order(
            order_id,
            leg.symbol,
            seen_trades,
        )
        for trade in residual_trades:
            chunk_metrics = self._extract_trade_metrics(trade, leg)
            if chunk_metrics["amount_out"] <= 0:
                continue
            if index + 1 < len(opportunity.trades):
                self._execute_progressive_leg(
                    opportunity,
                    state,
                    index + 1,
                    chunk_metrics["amount_out"],
                    timings,
                )

        resolved_order = self._resolve_order_payload(latest_payload, leg, order_id)
        metrics = self._extract_execution_metrics(resolved_order, leg)

        state.orders.append(resolved_order)
        state.execution_records.append((leg, resolved_order, metrics))
        self._log_slippage_effect(leg, metrics)

        if index + 1 >= len(opportunity.trades):
            state.final_amount += metrics["amount_out"]
            state.final_amount_without_fees += metrics["amount_out_without_fee"]

        if timings is not None:
            total_duration = time.perf_counter() - leg_started_perf
            fill_duration = max(total_duration - submit_duration, 0.0)
            timings.append(
                {
                    "symbol": leg.symbol,
                    "side": leg.side,
                    "submit_duration": submit_duration,
                    "fill_duration": fill_duration,
                    "total_duration": total_duration,
                }
            )

    def _fetch_new_trades_for_order(
        self,
        order_id: str,
        symbol: str,
        seen: set[str],
    ) -> List[Dict[str, Any]]:
        try:
            trades = self.exchange.fetch_my_trades(
                symbol,
                limit=self._TRADE_HISTORY_LIMIT,
            )
        except Exception as exc:  # pragma: no cover - ccxt/network failure
            logger.debug(
                "Unable to fetch trades for %s when streaming order %s: %s",
                symbol,
                order_id,
                exc,
            )
            return []

        new_trades: List[Dict[str, Any]] = []
        for trade in trades:
            if self._extract_order_id(trade) != order_id and trade.get("order") != order_id:
                continue
            trade_key = self._identify_trade(trade)
            if trade_key in seen:
                continue
            seen.add(trade_key)
            new_trades.append(trade)

        new_trades.sort(key=lambda trade: trade.get("timestamp") or 0)
        return new_trades

    def _watch_order_update(
        self, order_id: str, symbol: str, timeout: float
    ) -> Tuple[Optional[Dict[str, Any]], bool]:
        watcher = getattr(self.exchange, "watch_order_via_websocket", None)
        if watcher is None:
            return None, False
        try:
            return watcher(order_id, symbol, timeout=timeout), True
        except Exception:
            logger.debug(
                "watch_order_via_websocket failed for %s %s", order_id, symbol, exc_info=True
            )
            return None, True

    def _order_poll_interval(self, started_at: float) -> float:
        elapsed = time.perf_counter() - started_at
        if elapsed < self._ORDER_POLL_FAST_WINDOW:
            return self._ORDER_POLL_INTERVAL_FAST
        return self._ORDER_POLL_INTERVAL_SLOW

    def _rest_poll_interval(self, used_websocket: bool) -> float:
        return (
            self._REST_POLL_INTERVAL_WEBSOCKET
            if used_websocket
            else self._REST_POLL_INTERVAL
        )

    def _identify_trade(self, trade: Dict[str, Any]) -> str:
        trade_id = trade.get("id") or trade.get("trade_id") or trade.get("tid")
        if trade_id:
            return str(trade_id)
        timestamp = trade.get("timestamp") or trade.get("datetime") or ""
        amount = trade.get("amount") or trade.get("filled") or ""
        cost = trade.get("cost") or ""
        return f"{timestamp}-{amount}-{cost}"

    def _extract_trade_metrics(
        self,
        trade: Dict[str, Any],
        leg: TriangularTradeLeg,
    ) -> Dict[str, float]:
        metrics = self._planned_execution_metrics(leg)

        base, quote = leg.symbol.split("/")
        traded_quantity = self._safe_float(trade.get("amount"))
        if traded_quantity is None:
            traded_quantity = self._safe_float(trade.get("filled"))
        traded_quantity = float(traded_quantity or 0.0)

        price = self._safe_float(trade.get("price"))
        cost = self._safe_float(trade.get("cost"))
        if cost is None and price is not None:
            cost = price * traded_quantity

        fee_totals: Dict[str, float] = {}
        for fee in self._normalised_trade_fees(trade):
            currency = fee.get("currency")
            fee_cost = self._safe_float(fee.get("cost"))
            if not currency or fee_cost is None:
                continue
            code = str(currency).upper()
            fee_totals[code] = fee_totals.get(code, 0.0) + fee_cost

        base_fee = fee_totals.get(base) or fee_totals.get(base.upper(), 0.0)
        quote_fee = fee_totals.get(quote) or fee_totals.get(quote.upper(), 0.0)
        base_fee = float(base_fee or 0.0)
        quote_fee = float(quote_fee or 0.0)

        if leg.side == "buy":
            amount_in = (cost if cost is not None else metrics["amount_in"]) + quote_fee
            amount_out_without_fee = traded_quantity
            amount_out = max(traded_quantity - base_fee, 0.0)
        else:
            amount_in = traded_quantity + base_fee
            amount_out_without_fee = cost if cost is not None else metrics["amount_out_without_fee"]
            amount_out = max((amount_out_without_fee or 0.0) - quote_fee, 0.0)

        metrics.update(
            {
                "amount_in": float(amount_in),
                "amount_out": float(amount_out),
                "amount_out_without_fee": float(amount_out_without_fee or 0.0),
                "traded_quantity": float(traded_quantity),
                "fee_breakdown": {k: float(v) for k, v in fee_totals.items()},
            }
        )

        return metrics

    @staticmethod
    def _normalised_trade_fees(trade: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
        fees: List[Dict[str, Any]] = []
        fee_field = trade.get("fee")
        if isinstance(fee_field, dict):
            fees.append(fee_field)
        elif isinstance(fee_field, list):
            fees.extend(f for f in fee_field if isinstance(f, dict))
        fees_field = trade.get("fees")
        if isinstance(fees_field, list):
            fees.extend(f for f in fees_field if isinstance(f, dict))
        return fees

    def _order_complete(self, payload: Dict[str, Any]) -> bool:
        status = str(payload.get("status") or "").lower()
        if status in {"closed", "canceled", "cancelled", "rejected"}:
            return True
        remaining = self._safe_float(payload.get("remaining"))
        if remaining is not None and remaining <= 0:
            return True
        filled = self._safe_float(payload.get("filled")) or 0.0
        if status == "open" and remaining is None and filled > 0:
            return True
        return False

    def _log_slippage_effect(
        self, leg: TriangularTradeLeg, metrics: Mapping[str, Any]
    ) -> None:
        planned_in = float(leg.amount_in)
        planned_out = float(leg.amount_out)
        actual_in = self._safe_float(metrics.get("amount_in"))
        actual_out = self._safe_float(metrics.get("amount_out"))
        if actual_in is None or actual_out is None:
            return

        input_delta = actual_in - planned_in
        input_pct = (input_delta / planned_in * 100.0) if planned_in else 0.0
        output_delta = actual_out - planned_out
        output_pct = (output_delta / planned_out * 100.0) if planned_out else 0.0

        logger.info(
            "Slippage effect for %s %s: input %+0.8f (%+.4f%%) output %+0.8f (%+.4f%%)",
            leg.side.upper(),
            leg.symbol,
            input_delta,
            input_pct,
            output_delta,
            output_pct,
        )

    def _log_route_slippage(
        self,
        opportunity: TriangularOpportunity,
        actual_final_amount: float,
        actual_final_without_fees: float,
        actual_profit_percentage: float,
    ) -> None:
        planned_final = float(opportunity.final_amount)
        planned_final_wo_fee = float(opportunity.final_amount_without_fees)
        delta_final = actual_final_amount - planned_final
        delta_final_pct = (
            (actual_final_amount / planned_final - 1.0) * 100.0
            if planned_final
            else 0.0
        )
        delta_wo_fee = actual_final_without_fees - planned_final_wo_fee
        delta_wo_fee_pct = (
            (actual_final_without_fees / planned_final_wo_fee - 1.0) * 100.0
            if planned_final_wo_fee
            else 0.0
        )

        logger.info(
            "Route execution slippage: final %+0.8f (%+.4f%%) without fees %+0.8f (%+.4f%%) profit %+0.4f%%",
            delta_final,
            delta_final_pct,
            delta_wo_fee,
            delta_wo_fee_pct,
            actual_profit_percentage,
        )

    def _submit_leg_order(
        self, leg: TriangularTradeLeg, available_amount: float
    ) -> Tuple[Dict[str, Any], float, float]:
        """Create a market order for ``leg``, retrying when funds are marginal."""

        attempts = 1 if self.dry_run else self._INSUFFICIENT_FUNDS_RETRIES
        attempt_available = float(available_amount)
        last_error: Optional[Exception] = None

        for attempt in range(attempts):
            synced_available = self._sync_available_with_exchange(leg, attempt_available)

            if synced_available <= 0:
                break

            amount, scale = self._determine_order_amount(leg, synced_available)

            if amount <= 0:
                raise ValueError(
                    f"Calculated non-positive trade amount ({amount}) for leg {leg.symbol}"
                )

            try:
                order = self.exchange.create_market_order(
                    leg.symbol,
                    leg.side,
                    amount,
                    test_order=self.dry_run,
                )
            except Exception as exc:
                if self.dry_run or not self._is_insufficient_funds_error(exc):
                    raise

                last_error = exc
                adjusted_available = self._reduce_available_after_insufficient_funds(
                    synced_available, attempt
                )
                if adjusted_available <= 0:
                    break
                if math.isclose(
                    adjusted_available,
                    attempt_available,
                    rel_tol=0,
                    abs_tol=1e-18,
                ):
                    adjusted_available = attempt_available * (1 - 1e-6)
                if adjusted_available <= 0:
                    break
                logger.debug(
                    "Retrying %s %s after insufficient funds (attempt %s/%s): %s -> %s",
                    leg.side,
                    leg.symbol,
                    attempt + 1,
                    attempts,
                    attempt_available,
                    adjusted_available,
                )
                attempt_available = adjusted_available
                continue

            return order, amount, scale

        if last_error is not None:
            raise last_error

        raise ValueError(
            f"Calculated non-positive trade amount for leg {leg.symbol} after adjustments"
        )

    def _reduce_available_after_insufficient_funds(
        self, available_amount: float, attempt: int
    ) -> float:
        """Return a slightly smaller available amount after an insufficient funds error."""

        if available_amount <= 0:
            return 0.0

        fractional_step = min(0.05, 2e-3 * (attempt + 1))
        target = available_amount * (1 - fractional_step)
        nudged = math.nextafter(available_amount, 0.0)

        candidates = [value for value in (target, nudged) if value > 0]
        if not candidates:
            return 0.0

        reduced = min(candidates)
        if math.isclose(reduced, available_amount, rel_tol=0, abs_tol=1e-18):
            reduced = available_amount * (1 - fractional_step)

        return max(reduced, 0.0)

    @staticmethod
    def _is_insufficient_funds_error(exc: Exception) -> bool:
        """Return ``True`` when ``exc`` indicates an insufficient funds rejection."""

        if CcxtInsufficientFunds is not None and isinstance(exc, CcxtInsufficientFunds):
            return True
        if exc.__class__.__name__.lower() == "insufficientfunds":
            return True
        message = str(exc)
        return "insufficient funds" in message.lower()

    def _sync_available_with_exchange(
        self, leg: TriangularTradeLeg, available_amount: float
    ) -> float:
        """Cap ``available_amount`` to the exchange-reported free balance for ``leg``."""

        if self.dry_run:
            return float(available_amount)

        currency = self._leg_balance_currency(leg)
        if currency is None:
            return float(available_amount)

        fetch_balance = getattr(self.exchange, "fetch_balance", None)
        if not callable(fetch_balance):
            return float(available_amount)

        try:
            balance = fetch_balance()
        except Exception as exc:  # pragma: no cover - network failure path
            logger.debug(
                "Unable to fetch balance while preparing %s %s: %s",
                leg.side,
                leg.symbol,
                exc,
            )
            return float(available_amount)

        exchange_available = self._extract_free_balance(balance, currency)
        if exchange_available is None:
            return float(available_amount)

        capped = min(float(available_amount), exchange_available)
        if capped < float(available_amount):
            logger.debug(
                "Capping available amount for %s %s to exchange balance %.16f (local %.16f)",
                leg.side,
                leg.symbol,
                exchange_available,
                available_amount,
            )

        return max(capped, 0.0)

    @staticmethod
    def _leg_balance_currency(leg: TriangularTradeLeg) -> Optional[str]:
        try:
            base, quote = leg.symbol.split("/")
        except ValueError:  # pragma: no cover - defensive guard
            return None
        return base if leg.side == "sell" else quote

    def _extract_free_balance(
        self, balance: Any, currency: str
    ) -> Optional[float]:
        """Return the free balance for ``currency`` from ``balance`` if available."""

        if balance is None:
            return None

        variants = {currency, currency.upper(), currency.lower()}

        if isinstance(balance, dict):
            free_section = balance.get("free")
            if isinstance(free_section, dict):
                for key in variants:
                    value = free_section.get(key)
                    if value is not None:
                        return self._coerce_float(value)

            for key in variants:
                entry = balance.get(key)
                value = self._extract_balance_entry(entry)
                if value is not None:
                    return value

        return self._coerce_float(balance if currency in variants else None)

    def _extract_balance_entry(self, entry: Any) -> Optional[float]:
        if isinstance(entry, dict):
            for field in ("free", "available", "total", "balance", "remaining"):
                value = entry.get(field)
                if value is not None:
                    coerced = self._coerce_float(value)
                    if coerced is not None:
                        return coerced
        else:
            coerced = self._coerce_float(entry)
            if coerced is not None:
                return coerced
        return None

    @staticmethod
    def _coerce_float(value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

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

        available_amount = float(available_amount)

        desired_amount = float(leg.traded_quantity)

        if self.dry_run:
            return desired_amount, 1.0

        def _finalise_amount(amount: float, scale: float) -> Tuple[float, float]:
            precision_fn = getattr(self.exchange, "amount_to_precision", None)
            raw_amount = float(amount)
            quantized = raw_amount
            if callable(precision_fn):
                try:
                    quantized = float(precision_fn(leg.symbol, raw_amount))
                except Exception:  # pragma: no cover - precision helpers are best-effort
                    quantized = raw_amount

                if quantized > raw_amount:
                    nudged = math.nextafter(raw_amount, 0.0)
                    try:
                        quantized_down = float(precision_fn(leg.symbol, nudged))
                    except Exception:  # pragma: no cover - best-effort fallback
                        quantized_down = nudged
                    if quantized_down <= raw_amount:
                        quantized = quantized_down

            if quantized > raw_amount:
                quantized = raw_amount
            if quantized <= 0:
                logger.debug(
                    "Rounded trade amount for %s dropped below precision (raw=%s)",
                    leg.symbol,
                    raw_amount,
                )
                return 0.0, 0.0
            if desired_amount > 0 and scale > 0:
                scale = min(scale, quantized / desired_amount)
            elif desired_amount > 0 and scale == 0:
                scale = quantized / desired_amount
            return quantized, max(scale, 0.0)

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
                return _finalise_amount(available_amount, scale)
            return _finalise_amount(desired_amount, 1.0)

        # Buying the base asset using the currently held quote currency.
        required_quote = float(leg.amount_in)
        if available_amount <= 0:
            logger.debug("No %s available to buy %s", quote, leg.symbol)
            return 0.0, 0.0
        if required_quote <= 0:
            return _finalise_amount(desired_amount, 1.0)

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
            return _finalise_amount(adjusted_amount, max(scale, 0.0))

        return _finalise_amount(desired_amount, 1.0)

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
        order_poll_started = time.perf_counter()
        last_rest_poll = time.perf_counter()

        while time.time() < deadline:
            poll_interval = self._order_poll_interval(order_poll_started)
            ws_update, used_websocket = self._watch_order_update(
                order_id,
                symbol,
                poll_interval,
            )
            if ws_update:
                latest_payload.update(ws_update)
                if self._order_complete(latest_payload):
                    break

            now = time.perf_counter()
            should_poll_rest = (not used_websocket) or (
                now - last_rest_poll >= self._rest_poll_interval(used_websocket)
            )
            try:
                latest = (
                    self.exchange.fetch_order(order_id, symbol)
                    if should_poll_rest
                    else None
                )
                if should_poll_rest:
                    last_rest_poll = now
            except Exception as exc:  # pragma: no cover - ccxt/network failure
                logger.debug(
                    "Unable to poll order %s for %s: %s",
                    order_id,
                    symbol,
                    exc,
                )
                latest = None

            if latest:
                latest_payload.update(latest)
                if self._order_complete(latest_payload):
                    break

            if not used_websocket and latest is None and ws_update is None:
                time.sleep(poll_interval)
            elif used_websocket and latest is None and ws_update is None:
                time.sleep(min(self._WEBSOCKET_IDLE_SLEEP, poll_interval))

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
