"""Utilities for estimating slippage using live order book data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Mapping, Optional, Tuple

from .exceptions import InsufficientLiquidityError
from .models import (
    OrderBookSnapshot,
    TriangularOpportunity,
    TriangularTradeLeg,
)


_DEFAULT_TOLERANCE = 1e-12


@dataclass(frozen=True)
class PrecisionAdapter:
    """Helpers to align simulated amounts with exchange precision."""

    amount_to_precision: Callable[[str, float], float]
    cost_to_precision: Callable[[str, float], float]

    @staticmethod
    def _quantize(
        func: Callable[[str, float], float], symbol: str, value: float
    ) -> float:
        try:
            return float(func(symbol, float(value)))
        except Exception:
            return float(value)

    def quantize_amount(self, symbol: str, value: float) -> float:
        return self._quantize(self.amount_to_precision, symbol, value)

    def quantize_cost(self, symbol: str, value: float) -> float:
        return self._quantize(self.cost_to_precision, symbol, value)


@dataclass(frozen=True)
class LegSlippage:
    """Summary of the slippage observed on a single trade leg."""

    symbol: str
    side: str
    best_price: float
    vwap_price: float
    slippage_pct: float
    expected_amount_in: float
    actual_amount_in: float
    input_slippage_pct: float
    expected_amount_out: float
    actual_amount_out: float
    output_slippage_pct: float


@dataclass(frozen=True)
class SlippageSimulation:
    """Result of replaying an opportunity against an order book snapshot."""

    opportunity: TriangularOpportunity
    legs: List[LegSlippage]


def _fill_buy_levels(
    symbol: str,
    order_book: OrderBookSnapshot,
    quote_amount: float,
) -> Tuple[float, float, float, float]:
    """Return ``(base_acquired, quote_spent, best_price, vwap_price)``."""

    if quote_amount <= 0:
        raise ValueError("quote_amount must be positive")

    best_level = order_book.best_ask()
    if best_level is None:
        raise InsufficientLiquidityError(
            f"No ask depth available for {symbol}"
        )

    best_price = float(best_level[0])
    remaining_quote = float(quote_amount)
    total_base = 0.0
    total_quote = 0.0

    for price, quantity in order_book.asks:
        if remaining_quote <= _DEFAULT_TOLERANCE:
            break
        if price <= 0 or quantity <= 0:
            continue
        price = float(price)
        quantity = float(quantity)
        max_quote_at_level = price * quantity
        quote_traded = min(remaining_quote, max_quote_at_level)
        if quote_traded <= 0:
            continue
        base_traded = quote_traded / price
        total_base += base_traded
        total_quote += quote_traded
        remaining_quote -= quote_traded

    if remaining_quote > _DEFAULT_TOLERANCE:
        raise InsufficientLiquidityError(
            f"Insufficient ask depth for {symbol} to spend {quote_amount}"
        )

    if total_base <= 0:
        raise InsufficientLiquidityError(
            f"Failed to acquire any base asset for {symbol}"
        )

    vwap = total_quote / total_base
    return total_base, total_quote, best_price, vwap


def _fill_sell_levels(
    symbol: str,
    order_book: OrderBookSnapshot,
    base_amount: float,
) -> Tuple[float, float, float, float]:
    """Return ``(quote_acquired, base_sold, best_price, vwap_price)``."""

    if base_amount <= 0:
        raise ValueError("base_amount must be positive")

    best_level = order_book.best_bid()
    if best_level is None:
        raise InsufficientLiquidityError(
            f"No bid depth available for {symbol}"
        )

    best_price = float(best_level[0])
    remaining_base = float(base_amount)
    total_base = 0.0
    total_quote = 0.0

    for price, quantity in order_book.bids:
        if remaining_base <= _DEFAULT_TOLERANCE:
            break
        if price <= 0 or quantity <= 0:
            continue
        price = float(price)
        quantity = float(quantity)
        base_traded = min(remaining_base, quantity)
        if base_traded <= 0:
            continue
        quote_traded = base_traded * price
        total_base += base_traded
        total_quote += quote_traded
        remaining_base -= base_traded

    if remaining_base > _DEFAULT_TOLERANCE:
        raise InsufficientLiquidityError(
            f"Insufficient bid depth for {symbol} to sell {base_amount}"
        )

    if total_base <= 0:
        raise InsufficientLiquidityError(
            f"Failed to sell any base asset for {symbol}"
        )

    vwap = total_quote / total_base
    return total_quote, total_base, best_price, vwap


def simulate_opportunity_with_order_books(
    opportunity: TriangularOpportunity,
    order_books: Mapping[str, OrderBookSnapshot],
    *,
    starting_amount: float,
    precision: Optional[PrecisionAdapter] = None,
) -> SlippageSimulation:
    """Replay ``opportunity`` using full order book depth.

    The returned opportunity reflects the volumes achievable when consuming the
    supplied order book snapshots. When ``precision`` is provided, simulated
    amounts are rounded down to the exchange precision to avoid overstating
    fills. A
    :class:`~cryptopy.src.trading.triangular_arbitrage.exceptions.InsufficientLiquidityError`
    is raised when the available depth cannot satisfy a trade leg.
    """

    if starting_amount <= 0:
        raise ValueError("starting_amount must be positive")

    if len(opportunity.trades) != len(opportunity.route.symbols):
        raise ValueError("Opportunity trades do not match route symbols")

    current_amount = float(starting_amount)
    current_amount_without_fees = float(starting_amount)
    adjusted_trades: List[TriangularTradeLeg] = []
    leg_summaries: List[LegSlippage] = []

    plan_scale = (
        starting_amount / opportunity.starting_amount
        if opportunity.starting_amount > 0
        else 1.0
    )

    def _quantize_amount(symbol: str, value: float) -> float:
        return precision.quantize_amount(symbol, value) if precision else float(value)

    def _quantize_cost(symbol: str, value: float) -> float:
        return precision.quantize_cost(symbol, value) if precision else float(value)

    def _ensure_positive(symbol: str, value: float, context: str) -> None:
        if value <= _DEFAULT_TOLERANCE:
            raise InsufficientLiquidityError(
                f"{context} for {symbol} fell below exchange precision"
            )

    for symbol, planned_leg in zip(opportunity.route.symbols, opportunity.trades):
        order_book = order_books.get(symbol)
        if order_book is None:
            raise KeyError(f"Missing order book snapshot for {symbol}")

        fee_rate = float(planned_leg.fee_rate)
        if planned_leg.side == "buy":
            expected_amount_in = _quantize_cost(
                symbol, float(planned_leg.amount_in) * plan_scale
            )
            expected_amount_out = _quantize_amount(
                symbol, float(planned_leg.amount_out) * plan_scale
            )
        elif planned_leg.side == "sell":
            expected_amount_in = _quantize_amount(
                symbol, float(planned_leg.amount_in) * plan_scale
            )
            expected_amount_out = _quantize_cost(
                symbol, float(planned_leg.amount_out) * plan_scale
            )
        else:
            raise ValueError(f"Unsupported trade side {planned_leg.side!r} for {symbol}")

        _ensure_positive(symbol, expected_amount_in, "Trade amount")
        _ensure_positive(symbol, expected_amount_out, "Trade output")

        actual_amount_in_trade: float
        actual_amount_out_trade: float

        if planned_leg.side == "buy":
            current_amount = _quantize_cost(symbol, current_amount)
            current_amount_without_fees = _quantize_cost(
                symbol, current_amount_without_fees
            )
            _ensure_positive(symbol, current_amount, "Quote available")
            base_acquired, quote_spent, best_price, vwap = _fill_buy_levels(
                symbol,
                order_book,
                current_amount,
            )
            quote_spent = _quantize_cost(symbol, quote_spent)
            base_acquired = _quantize_amount(symbol, base_acquired)
            fee_paid = _quantize_amount(symbol, base_acquired * fee_rate)
            base_after_fee = _quantize_amount(symbol, base_acquired - fee_paid)
            quote_without_fees = current_amount_without_fees
            base_without_fee_flow = quote_without_fees / vwap if vwap > 0 else 0.0
            base_without_fee_flow = _quantize_amount(symbol, base_without_fee_flow)

            actual_amount_in_trade = quote_spent
            actual_amount_out_trade = base_after_fee

            adjusted_trades.append(
                TriangularTradeLeg(
                    symbol=symbol,
                    side="buy",
                    amount_in=quote_spent,
                    amount_out=base_after_fee,
                    amount_out_without_fee=base_acquired,
                    average_price=vwap,
                    fee_rate=fee_rate,
                    fee_paid=fee_paid,
                    traded_quantity=base_acquired,
                )
            )

            current_amount = base_after_fee
            current_amount_without_fees = base_without_fee_flow
            _ensure_positive(symbol, current_amount, "Post-fee base amount")
            slippage_pct = (
                ((vwap - best_price) / best_price) * 100.0 if best_price else 0.0
            )
        elif planned_leg.side == "sell":
            current_amount = _quantize_amount(symbol, current_amount)
            current_amount_without_fees = _quantize_amount(
                symbol, current_amount_without_fees
            )
            _ensure_positive(symbol, current_amount, "Base available")
            quote_acquired, base_sold, best_price, vwap = _fill_sell_levels(
                symbol,
                order_book,
                current_amount,
            )
            base_sold = _quantize_amount(symbol, base_sold)
            quote_acquired = _quantize_cost(symbol, quote_acquired)
            fee_paid = _quantize_cost(symbol, quote_acquired * fee_rate)
            quote_after_fee = _quantize_cost(symbol, quote_acquired - fee_paid)
            base_without_fee = current_amount_without_fees
            quote_without_fee_flow = base_without_fee * vwap
            quote_without_fee_flow = _quantize_cost(symbol, quote_without_fee_flow)

            actual_amount_in_trade = base_sold
            actual_amount_out_trade = quote_after_fee

            adjusted_trades.append(
                TriangularTradeLeg(
                    symbol=symbol,
                    side="sell",
                    amount_in=base_sold,
                    amount_out=quote_after_fee,
                    amount_out_without_fee=quote_acquired,
                    average_price=vwap,
                    fee_rate=fee_rate,
                    fee_paid=fee_paid,
                    traded_quantity=base_sold,
                )
            )

            current_amount = quote_after_fee
            current_amount_without_fees = quote_without_fee_flow
            _ensure_positive(symbol, current_amount, "Post-fee quote amount")
            slippage_pct = (
                ((best_price - vwap) / best_price) * 100.0 if best_price else 0.0
            )

        leg_summaries.append(
            LegSlippage(
                symbol=symbol,
                side=planned_leg.side,
                best_price=best_price,
                vwap_price=vwap,
                slippage_pct=slippage_pct,
                expected_amount_in=expected_amount_in,
                actual_amount_in=expected_amount_in,
                input_slippage_pct=(
                    ((expected_amount_in - actual_amount_in_trade) / expected_amount_in) * 100.0
                    if expected_amount_in
                    else 0.0
                ),
                expected_amount_out=expected_amount_out,
                actual_amount_out=expected_amount_out,
                output_slippage_pct=(
                    ((expected_amount_out - actual_amount_out_trade) / expected_amount_out) * 100.0
                    if expected_amount_out
                    else 0.0
                ),
            )
        )

    adjusted_opportunity = TriangularOpportunity(
        route=opportunity.route,
        starting_amount=starting_amount,
        final_amount=current_amount,
        final_amount_without_fees=current_amount_without_fees,
        trades=adjusted_trades,
    )

    return SlippageSimulation(adjusted_opportunity, leg_summaries)

