"""Utilities for estimating slippage using live order book data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Mapping, Tuple

from .exceptions import InsufficientLiquidityError
from .models import (
    OrderBookSnapshot,
    TriangularOpportunity,
    TriangularTradeLeg,
)


_DEFAULT_TOLERANCE = 1e-12


@dataclass(frozen=True)
class LegSlippage:
    """Summary of the slippage observed on a single trade leg."""

    symbol: str
    side: str
    best_price: float
    vwap_price: float
    slippage_pct: float


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
) -> SlippageSimulation:
    """Replay ``opportunity`` using full order book depth.

    The returned opportunity reflects the volumes achievable when consuming the
    supplied order book snapshots. A
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

    for symbol, planned_leg in zip(opportunity.route.symbols, opportunity.trades):
        order_book = order_books.get(symbol)
        if order_book is None:
            raise KeyError(f"Missing order book snapshot for {symbol}")

        fee_rate = float(planned_leg.fee_rate)
        if planned_leg.side == "buy":
            base_acquired, quote_spent, best_price, vwap = _fill_buy_levels(
                symbol,
                order_book,
                current_amount,
            )
            fee_paid = base_acquired * fee_rate
            base_after_fee = base_acquired - fee_paid
            quote_without_fees = current_amount_without_fees
            base_without_fee_flow = (
                quote_without_fees / vwap if vwap > 0 else 0.0
            )

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
            slippage_pct = (
                ((vwap - best_price) / best_price) * 100.0 if best_price else 0.0
            )
        else:
            quote_acquired, base_sold, best_price, vwap = _fill_sell_levels(
                symbol,
                order_book,
                current_amount,
            )
            fee_paid = quote_acquired * fee_rate
            quote_after_fee = quote_acquired - fee_paid
            base_without_fee = current_amount_without_fees
            quote_without_fee_flow = base_without_fee * vwap

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

