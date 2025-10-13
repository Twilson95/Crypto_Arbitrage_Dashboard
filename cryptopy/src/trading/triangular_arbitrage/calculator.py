"""Core arbitrage opportunity calculations."""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from .exceptions import InsufficientLiquidityError
from .models import (
    OrderBookSnapshot,
    TriangularOpportunity,
    TriangularRoute,
    TriangularTradeLeg,
)


class TriangularArbitrageCalculator:
    """Evaluates triangular arbitrage opportunities using order book depth."""

    def __init__(
        self,
        exchange: Any,
        *,
        slippage_buffer: float = 0.0,
    ) -> None:
        self.exchange = exchange
        self.slippage_buffer = slippage_buffer

    def evaluate_route(
        self,
        route: TriangularRoute,
        order_books: Dict[str, OrderBookSnapshot],
        *,
        starting_amount: float,
        min_profit_percentage: float = 0.0,
    ) -> Optional[TriangularOpportunity]:
        current_currency = route.starting_currency
        current_amount = starting_amount
        trades: List[TriangularTradeLeg] = []

        for symbol in route.symbols:
            order_book = order_books.get(symbol)
            if order_book is None:
                raise KeyError(f"Missing order book for {symbol}")

            base, quote = symbol.split("/")
            if current_currency == quote:
                trade = self._simulate_buy(symbol, current_amount, order_book)
                current_currency = base
                current_amount = trade.amount_out
                trades.append(trade)
            elif current_currency == base:
                trade = self._simulate_sell(symbol, current_amount, order_book)
                current_currency = quote
                current_amount = trade.amount_out
                trades.append(trade)
            else:
                raise ValueError(
                    f"Currency mismatch. Have {current_currency} but symbol {symbol} offers {base}/{quote}."
                )

        opportunity = TriangularOpportunity(
            route=route,
            starting_amount=starting_amount,
            final_amount=current_amount,
            trades=trades,
        )

        if opportunity.profit_percentage < min_profit_percentage:
            return None
        return opportunity

    def find_profitable_routes(
        self,
        routes: Iterable[TriangularRoute],
        order_books: Dict[str, OrderBookSnapshot],
        *,
        starting_amount: float,
        min_profit_percentage: float = 0.0,
        max_route_length: Optional[int] = None,
    ) -> List[TriangularOpportunity]:
        opportunities: List[TriangularOpportunity] = []
        for route in routes:
            if max_route_length is not None and len(route.symbols) > max_route_length:
                continue
            try:
                opportunity = self.evaluate_route(
                    route,
                    order_books,
                    starting_amount=starting_amount,
                    min_profit_percentage=min_profit_percentage,
                )
            except (InsufficientLiquidityError, KeyError, ValueError):
                continue
            if opportunity is not None:
                opportunities.append(opportunity)
        opportunities.sort(key=lambda opp: opp.profit_percentage, reverse=True)
        return opportunities

    def _simulate_buy(
        self,
        symbol: str,
        quote_amount: float,
        order_book: OrderBookSnapshot,
    ) -> TriangularTradeLeg:
        if quote_amount <= 0:
            raise ValueError("quote_amount must be positive")

        remaining_quote = quote_amount
        acquired_base = 0.0
        total_quote_used = 0.0

        for price, base_available in order_book.asks:
            if remaining_quote <= 0:
                break
            max_quote_at_level = base_available * price
            if max_quote_at_level <= remaining_quote:
                acquired_base += base_available
                remaining_quote -= max_quote_at_level
                total_quote_used += max_quote_at_level
            else:
                partial_base = remaining_quote / price
                acquired_base += partial_base
                total_quote_used += remaining_quote
                remaining_quote = 0.0
                break

        if acquired_base == 0 or remaining_quote > 1e-12:
            raise InsufficientLiquidityError(
                f"Not enough liquidity on {symbol} to spend {quote_amount} {symbol.split('/')[1]}"
            )

        average_price = total_quote_used / acquired_base
        fee_rate = self.exchange.get_taker_fee(symbol)
        base_after_fee = acquired_base * (1 - fee_rate)
        fee_paid = acquired_base - base_after_fee
        if self.slippage_buffer:
            base_after_fee *= (1 - self.slippage_buffer)

        return TriangularTradeLeg(
            symbol=symbol,
            side="buy",
            amount_in=total_quote_used,
            amount_out=base_after_fee,
            average_price=average_price,
            fee_paid=fee_paid,
            traded_quantity=acquired_base,
        )

    def _simulate_sell(
        self,
        symbol: str,
        base_amount: float,
        order_book: OrderBookSnapshot,
    ) -> TriangularTradeLeg:
        if base_amount <= 0:
            raise ValueError("base_amount must be positive")

        remaining_base = base_amount
        quote_acquired = 0.0

        for price, base_available in order_book.bids:
            if remaining_base <= 0:
                break
            trade_amount = min(base_available, remaining_base)
            quote_acquired += trade_amount * price
            remaining_base -= trade_amount

        if remaining_base > 1e-12:
            raise InsufficientLiquidityError(
                f"Not enough liquidity on {symbol} to sell {base_amount} {symbol.split('/')[0]}"
            )

        average_price = quote_acquired / base_amount
        fee_rate = self.exchange.get_taker_fee(symbol)
        quote_after_fee = quote_acquired * (1 - fee_rate)
        fee_paid = quote_acquired - quote_after_fee
        if self.slippage_buffer:
            quote_after_fee *= (1 - self.slippage_buffer)

        return TriangularTradeLeg(
            symbol=symbol,
            side="sell",
            amount_in=base_amount,
            amount_out=quote_after_fee,
            average_price=average_price,
            fee_paid=fee_paid,
            traded_quantity=base_amount,
        )
