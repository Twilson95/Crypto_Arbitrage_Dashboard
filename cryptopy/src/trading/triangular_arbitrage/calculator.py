"""Core arbitrage opportunity calculations."""
from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Tuple

from cryptopy.src.trading.triangular_arbitrage.exceptions import (
    InsufficientLiquidityError,
)
from cryptopy.src.trading.triangular_arbitrage.models import (
    PriceSnapshot,
    RouteEvaluationStats,
    TriangularOpportunity,
    TriangularRoute,
    TriangularTradeLeg,
)


class TriangularArbitrageCalculator:
    """Evaluates triangular arbitrage opportunities using ticker prices."""

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
        prices: Dict[str, PriceSnapshot],
        *,
        starting_amount: float,
        min_profit_percentage: float = 0.0,
    ) -> Optional[TriangularOpportunity]:
        current_currency = route.starting_currency
        current_amount = starting_amount
        current_amount_without_fees = starting_amount
        trades: List[TriangularTradeLeg] = []

        for symbol in route.symbols:
            price_snapshot = prices.get(symbol)
            if price_snapshot is None:
                raise KeyError(f"Missing price snapshot for {symbol}")

            base, quote = symbol.split("/")
            if current_currency == quote:
                (
                    trade,
                    current_amount,
                    current_amount_without_fees,
                ) = self._simulate_buy(
                    symbol,
                    current_amount,
                    current_amount_without_fees,
                    price_snapshot,
                )
                current_currency = base
                trades.append(trade)
            elif current_currency == base:
                (
                    trade,
                    current_amount,
                    current_amount_without_fees,
                ) = self._simulate_sell(
                    symbol,
                    current_amount,
                    current_amount_without_fees,
                    price_snapshot,
                )
                current_currency = quote
                trades.append(trade)
            else:
                raise ValueError(
                    f"Currency mismatch. Have {current_currency} but symbol {symbol} offers {base}/{quote}."
                )

        opportunity = TriangularOpportunity(
            route=route,
            starting_amount=starting_amount,
            final_amount=current_amount,
            final_amount_without_fees=current_amount_without_fees,
            trades=trades,
        )

        if opportunity.profit_percentage < min_profit_percentage:
            return None
        return opportunity

    def find_profitable_routes(
        self,
        routes: Iterable[TriangularRoute],
        prices: Dict[str, PriceSnapshot],
        *,
        starting_amount: float,
        min_profit_percentage: float = 0.0,
        max_route_length: Optional[int] = None,
    ) -> Tuple[List[TriangularOpportunity], RouteEvaluationStats]:
        opportunities: List[TriangularOpportunity] = []
        total_routes = 0
        filtered_by_length = 0
        considered = 0
        evaluation_errors = 0
        rejected_by_profit = 0
        error_reasons: Counter[str] = Counter()
        best_opportunity: Optional[TriangularOpportunity] = None
        for route in routes:
            total_routes += 1
            if max_route_length is not None and len(route.symbols) > max_route_length:
                filtered_by_length += 1
                continue
            considered += 1
            try:
                opportunity = self.evaluate_route(
                    route,
                    prices,
                    starting_amount=starting_amount,
                    min_profit_percentage=float("-inf"),
                )
            except (InsufficientLiquidityError, KeyError, ValueError) as exc:
                evaluation_errors += 1
                error_detail = exc.args[0] if getattr(exc, "args", None) else str(exc)
                error_message = f"{exc.__class__.__name__}: {error_detail}"
                error_reasons[error_message] += 1
                continue
            if opportunity is not None:
                if (
                    best_opportunity is None
                    or opportunity.profit_percentage > best_opportunity.profit_percentage
                ):
                    best_opportunity = opportunity
            if (
                opportunity is None
                or opportunity.profit_percentage < min_profit_percentage
            ):
                rejected_by_profit += 1
                continue
            opportunities.append(opportunity)
        opportunities.sort(key=lambda opp: opp.profit_percentage, reverse=True)
        stats = RouteEvaluationStats(
            total_routes=total_routes,
            considered=considered,
            filtered_by_length=filtered_by_length,
            evaluation_errors=evaluation_errors,
            rejected_by_profit=rejected_by_profit,
            evaluation_error_reasons=dict(error_reasons),
            best_opportunity=best_opportunity,
        )
        return opportunities, stats

    def _simulate_buy(
        self,
        symbol: str,
        quote_amount: float,
        quote_amount_without_fees: float,
        snapshot: PriceSnapshot,
    ) -> Tuple[TriangularTradeLeg, float, float]:
        if quote_amount <= 0:
            raise ValueError("quote_amount must be positive")

        ask = snapshot.ask
        if ask is None or ask <= 0:
            raise InsufficientLiquidityError(
                f"No ask price available for {symbol} to spend {quote_amount} {symbol.split('/')[1]}"
            )

        slippage_factor = 1.0 - self.slippage_buffer if self.slippage_buffer else 1.0
        acquired_base = quote_amount / ask
        fee_rate = self.exchange.get_taker_fee(symbol)
        base_after_fee = acquired_base * (1 - fee_rate)
        fee_paid = acquired_base - base_after_fee
        base_after_slippage = base_after_fee * slippage_factor
        base_without_fee_actual = acquired_base * slippage_factor

        acquired_base_without_fees = quote_amount_without_fees / ask
        base_without_fee_flow = acquired_base_without_fees * slippage_factor

        trade = TriangularTradeLeg(
            symbol=symbol,
            side="buy",
            amount_in=quote_amount,
            amount_out=base_after_slippage,
            amount_out_without_fee=base_without_fee_actual,
            average_price=ask,
            fee_rate=fee_rate,
            fee_paid=fee_paid,
            traded_quantity=acquired_base,
        )

        return trade, base_after_slippage, base_without_fee_flow

    def _simulate_sell(
        self,
        symbol: str,
        base_amount: float,
        base_amount_without_fees: float,
        snapshot: PriceSnapshot,
    ) -> Tuple[TriangularTradeLeg, float, float]:
        if base_amount <= 0:
            raise ValueError("base_amount must be positive")

        bid = snapshot.bid
        if bid is None or bid <= 0:
            raise InsufficientLiquidityError(
                f"No bid price available for {symbol} to sell {base_amount} {symbol.split('/')[0]}"
            )

        slippage_factor = 1.0 - self.slippage_buffer if self.slippage_buffer else 1.0
        quote_acquired = base_amount * bid
        fee_rate = self.exchange.get_taker_fee(symbol)
        quote_after_fee = quote_acquired * (1 - fee_rate)
        fee_paid = quote_acquired - quote_after_fee
        quote_after_slippage = quote_after_fee * slippage_factor
        quote_without_fee_actual = quote_acquired * slippage_factor

        quote_acquired_without_fees = base_amount_without_fees * bid
        quote_without_fee_flow = quote_acquired_without_fees * slippage_factor

        trade = TriangularTradeLeg(
            symbol=symbol,
            side="sell",
            amount_in=base_amount,
            amount_out=quote_after_slippage,
            amount_out_without_fee=quote_without_fee_actual,
            average_price=bid,
            fee_rate=fee_rate,
            fee_paid=fee_paid,
            traded_quantity=base_amount,
        )

        return trade, quote_after_slippage, quote_without_fee_flow
