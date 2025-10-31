"""Helpers for logging hypothetical opportunities under reduced taker fees."""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from .calculator import TriangularArbitrageCalculator
from .models import PriceSnapshot, TriangularOpportunity, TriangularRoute


class _ReducedFeeExchangeProxy:
    """Delegate exchange wrapper that caps taker fees to a reduced rate."""

    def __init__(self, exchange: Any, reduced_taker_fee: float) -> None:
        self._exchange = exchange
        self._reduced_taker_fee = float(reduced_taker_fee)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._exchange, name)

    def get_taker_fee(self, symbol: str) -> float:  # pragma: no cover - thin wrapper
        try:
            current = float(self._exchange.get_taker_fee(symbol))
        except Exception:  # pragma: no cover - defensive guard
            current = self._reduced_taker_fee
        return min(current, self._reduced_taker_fee)


@dataclass
class ReducedFeeLogContext:
    reason: str
    evaluation_started: str
    candidate_routes: int
    evaluable_routes: int


class ReducedFeeOpportunityLogger:
    """Re-evaluates opportunities with reduced fees and logs profitable cases."""

    def __init__(
        self,
        exchange: Any,
        *,
        reduced_taker_fee: float = 0.0024,
        log_path: Path | str = "reduced_fee_opportunities.jsonl",
    ) -> None:
        self._log_path = Path(log_path)
        self._reduced_taker_fee = float(reduced_taker_fee)
        self._calculator = TriangularArbitrageCalculator(
            _ReducedFeeExchangeProxy(exchange, self._reduced_taker_fee)
        )

    # ------------------------------------------------------------------ #
    def log_from_opportunity(
        self,
        prices: Mapping[str, PriceSnapshot],
        raw_opportunity: TriangularOpportunity,
        adjusted_opportunity: TriangularOpportunity,
        *,
        min_profit_percentage: float,
        context: ReducedFeeLogContext,
        slippage_impact_pct: float,
        slippage_details: Optional[Dict[str, float]] = None,
    ) -> None:
        """Log the opportunity if reduced fees would render it profitable."""

        reduced = self._evaluate_route(prices, raw_opportunity.route, raw_opportunity.starting_amount)
        if reduced is None:
            return

        estimated_profit_pct = reduced.profit_percentage - slippage_impact_pct
        if estimated_profit_pct <= min_profit_percentage:
            return
        estimated_profit = reduced.starting_amount * (estimated_profit_pct / 100.0)

        record = {
            "observed_at": datetime.now(timezone.utc).isoformat(),
            "context": {
                "reason": context.reason,
                "evaluation_started": context.evaluation_started,
                "candidate_routes": context.candidate_routes,
                "evaluable_routes": context.evaluable_routes,
            },
            "route": list(raw_opportunity.route.symbols),
            "starting_currency": raw_opportunity.route.starting_currency,
            "starting_amount": reduced.starting_amount,
            "current": {
                "profit": adjusted_opportunity.profit,
                "profit_pct": adjusted_opportunity.profit_percentage,
            },
            "reduced_fee": {
                "capped_taker_fee": self._reduced_taker_fee,
                "profit_no_slippage": reduced.profit,
                "profit_pct_no_slippage": reduced.profit_percentage,
                "estimated_profit": estimated_profit,
                "estimated_profit_pct": estimated_profit_pct,
                "profit_gain_pct": estimated_profit_pct - adjusted_opportunity.profit_percentage,
            },
            "slippage": slippage_details or {
                "estimated_impact_pct": max(slippage_impact_pct, 0.0),
            },
        }

        self._append(record)

    def log_from_stats(
        self,
        prices: Mapping[str, PriceSnapshot],
        opportunity: TriangularOpportunity,
        *,
        min_profit_percentage: float,
        context: ReducedFeeLogContext,
    ) -> None:
        """Evaluate a non-profitable opportunity (no slippage adjustments)."""

        reduced = self._evaluate_route(prices, opportunity.route, opportunity.starting_amount)
        if reduced is None:
            return

        if reduced.profit_percentage <= min_profit_percentage:
            return

        record = {
            "observed_at": datetime.now(timezone.utc).isoformat(),
            "context": {
                "reason": context.reason,
                "evaluation_started": context.evaluation_started,
                "candidate_routes": context.candidate_routes,
                "evaluable_routes": context.evaluable_routes,
            },
            "route": list(opportunity.route.symbols),
            "starting_currency": opportunity.route.starting_currency,
            "starting_amount": reduced.starting_amount,
            "current": {
                "profit": opportunity.profit,
                "profit_pct": opportunity.profit_percentage,
            },
            "reduced_fee": {
                "capped_taker_fee": self._reduced_taker_fee,
                "profit_no_slippage": reduced.profit,
                "profit_pct_no_slippage": reduced.profit_percentage,
                "estimated_profit": reduced.profit,
                "estimated_profit_pct": reduced.profit_percentage,
                "profit_gain_pct": reduced.profit_percentage - opportunity.profit_percentage,
            },
            "slippage": {"estimated_impact_pct": 0.0},
        }

        self._append(record)

    # ------------------------------------------------------------------ #
    def _evaluate_route(
        self,
        prices: Mapping[str, PriceSnapshot],
        route: TriangularRoute,
        starting_amount: float,
    ) -> Optional[TriangularOpportunity]:
        try:
            return self._calculator.evaluate_route(
                route,
                dict(prices),
                starting_amount=starting_amount,
                min_profit_percentage=float("-inf"),
            )
        except Exception:  # pragma: no cover - defensive guard against transient data issues
            return None

    def _append(self, record: Dict[str, Any]) -> None:
        try:
            if not self._log_path.parent.exists():
                self._log_path.parent.mkdir(parents=True, exist_ok=True)
            with self._log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
        except OSError:  # pragma: no cover - best effort logging
            pass
