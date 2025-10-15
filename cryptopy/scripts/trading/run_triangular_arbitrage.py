"""Command line entry point for the triangular arbitrage toolkit."""
from __future__ import annotations

import argparse
import asyncio
import logging
import math
import itertools
from datetime import datetime
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from asyncio import QueueEmpty

import yaml

from cryptopy.src.trading.triangular_arbitrage import (
    ExchangeConnection,
    InsufficientLiquidityError,
    PriceSnapshot,
    TriangularArbitrageCalculator,
    TriangularArbitrageExecutor,
    TriangularRoute,
)


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Runtime defaults
# ---------------------------------------------------------------------------
# These flags can be adjusted directly in the source file to quickly change the
# runner behaviour without editing the CLI invocation. Command-line arguments
# still take precedence when explicitly supplied.
EXCHANGE_DEFAULT = "kraken"
STARTING_CURRENCY_DEFAULT = "USD"
ENABLE_EXECUTION_DEFAULT = False
LIVE_TRADING_DEFAULT = False
USE_TESTNET_DEFAULT = True
DISABLE_WEBSOCKET_DEFAULT = False
LOG_LEVEL_DEFAULT = "DEBUG"
MAX_ROUTE_LENGTH_DEFAULT: Optional[int] = 3
EVALUATION_INTERVAL_DEFAULT = 30.0
WEBSOCKET_TIMEOUT_DEFAULT = 10.0
PRICE_REFRESH_INTERVAL_DEFAULT = 30.0
PRICE_MAX_AGE_DEFAULT = 60.0
ASSET_FILTER_DEFAULT: Sequence[str] = ()

# Location where executed trades will be persisted when trade logging is enabled.
TRADE_LOG_PATH_DEFAULT: Optional[Path] = (
    Path(__file__).resolve().parents[3] / "logs" / "triangular_trades.csv"
)

# Optional configuration file used to source API credentials when present.
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[3] / "config.yaml"
CONFIG_SECTION_BY_EXCHANGE = {
    "kraken": "kraken_websocket",
}

MARKET_FILTER_REASON_DESCRIPTIONS = {
    "synthetic_currency": (
        "Symbol includes a settlement suffix (e.g. ETH/USD:BTC) whose payouts occur in a third "
        "currency, so trimming the suffix would misrepresent the contract."
    ),
    "derivative_settlement": (
        "Instrument settles in a currency outside the configured starting currencies, so cash "
        "flows cannot be modelled accurately for the current route discovery run."
    ),
    "derivative_type": (
        "Marked as swap/future/option; pricing and settlement differ from spot requirements for "
        "triangular arbitrage."
    ),
    "derivative_flag": (
        "Exchange metadata flags the market as derivative-only, indicating non-spot behaviour."
    ),
    "asset_filter": (
        "Base or quote currency falls outside the configured asset filter, so the market was "
        "excluded from discovery."
    ),
}


def _chunked(sequence: Sequence[str], size: int) -> Iterable[Sequence[str]]:
    """Yield ``sequence`` slices of ``size`` elements (best-effort for the last chunk)."""

    if size <= 0:
        size = len(sequence) or 1
    for index in range(0, len(sequence), size):
        yield sequence[index : index + size]


def generate_triangular_routes(
    markets: Dict[str, Dict[str, object]],
    *,
    starting_currencies: Optional[Sequence[str]] = None,
    allowed_assets: Optional[Sequence[str]] = None,
) -> List[TriangularRoute]:
    """Construct all distinct three-leg triangular routes from exchange markets."""

    currency_graph: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    allowed_assets_set: Optional[set[str]] = None
    if allowed_assets:
        allowed_assets_set = {str(asset).upper() for asset in allowed_assets}
    for symbol, metadata in markets.items():
        if not isinstance(metadata, dict):
            continue
        if metadata.get("active") is False:
            continue
        base = metadata.get("base")
        quote = metadata.get("quote")
        if not base or not quote:
            continue
        base_str = str(base)
        quote_str = str(quote)
        if allowed_assets_set is not None:
            if base_str.upper() not in allowed_assets_set or quote_str.upper() not in allowed_assets_set:
                continue
        currency_graph[quote_str].append((symbol, base_str))
        currency_graph[base_str].append((symbol, quote_str))

    allowed_currencies: Optional[set[str]] = None
    if starting_currencies:
        allowed_currencies = {str(currency).upper() for currency in starting_currencies}

    routes: List[TriangularRoute] = []
    seen: set[Tuple[str, Tuple[str, str, str]]] = set()

    for start_currency, first_edges in currency_graph.items():
        start_currency_key = str(start_currency)
        if allowed_currencies and start_currency_key.upper() not in allowed_currencies:
            continue
        for symbol_a, currency_b in first_edges:
            for symbol_b, currency_c in currency_graph.get(currency_b, []):
                if symbol_b == symbol_a:
                    continue
                for symbol_c, currency_d in currency_graph.get(currency_c, []):
                    if symbol_c in (symbol_a, symbol_b):
                        continue
                    if str(currency_d) != start_currency_key:
                        continue
                    route_symbols = (symbol_a, symbol_b, symbol_c)
                    key = (start_currency_key, route_symbols)
                    if key in seen:
                        continue
                    seen.add(key)
                    routes.append(TriangularRoute(route_symbols, start_currency_key))

    additional_routes: List[TriangularRoute] = []
    for route in list(routes):
        reversed_symbols = tuple(reversed(route.symbols))
        key = (route.starting_currency, reversed_symbols)
        if key in seen:
            continue
        seen.add(key)
        additional_routes.append(TriangularRoute(reversed_symbols, route.starting_currency))

    routes.extend(additional_routes)

    routes.sort(key=lambda route: (route.starting_currency, route.symbols))
    return routes


def collect_available_assets(markets: Dict[str, Dict[str, object]]) -> List[str]:
    """Return sorted unique asset codes derived from exchange market metadata."""

    assets: set[str] = set()
    for metadata in markets.values():
        if not isinstance(metadata, dict):
            continue
        base = metadata.get("base")
        quote = metadata.get("quote")
        if base:
            base_main, _ = _split_currency_parts(str(base))
            assets.add(base_main.upper())
        if quote:
            quote_main, _ = _split_currency_parts(str(quote))
            assets.add(quote_main.upper())
        settle = metadata.get("settle")
        if settle:
            settle_main, settle_suffix = _split_currency_parts(str(settle))
            assets.add((settle_suffix or settle_main).upper())
    return sorted(assets)


def compute_route_log_cost(
    route: TriangularRoute,
    prices: Dict[str, PriceSnapshot],
    markets: Dict[str, Dict[str, object]],
    fee_lookup: Dict[str, float],
) -> Optional[float]:
    """Return the logarithmic cost of a route; negative implies arbitrage potential."""

    current_currency = route.starting_currency
    log_sum = 0.0

    for symbol in route.symbols:
        market = markets.get(symbol)
        price_snapshot = prices.get(symbol)
        if market is None or price_snapshot is None:
            return None

        base = market.get("base")
        quote = market.get("quote")
        if not base or not quote:
            return None

        fee = fee_lookup.get(symbol, 0.0)
        fee_factor = 1.0 - float(fee)
        if fee_factor <= 0:
            return None

        if current_currency == quote:
            ask = price_snapshot.ask
            if ask is None or ask <= 0:
                return None
            cost = float(ask) / fee_factor
            next_currency = str(base)
        elif current_currency == base:
            bid = price_snapshot.bid
            if bid is None or bid <= 0:
                return None
            cost = 1.0 / (float(bid) * fee_factor)
            next_currency = str(quote)
        else:
            return None

        log_sum += math.log(cost)
        current_currency = next_currency

    if current_currency != route.starting_currency:
        return None

    return log_sum


def select_routes_with_negative_log_sum(
    routes: Sequence[TriangularRoute],
    prices: Dict[str, PriceSnapshot],
    markets: Dict[str, Dict[str, object]],
    fee_lookup: Dict[str, float],
) -> Tuple[List[TriangularRoute], int, Optional[Tuple[TriangularRoute, float]]]:
    """Return routes whose cumulative log cost is negative along with the closest candidate.

    The integer in the returned tuple represents how many routes had sufficient
    market data to be evaluated (i.e. their log cost could be computed). The
    optional tuple provides the route with the lowest log cost (most negative or
    closest to zero) even if it was not arbitrage-positive.
    """

    selected: List[TriangularRoute] = []
    evaluable_routes = 0
    best_route: Optional[TriangularRoute] = None
    best_log_cost: Optional[float] = None
    for route in routes:
        log_cost = compute_route_log_cost(route, prices, markets, fee_lookup)
        if log_cost is None:
            continue
        evaluable_routes += 1
        if best_log_cost is None or log_cost < best_log_cost:
            best_log_cost = log_cost
            best_route = route
        if log_cost < 0:
            selected.append(route)
    closest = (best_route, best_log_cost) if best_route is not None and best_log_cost is not None else None
    return selected, evaluable_routes, closest


@dataclass
class OpportunityExecution:
    """Records the most recent execution to avoid duplicate orders."""

    route: TriangularRoute
    profit_signature: Tuple[float, float]


@dataclass
class CachedPrice:
    """Stores a price snapshot along with the time it was observed."""

    snapshot: PriceSnapshot
    updated_at: float


@dataclass(frozen=True)
class MarketFilterStats:
    """Summarises how many markets were usable for route discovery."""

    total: int
    retained: int
    skipped_by_reason: Dict[str, int]

    @property
    def skipped(self) -> int:
        return self.total - self.retained


def _split_currency_parts(value: str) -> Tuple[str, Optional[str]]:
    """Return the main currency code and any settlement suffix."""

    if ":" not in value:
        return value, None
    main, suffix = value.split(":", 1)
    return main, suffix or None


def filter_markets_for_triangular_routes(
    markets: Dict[str, Dict[str, object]],
    *,
    starting_currencies: Optional[Sequence[str]] = None,
    asset_filter: Optional[Sequence[str]] = None,
) -> Tuple[Dict[str, Dict[str, object]], MarketFilterStats]:
    """Remove inactive or derivative markets that cannot seed triangular routes."""

    filtered: Dict[str, Dict[str, object]] = {}
    skipped = Counter()
    total = 0
    allowed_settlements = (
        {currency.upper() for currency in starting_currencies}
        if starting_currencies
        else set()
    )
    allowed_assets = (
        {asset.upper() for asset in asset_filter}
        if asset_filter
        else None
    )

    for symbol, metadata in markets.items():
        total += 1
        if not isinstance(metadata, dict):
            skipped["invalid_metadata"] += 1
            continue

        if metadata.get("active") is False:
            skipped["inactive"] += 1
            continue

        base = metadata.get("base")
        quote = metadata.get("quote")
        if not base or not quote:
            skipped["missing_currency"] += 1
            continue

        base_str = str(base)
        quote_str = str(quote)
        base_main, base_suffix = _split_currency_parts(base_str)
        quote_main, quote_suffix = _split_currency_parts(quote_str)
        if base_suffix or quote_suffix:
            skipped["synthetic_currency"] += 1
            continue

        if allowed_assets is not None:
            if base_main.upper() not in allowed_assets or quote_main.upper() not in allowed_assets:
                skipped["asset_filter"] += 1
                continue

        settle_value = metadata.get("settle")
        if settle_value:
            settle_str = str(settle_value)
            settle_main, settle_suffix = _split_currency_parts(settle_str)
            settle_upper = (settle_suffix or settle_main).upper()
            base_upper = base_main.upper()
            quote_upper = quote_main.upper()
            if settle_upper not in {base_upper, quote_upper} and (
                not allowed_settlements or settle_upper not in allowed_settlements
            ):
                skipped["derivative_settlement"] += 1
                continue

        market_type = str(metadata.get("type") or "").lower()
        if market_type in {"swap", "future", "futures", "perpetual", "option"}:
            skipped["derivative_type"] += 1
            continue

        if metadata.get("swap") or metadata.get("future") or metadata.get("option"):
            skipped["derivative_flag"] += 1
            continue

        spot_flag = metadata.get("spot")
        if spot_flag is False:
            skipped["non_spot"] += 1
            continue

        filtered[symbol] = metadata

    stats = MarketFilterStats(total=total, retained=len(filtered), skipped_by_reason=dict(skipped))
    if stats.skipped:
        for reason, count in stats.skipped_by_reason.items():
            description = MARKET_FILTER_REASON_DESCRIPTIONS.get(reason)
            if description:
                logger.debug(
                    f"Filtered markets reason {reason.replace('_', ' ')} ({count}): {description}"
                )
    return filtered, stats


def load_credentials_from_config(exchange: str, config_path: Optional[str]) -> Dict[str, str]:
    """Load API credentials for ``exchange`` from a YAML configuration file."""

    raw_path = config_path if config_path else DEFAULT_CONFIG_PATH
    try:
        config_file = Path(raw_path).expanduser()
    except TypeError:
        logger.warning(
            f"Invalid config path {raw_path!r} supplied; skipping credential load."
        )
        return {}

    if isinstance(config_file, str):  # pragma: no cover - defensive for unexpected types
        config_file = Path(config_file)

    if not config_file.exists():
        logger.debug(f"Config path {config_file} does not exist; skipping credential load")
        return {}

    try:
        data = yaml.safe_load(config_file.read_text()) or {}
    except FileNotFoundError:  # pragma: no cover - race condition guard
        return {}
    except yaml.YAMLError:  # pragma: no cover - malformed config
        logger.warning(f"Unable to parse credentials from {config_file}")
        return {}

    section_name = CONFIG_SECTION_BY_EXCHANGE.get(exchange.lower())
    if not section_name:
        return {}

    section: Dict[str, Any] = {}
    if isinstance(data, dict):
        for key, candidate in data.items():
            if isinstance(key, str) and key.lower() == section_name.lower():
                section = candidate or {}
                break

    if not isinstance(section, dict) or not section:
        logger.debug(f"No credentials found for {exchange} in {config_file}")
        return {}

    credentials: Dict[str, str] = {}
    for raw_key, raw_value in section.items():
        if raw_value in (None, ""):
            continue
        key = str(raw_key).strip().lower()
        if key in {"apikey", "api_key", "key"} and "apiKey" not in credentials:
            credentials["apiKey"] = str(raw_value)
        elif key in {"secret", "api_secret", "secretkey"} and "secret" not in credentials:
            credentials["secret"] = str(raw_value)
        elif key in {"password", "passphrase"} and "password" not in credentials:
            credentials["password"] = str(raw_value)

    if not credentials:
        logger.debug(f"No credentials found for {exchange} in {config_file}")
        return {}

    masked = ", ".join(
        f"{key}={'***' if key != 'password' else '***'}"
        for key in sorted(credentials)
    )
    logger.debug(f"Loaded credentials for {exchange} from {config_file}: {masked}")
    return credentials


async def watch_realtime_prices(
    exchange: ExchangeConnection,
    symbols: Sequence[str],
    *,
    poll_interval: float,
    websocket_timeout: float,
    price_cache: Dict[str, CachedPrice],
    trigger_queue: "asyncio.Queue[str]",
    stop_event: asyncio.Event,
) -> None:
    """Listen for ticker updates (websocket/REST) and notify the evaluator."""

    if not symbols:
        await stop_event.wait()
        return

    loop = asyncio.get_running_loop()

    try:
        async for payload in exchange.watch_tickers(
            symbols,
            poll_interval=poll_interval,
            websocket_timeout=websocket_timeout,
        ):
            updated = False
            if payload:
                now = loop.time()
                for symbol, ticker in payload.items():
                    snapshot = PriceSnapshot.from_ccxt(symbol, ticker)
                    if snapshot is None:
                        continue
                    price_cache[symbol] = CachedPrice(snapshot, now)
                    updated = True

            if updated:
                try:
                    trigger_queue.put_nowait("websocket_update")
                except asyncio.QueueFull:  # pragma: no cover - unbounded by default
                    logger.debug(
                        "Trigger queue full; dropping websocket_update notification"
                    )
            if stop_event.is_set():
                break
    except asyncio.CancelledError:
        raise
    except Exception:  # pragma: no cover - network failure path
        logger.exception("Market data watcher stopped unexpectedly")
        stop_event.set()


async def refresh_prices_via_rest(
    exchange: ExchangeConnection,
    symbols: Sequence[str],
    *,
    refresh_interval: float,
    cache: Dict[str, CachedPrice],
    trigger_queue: "asyncio.Queue[str]",
    stop_event: asyncio.Event,
) -> None:
    """Refresh price snapshots at a controlled cadence using REST tickers."""

    if not symbols:
        await stop_event.wait()
        return

    loop = asyncio.get_running_loop()
    refresh_interval = max(refresh_interval, 0.1)
    symbol_cycle = itertools.cycle(symbols)
    per_symbol_delay = max(refresh_interval / max(len(symbols), 1), 0.1)

    try:
        while not stop_event.is_set():
            symbol = next(symbol_cycle)
            entry = cache.get(symbol)
            now = loop.time()
            needs_refresh = entry is None or (now - entry.updated_at) >= refresh_interval
            if needs_refresh:
                try:
                    ticker = await asyncio.to_thread(
                        exchange.market_data_client.fetch_ticker,
                        symbol,
                    )
                except Exception:
                    logger.debug(
                        f"Price refresh failed for {symbol}", exc_info=True
                    )
                else:
                    snapshot = PriceSnapshot.from_ccxt(symbol, ticker)
                    if snapshot is not None:
                        cache[symbol] = CachedPrice(snapshot, loop.time())
                        try:
                            trigger_queue.put_nowait("rest_refresh")
                        except asyncio.QueueFull:  # pragma: no cover - unbounded by default
                            logger.debug(
                                "Trigger queue full; dropping rest_refresh notification"
                            )
            await asyncio.sleep(per_symbol_delay)
    except asyncio.CancelledError:
        raise
    except Exception:  # pragma: no cover - defensive guard
        logger.exception("Price maintainer stopped unexpectedly")
        stop_event.set()


async def prime_price_cache(
    exchange: ExchangeConnection,
    symbols: Sequence[str],
    *,
    price_cache: Dict[str, CachedPrice],
) -> int:
    """Populate ``price_cache`` with initial ticker data and return insert count."""

    if not symbols:
        return 0

    loop = asyncio.get_running_loop()
    inserted = 0
    client = exchange.market_data_client

    client_has = getattr(client, "has", {})
    if isinstance(client_has, dict):
        has_fetch_tickers = bool(client_has.get("fetchTickers"))
    else:  # pragma: no cover - defensive branch for exotic client implementations
        getter = getattr(client_has, "get", lambda *_args, **_kwargs: False)
        has_fetch_tickers = bool(getter("fetchTickers"))

    def _store_snapshot(symbol: str, payload: Optional[Dict[str, Any]]) -> None:
        nonlocal inserted
        if not payload:
            return
        snapshot = PriceSnapshot.from_ccxt(symbol, payload)
        if snapshot is None:
            return
        price_cache[symbol] = CachedPrice(snapshot, loop.time())
        inserted += 1

    fetched: Dict[str, Dict[str, Any]] = {}
    if has_fetch_tickers:
        try:
            fetched_payload = await asyncio.to_thread(client.fetch_tickers, symbols)
        except TypeError:
            try:
                fetched_payload = await asyncio.to_thread(client.fetch_tickers)
            except Exception:
                logger.debug("Initial bulk fetch_tickers call failed", exc_info=True)
                fetched_payload = None
        except Exception:
            logger.debug("Initial fetch_tickers call failed", exc_info=True)
            fetched_payload = None
        else:
            fetched = {symbol: fetched_payload.get(symbol) for symbol in symbols if fetched_payload}

    for symbol, payload in fetched.items():
        _store_snapshot(symbol, payload)

    missing = [symbol for symbol in symbols if symbol not in price_cache]
    if not missing:
        return inserted

    chunk_size = max(min(20, len(missing)), 1)
    for chunk in _chunked(missing, chunk_size):
        tasks = [
            asyncio.to_thread(client.fetch_ticker, symbol)
            for symbol in chunk
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for symbol, result in zip(chunk, results):
            if isinstance(result, Exception):
                logger.debug(f"Initial price fetch failed for {symbol}: {result}")
                continue
            _store_snapshot(symbol, result)

    return inserted


async def evaluate_and_execute(
    calculator: TriangularArbitrageCalculator,
    executor: Optional[TriangularArbitrageExecutor],
    routes: Sequence[TriangularRoute],
    *,
    price_cache: Dict[str, CachedPrice],
    markets: Dict[str, Dict[str, object]],
    fee_lookup: Dict[str, float],
    starting_amount: float,
    min_profit_percentage: float,
    max_route_length: Optional[int],
    evaluation_interval: float,
    enable_execution: bool,
    price_max_age: float,
    trigger_queue: "asyncio.Queue[str]",
    stop_event: asyncio.Event,
) -> None:
    """Evaluate cached prices and execute profitable opportunities."""

    last_execution: Optional[OpportunityExecution] = None

    while not stop_event.is_set():
        reasons: List[str]
        try:
            reason = await asyncio.wait_for(trigger_queue.get(), timeout=evaluation_interval)
        except asyncio.TimeoutError:
            reasons = ["periodic"]
        else:
            reasons = [reason]
            while True:
                try:
                    reasons.append(trigger_queue.get_nowait())
                except QueueEmpty:
                    break

        if stop_event.is_set():
            break

        evaluation_started_wall_clock = datetime.now().isoformat()
        evaluation_started_at = perf_counter()
        reason_summary = ",".join(sorted(set(reasons)))

        loop = asyncio.get_running_loop()
        now = loop.time()
        fresh_prices: Dict[str, PriceSnapshot] = {
            symbol: entry.snapshot
            for symbol, entry in price_cache.items()
            if now - entry.updated_at <= price_max_age
        }

        if not fresh_prices:
            duration = perf_counter() - evaluation_started_at
            logger.info(
                f"Route evaluation ({reason_summary}) started at {evaluation_started_wall_clock}; "
                f"skipped in {duration:.3f}s due to missing fresh prices (max age {price_max_age:.1f}s)"
            )
            continue

        (
            candidate_routes,
            evaluable_route_count,
            closest_log_sum,
        ) = select_routes_with_negative_log_sum(
            routes,
            fresh_prices,
            markets,
            fee_lookup,
        )
        candidate_count = len(candidate_routes)
        discovered_count = len(routes)
        if candidate_count == 0:
            duration = perf_counter() - evaluation_started_at
            logger.info(
                f"Route evaluation ({reason_summary}) started at {evaluation_started_wall_clock}; "
                f"evaluated {candidate_count}/{evaluable_route_count} candidate routes "
                f"(from {discovered_count} discovered) in {duration:.3f}s"
            )
            logger.debug("No routes satisfied the negative log-sum arbitrage condition.")
            if closest_log_sum:
                closest_route, log_sum = closest_log_sum
                message_parts = [
                    f"Closest route by log-sum: route={' -> '.join(closest_route.symbols)}",
                    f"log_sum={log_sum:.6f}",
                ]
                try:
                    closest_opportunity = calculator.evaluate_route(
                        closest_route,
                        fresh_prices,
                        starting_amount=starting_amount,
                        min_profit_percentage=float("-inf"),
                    )
                except (InsufficientLiquidityError, KeyError, ValueError) as exc:
                    message_parts.append(f"evaluation failed: {exc}")
                else:
                    if closest_opportunity is not None:
                        message_parts.append(
                            f"estimated profit={closest_opportunity.profit:.6f} "
                            f"({closest_opportunity.profit_percentage:.4f}%)"
                        )
                        message_parts.append(
                            f"profit_without_fees={closest_opportunity.profit_without_fees:.6f}; "
                            f"fee_impact={closest_opportunity.fee_impact:.6f} "
                            f"{closest_opportunity.route.starting_currency}"
                        )
                logger.info("; ".join(message_parts))
            continue

        try:
            opportunities, stats = calculator.find_profitable_routes(
                candidate_routes,
                fresh_prices,
                starting_amount=starting_amount,
                min_profit_percentage=min_profit_percentage,
                max_route_length=max_route_length,
            )
        except (InsufficientLiquidityError, KeyError, ValueError) as exc:
            duration = perf_counter() - evaluation_started_at
            logger.info(
                f"Route evaluation ({reason_summary}) started at {evaluation_started_wall_clock}; "
                f"evaluated {candidate_count}/{evaluable_route_count} candidate routes "
                f"(from {discovered_count} discovered) in {duration:.3f}s"
            )
            logger.debug(f"Skipping evaluation due to error: {exc}")
            continue

        duration = perf_counter() - evaluation_started_at
        logger.info(
            f"Route evaluation ({reason_summary}) started at {evaluation_started_wall_clock}; "
            f"evaluated {candidate_count}/{evaluable_route_count} candidate routes "
            f"(from {discovered_count} discovered) in {duration:.3f}s"
        )

        if not opportunities:
            logger.info(
                "No profitable opportunities found; "
                f"{stats.rejected_by_profit} route(s) fell below the {min_profit_percentage:.4f}% minimum, "
                f"{stats.evaluation_errors} route(s) encountered errors, "
                f"{stats.filtered_by_length} route(s) exceeded the length limit."
            )
            if stats.evaluation_error_reasons:
                sorted_reasons = sorted(
                    stats.evaluation_error_reasons.items(), key=lambda item: item[1], reverse=True
                )
                formatted_reasons = ", ".join(
                    f"{reason} ({count})" for reason, count in sorted_reasons
                )
                logger.info(f"Route error breakdown: {formatted_reasons}")
            if stats.best_opportunity:
                logger.info(
                    f"Closest opportunity: route={' -> '.join(stats.best_opportunity.route.symbols)} "
                    f"profit={stats.best_opportunity.profit:.6f} "
                    f"({stats.best_opportunity.profit_percentage:.4f}%) "
                    f"profit_without_fees={stats.best_opportunity.profit_without_fees:.6f} "
                    f"fee_impact={stats.best_opportunity.fee_impact:.6f} "
                    f"{stats.best_opportunity.route.starting_currency}"
                )
            continue

        best = opportunities[0]
        logger.info(
            f"Best opportunity: route={' -> '.join(best.route.symbols)} "
            f"profit={best.profit:.6f} ({best.profit_percentage:.4f}%) "
            f"profit_without_fees={best.profit_without_fees:.6f} "
            f"fee_impact={best.fee_impact:.6f} {best.route.starting_currency}"
        )

        profit_signature = (round(best.final_amount, 8), round(best.profit_percentage, 4))
        if last_execution and last_execution.route == best.route and last_execution.profit_signature == profit_signature:
            logger.debug("Opportunity already executed recently; skipping duplicate execution")
            continue

        if not enable_execution or executor is None:
            logger.info(
                f"Execution disabled. Opportunity would yield {best.profit:.6f} "
                f"({best.profit_percentage:.4f}%) if executed (fees consumed {best.fee_impact:.6f} "
                f"{best.route.starting_currency})."
            )
            last_execution = OpportunityExecution(best.route, profit_signature)
            continue

        try:
            orders = await executor.execute_async(best)
        except ValueError as exc:
            logger.debug(f"Skipping execution: {exc}")
            continue

        last_execution = OpportunityExecution(best.route, profit_signature)
        if orders:
            logger.info(f"Placed {len(orders)} order(s) for opportunity.")
            try:
                trigger_queue.put_nowait("post_trade")
            except asyncio.QueueFull:  # pragma: no cover - unbounded by default
                logger.debug("Trigger queue full; dropping post_trade notification")


async def run_from_args(args: argparse.Namespace) -> None:
    exchange_name = args.exchange or EXCHANGE_DEFAULT
    if not exchange_name:
        raise SystemExit("An exchange must be specified either inline or via --exchange.")

    if args.live_trading and not args.enable_execution:
        raise SystemExit("--live-trading requires --enable-execution to be set.")

    credentials = load_credentials_from_config(exchange_name, args.config)
    if args.api_key:
        credentials["apiKey"] = args.api_key
    if args.secret:
        credentials["secret"] = args.secret
    if args.password:
        credentials["password"] = args.password

    if credentials:
        masked = ", ".join(
            f"{key}={'***' if key != 'password' else '***'}" for key in sorted(credentials)
        )
        logger.debug(f"Using credential set for {exchange_name}: {masked}")
    else:
        logger.debug(f"No credentials available for {exchange_name} after config/CLI merge")

    starting_currency = (args.starting_currency or STARTING_CURRENCY_DEFAULT).upper()
    asset_filter_entries = args.asset_filter or list(ASSET_FILTER_DEFAULT)
    asset_filter: List[str] = []
    for entry in asset_filter_entries:
        if entry is None:
            continue
        for part in str(entry).replace(",", " ").split():
            if part:
                asset_filter.append(part.upper())
    if asset_filter and starting_currency not in asset_filter:
        asset_filter.append(starting_currency)
    # Deduplicate while preserving order
    seen_assets: set[str] = set()
    deduped_assets: List[str] = []
    for asset in asset_filter:
        if asset in seen_assets:
            continue
        seen_assets.add(asset)
        deduped_assets.append(asset)
    asset_filter = deduped_assets
    max_route_length = args.max_route_length
    if max_route_length is not None and max_route_length <= 0:
        max_route_length = None

    price_refresh_interval = max(args.price_refresh_interval, 0.1)
    price_max_age = max(args.price_max_age, price_refresh_interval)

    trade_log_path: Optional[Path] = None
    if args.trade_log:
        trade_log_path = Path(args.trade_log).expanduser()

    exchange = ExchangeConnection(
        exchange_name,
        credentials=credentials or None,
        use_testnet=args.use_testnet,
        enable_websocket=not args.disable_websocket,
        make_trades=args.live_trading,
    )

    if args.use_testnet and not exchange.sandbox_supported:
        logger.warning(
            f"Sandbox mode is not available for {exchange_name} via ccxt; requests will hit the production API."
        )

    if asset_filter:
        logger.info(f"Restricting discovery to assets: {', '.join(asset_filter)}")

    markets = exchange.get_markets()
    available_assets = collect_available_assets(markets)
    if available_assets:
        logger.info(
            f"Available assets on {exchange_name}: {', '.join(available_assets)}"
        )
    markets, market_filter_stats = filter_markets_for_triangular_routes(
        markets,
        starting_currencies=[starting_currency],
        asset_filter=asset_filter or None,
    )
    if market_filter_stats.skipped:
        sorted_reasons = sorted(
            market_filter_stats.skipped_by_reason.items(), key=lambda item: item[1], reverse=True
        )
        reason_parts = []
        for reason, count in sorted_reasons:
            description = MARKET_FILTER_REASON_DESCRIPTIONS.get(reason)
            if description:
                reason_parts.append(
                    f"{reason.replace('_', ' ')} ({count}) - {description}"
                )
            else:
                reason_parts.append(f"{reason.replace('_', ' ')} ({count})")
        reason_summary = "; ".join(reason_parts)
        logger.info(
            f"Filtered {market_filter_stats.skipped} market(s) prior to route discovery: {reason_summary}"
        )
    routes = generate_triangular_routes(
        markets,
        starting_currencies=[starting_currency],
        allowed_assets=asset_filter or None,
    )
    if not routes:
        raise SystemExit(f"No triangular routes discovered for {exchange_name}.")
    logger.info(
        f"Discovered {len(routes)} triangular routes on {exchange_name} starting and ending in {starting_currency}"
    )

    calculator = TriangularArbitrageCalculator(
        exchange,
        slippage_buffer=args.slippage_buffer,
    )
    executor: Optional[TriangularArbitrageExecutor] = None
    if args.enable_execution:
        executor = TriangularArbitrageExecutor(
            exchange,
            dry_run=not args.live_trading,
            trade_log_path=trade_log_path,
        )
        if trade_log_path:
            logger.info(f"Logging executed trades to {trade_log_path}")

    symbols = sorted({symbol for route in routes for symbol in route.symbols})
    fee_lookup = {symbol: float(exchange.get_taker_fee(symbol)) for symbol in symbols}
    if fee_lookup:
        min_fee = min(fee_lookup.values())
        max_fee = max(fee_lookup.values())
        avg_fee = sum(fee_lookup.values()) / len(fee_lookup)
        fee_sources = exchange.get_fee_sources()
        source_counts = Counter(fee_sources.get(symbol, "default") for symbol in symbols)
        source_summary = ", ".join(
            f"{source}:{count}" for source, count in sorted(source_counts.items(), key=lambda item: item[1], reverse=True)
        )
        logger.info(
            f"Taker fee snapshot across {len(fee_lookup)} symbols: "
            f"min={min_fee:.4%}, max={max_fee:.4%}, avg={avg_fee:.4%} (sources: {source_summary})"
        )
    price_cache: Dict[str, CachedPrice] = {}
    trigger_queue: "asyncio.Queue[str]" = asyncio.Queue()
    stop_event = asyncio.Event()

    if symbols:
        initial_loaded = await prime_price_cache(
            exchange,
            symbols,
            price_cache=price_cache,
        )
        if initial_loaded:
            await trigger_queue.put("initial_snapshot")
            logger.debug(
                f"Primed price cache with {initial_loaded} snapshot(s) prior to evaluation"
            )

    market_data_task = asyncio.create_task(
        watch_realtime_prices(
            exchange,
            symbols,
            poll_interval=args.poll_interval,
            websocket_timeout=args.websocket_timeout,
            price_cache=price_cache,
            trigger_queue=trigger_queue,
            stop_event=stop_event,
        ),
        name="realtime-price-watcher",
    )

    price_task = asyncio.create_task(
        refresh_prices_via_rest(
            exchange,
            symbols,
            refresh_interval=price_refresh_interval,
            cache=price_cache,
            trigger_queue=trigger_queue,
            stop_event=stop_event,
        ),
        name="rest-price-refresher",
    )

    evaluator = asyncio.create_task(
        evaluate_and_execute(
            calculator,
            executor,
            routes,
            price_cache=price_cache,
            markets=markets,
            fee_lookup=fee_lookup,
            starting_amount=args.starting_amount,
            min_profit_percentage=args.min_profit_percentage,
            max_route_length=max_route_length,
            evaluation_interval=args.evaluation_interval,
            enable_execution=args.enable_execution,
            price_max_age=price_max_age,
            trigger_queue=trigger_queue,
            stop_event=stop_event,
        ),
        name="arbitrage-evaluator",
    )

    try:
        await asyncio.gather(market_data_task, price_task, evaluator)
    except asyncio.CancelledError:  # pragma: no cover - cancellation path
        pass
    except KeyboardInterrupt:
        logger.info("Interrupted by user. Shutting down...")
    finally:
        stop_event.set()
        market_data_task.cancel()
        price_task.cancel()
        evaluator.cancel()
        await asyncio.gather(market_data_task, price_task, evaluator, return_exceptions=True)
        await exchange.aclose()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Monitor and execute triangular arbitrage opportunities in real time.",
    )
    parser.add_argument(
        "--exchange",
        default=EXCHANGE_DEFAULT,
        help="Name of the exchange supported by ccxt (e.g. binance, kraken).",
    )
    parser.add_argument(
        "--starting-currency",
        default=STARTING_CURRENCY_DEFAULT,
        help="Currency used to seed and settle each arbitrage route.",
    )
    parser.add_argument(
        "--asset-filter",
        action="append",
        default=list(ASSET_FILTER_DEFAULT),
        metavar="ASSET",
        help=(
            "Restrict discovery to markets whose base/quote currencies appear in this list. "
            "Provide multiple times or comma-separated values; leave unset to include all assets."
        ),
    )
    parser.add_argument(
        "--starting-amount",
        type=float,
        default=100.0,
        help="Amount of the starting currency used for simulations and orders.",
    )
    parser.add_argument(
        "--min-profit-percentage",
        type=float,
        default=0.0,
        help="Minimum profit percentage required before executing an opportunity.",
    )
    parser.add_argument(
        "--max-route-length",
        type=int,
        default=MAX_ROUTE_LENGTH_DEFAULT,
        help="Optional maximum number of legs per route to evaluate.",
    )
    parser.add_argument(
        "--slippage-buffer",
        type=float,
        default=0.0,
        help="Fractional buffer applied after fees to account for slippage (e.g. 0.01 for 1%%).",
    )
    parser.add_argument(
        "--price-refresh-interval",
        type=float,
        default=PRICE_REFRESH_INTERVAL_DEFAULT,
        help="Target interval in seconds between REST ticker refreshes per symbol.",
    )
    parser.add_argument(
        "--price-max-age",
        type=float,
        default=PRICE_MAX_AGE_DEFAULT,
        help="Maximum age in seconds for cached price snapshots to be considered fresh.",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=2.0,
        help="Fallback polling interval (seconds) when websockets are unavailable.",
    )
    parser.add_argument(
        "--websocket-timeout",
        type=float,
        default=WEBSOCKET_TIMEOUT_DEFAULT,
        help="Maximum time to wait for a websocket ticker update before polling via REST.",
    )
    parser.add_argument(
        "--evaluation-interval",
        type=float,
        default=EVALUATION_INTERVAL_DEFAULT,
        help="Minimum interval in seconds between opportunity evaluations.",
    )
    parser.add_argument(
        "--trade-log",
        default=TRADE_LOG_PATH_DEFAULT,
        help="Optional path to a CSV file where executed legs will be appended.",
    )
    parser.add_argument(
        "--api-key", dest="api_key", default=None, help="API key for authenticated requests.")
    parser.add_argument(
        "--secret", dest="secret", default=None, help="API secret for authenticated requests."
    )
    parser.add_argument(
        "--password", dest="password", default=None, help="Exchange password if required."
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional path to a YAML file containing API credentials.",
    )
    parser.add_argument(
        "--use-testnet",
        action=argparse.BooleanOptionalAction,
        default=USE_TESTNET_DEFAULT,
        help="Enable or disable the exchange sandbox/testnet when supported.",
    )
    parser.add_argument(
        "--disable-websocket",
        action=argparse.BooleanOptionalAction,
        default=DISABLE_WEBSOCKET_DEFAULT,
        help="Disable websocket usage and rely on REST polling for price updates.",
    )
    parser.add_argument(
        "--enable-execution",
        action=argparse.BooleanOptionalAction,
        default=ENABLE_EXECUTION_DEFAULT,
        help="Allow the executor to place simulated or live orders.",
    )
    parser.add_argument(
        "--live-trading",
        action=argparse.BooleanOptionalAction,
        default=LIVE_TRADING_DEFAULT,
        help="Execute real trades instead of running in dry-run mode.",
    )
    parser.add_argument(
        "--log-level",
        default=LOG_LEVEL_DEFAULT,
        help="Configure the logging level (e.g. DEBUG, INFO).",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    default_level = getattr(logging, LOG_LEVEL_DEFAULT.upper(), logging.INFO)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), default_level))
    try:
        asyncio.run(run_from_args(args))
    except KeyboardInterrupt:  # pragma: no cover - outer signal handler
        logger.info("Interrupted by user. Goodbye!")


if __name__ == "__main__":  # pragma: no cover - script entry point
    main()

