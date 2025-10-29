"""Command line entry point for the triangular arbitrage toolkit."""
from __future__ import annotations

import argparse
import asyncio
import logging
import math
import os
import itertools
import time
from datetime import datetime
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from asyncio import QueueEmpty

try:  # pragma: no cover - optional dependency
    import psutil
except Exception:  # pragma: no cover - psutil is optional at runtime
    psutil = None

import yaml

from cryptopy.src.trading.triangular_arbitrage import (
    ExchangeConnection,
    ExchangeRequestTimeout,
    InsufficientLiquidityError,
    OrderBookSnapshot,
    PriceSnapshot,
    PrecisionAdapter,
    SlippageSimulation,
    TriangularArbitrageCalculator,
    TriangularArbitrageExecutor,
    TriangularRoute,
    simulate_opportunity_with_order_books,
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
ENABLE_EXECUTION_DEFAULT = True
LIVE_TRADING_DEFAULT = True
USE_TESTNET_DEFAULT = False
DISABLE_WEBSOCKET_DEFAULT = False
LOG_LEVEL_DEFAULT = "INFO"
MAX_ROUTE_LENGTH_DEFAULT: Optional[int] = 3
MAX_EXECUTIONS_DEFAULT: Optional[int] = 1
MIN_PROFIT_PERCENTAGE_DEFAULT = 0.1
EVALUATION_INTERVAL_DEFAULT = 30.0
WEBSOCKET_TIMEOUT_DEFAULT = 1_000_000.0
PRICE_REFRESH_INTERVAL_DEFAULT = 3_000_000.0
PRICE_MAX_AGE_DEFAULT = 60.0
SLIPPAGE_USAGE_FRACTION_DEFAULT = 0.10
PARTIAL_FILL_MODE_DEFAULT = "staggered"
STAGGERED_LEG_DELAY_DEFAULT = 0.04
STAGGERED_SLIPPAGE_ASSUMPTION_DEFAULT = 0.01
PRE_TRADE_SLIPPAGE_ENABLED_DEFAULT = False
PRE_TRADE_SLIPPAGE_DEPTH_DEFAULT = 10
MIN_DAILY_VOLUME_DEFAULT = 0.0
REFRESH_TRADING_FEES_DEFAULT = True
ENABLE_BENCHMARKING_DEFAULT = False
BENCHMARK_INTERVAL_DEFAULT = 1.0
ASSET_FILTER_DEFAULT: Sequence[str] = ("USD",
                                       "USDC","USDT","USDG",
                                       "BTC","ETH","SOL","DOGE","ADA","XRP",
                                       "LTC","BCH","SUI","LINK","SEI","TAO","PEPE","WIF","SHIB",
                                       "APT","TIA","AAVE","NEAR","DOT","AVAX","ICP","UNI","INJ",
                                       "FLR","PAXG","TRX","XLM","FIL","ARB","LDO","OP","STRK",
                                       "ATOM","GRT","ALGO","KAVA","MINA","KSM",
                                       "ZEC","PAXG","CHF","BNB","SPX","ENA","STBL","ZORA","USELESS",
                                       "FARTCOIN","AVAX","XPL","PUMP","KAS","CRV","DASH"
                                       )

# Location where executed trades will be persisted when trade logging is enabled.
TRADE_LOG_PATH_DEFAULT: Optional[str] = "../../../data/logs/triangular_trades.csv"

# Optional configuration file used to source API credentials when present.
DEFAULT_CONFIG_PATH = "../../config/exchange_config.yaml"
CONFIG_SECTION_BY_EXCHANGE = {
    "kraken": "kraken_websocket",
    "bitmex": "Bitmex",
    "coinbase": "Coinbase"
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
    "low_volume": (
        "Reported 24h volume fell below the configured minimum, suggesting insufficient liquidity for "
        "reliable execution."
    ),
}


@dataclass
class BenchmarkSample:
    """Single resource usage sample collected during benchmarking."""

    timestamp: float
    cpu_percent: float
    rss_bytes: int
    threads: int


class BenchmarkRecorder:
    """Collect periodic CPU and memory metrics for the current process."""

    def __init__(self, *, enabled: bool, interval: float = 1.0) -> None:
        self.enabled = bool(enabled)
        self.interval = max(interval, 0.1)
        self.samples: List[BenchmarkSample] = []
        self._process: Optional["psutil.Process"] = None
        self._stop_event: Optional[asyncio.Event] = None
        self._task: Optional[asyncio.Task[None]] = None
        self._start_time: float = 0.0
        self.logical_cpus: Optional[int] = None
        self.physical_cpus: Optional[int] = None

    async def start(self) -> None:
        """Begin sampling if benchmarking is enabled and psutil is available."""

        if not self.enabled:
            return
        if psutil is None:
            logger.warning(
                "Benchmarking requested but psutil is not installed; install psutil to enable metrics."
            )
            self.enabled = False
            return

        self._process = psutil.Process()
        self._process.cpu_percent(interval=None)  # Prime the internal counters.
        self.logical_cpus = psutil.cpu_count(logical=True) or os.cpu_count()
        self.physical_cpus = psutil.cpu_count(logical=False)
        self._start_time = perf_counter()
        self.samples.clear()
        self._stop_event = asyncio.Event()
        # Record a baseline memory sample without CPU usage to anchor the series.
        self._record_sample(include_cpu=False)
        self._task = asyncio.create_task(self._run_loop(), name="benchmark-recorder")

        logger.info(
            "Benchmarking enabled: interval=%.2fs logical_cpu=%s physical_cpu=%s",
            self.interval,
            self.logical_cpus if self.logical_cpus is not None else "unknown",
            self.physical_cpus if self.physical_cpus is not None else "unknown",
        )

    async def stop(self) -> None:
        """Stop sampling and await the background task."""

        if not self.enabled:
            return
        if self._stop_event and not self._stop_event.is_set():
            self._stop_event.set()
        if self._task:
            await self._task
        self._task = None
        self._stop_event = None

    async def _run_loop(self) -> None:
        assert self._stop_event is not None  # For type-checkers
        try:
            while True:
                try:
                    await asyncio.wait_for(self._stop_event.wait(), timeout=self.interval)
                    break
                except asyncio.TimeoutError:
                    self._record_sample()
        finally:
            # Capture a final snapshot to report end-of-run memory usage.
            self._record_sample()

    def _record_sample(self, *, include_cpu: bool = True) -> None:
        if not self.enabled or self._process is None:
            return
        try:
            with self._process.oneshot():
                cpu_percent = (
                    self._process.cpu_percent(interval=None) if include_cpu else 0.0
                )
                rss_bytes = self._process.memory_info().rss
                threads = self._process.num_threads()
        except Exception as exc:  # pragma: no cover - sampling is best effort
            logger.debug("Benchmark recorder failed to sample metrics: %s", exc)
            return
        timestamp = perf_counter() - self._start_time
        self.samples.append(
            BenchmarkSample(
                timestamp=timestamp,
                cpu_percent=cpu_percent,
                rss_bytes=rss_bytes,
                threads=threads,
            )
        )

    def summary(self) -> Optional[Dict[str, float]]:
        """Return aggregate statistics for the collected samples."""

        if not self.enabled or not self.samples:
            return None

        cpu_values = [sample.cpu_percent for sample in self.samples]
        rss_values = [sample.rss_bytes for sample in self.samples]
        thread_values = [sample.threads for sample in self.samples]

        total_samples = len(self.samples)
        duration = self.samples[-1].timestamp if self.samples else 0.0

        def _average(values: List[float]) -> float:
            return float(sum(values) / len(values)) if values else 0.0

        summary: Dict[str, float] = {
            "samples": float(total_samples),
            "duration": duration,
            "cpu_avg": _average(cpu_values),
            "cpu_max": max(cpu_values) if cpu_values else 0.0,
            "rss_avg": _average(rss_values),
            "rss_max": max(rss_values) if rss_values else 0.0,
            "threads_avg": _average(thread_values),
            "threads_max": float(max(thread_values)) if thread_values else 0.0,
            "logical_cpus": float(self.logical_cpus or os.cpu_count() or 0),
        }
        if self.physical_cpus is not None:
            summary["physical_cpus"] = float(self.physical_cpus)
        return summary


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


@dataclass
class SlippageDecision:
    """Outcome of replaying an opportunity against live order books."""

    simulation: SlippageSimulation
    total_slippage_pct: float
    scale: float
    max_leg_slippage_pct: float
    max_leg_output_slippage_pct: float


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


def _extract_daily_volume(metadata: Dict[str, object]) -> Optional[float]:
    """Extract the reported 24-hour volume from market metadata when available."""

    candidates: List[object] = []
    for key in ("quoteVolume", "baseVolume", "volume"):
        if key in metadata:
            candidates.append(metadata.get(key))

    info = metadata.get("info")
    if isinstance(info, dict):
        for key in ("quoteVolume", "baseVolume", "volume"):
            if key in info:
                candidates.append(info.get(key))

    for raw_value in candidates:
        if raw_value in (None, ""):
            continue
        try:
            numeric = float(raw_value)
        except (TypeError, ValueError):
            continue
        if numeric > 0:
            return numeric
    return None


def filter_markets_for_triangular_routes(
    markets: Dict[str, Dict[str, object]],
    *,
    starting_currencies: Optional[Sequence[str]] = None,
    asset_filter: Optional[Sequence[str]] = None,
    min_daily_volume: float = MIN_DAILY_VOLUME_DEFAULT,
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

        if min_daily_volume > 0.0:
            volume = _extract_daily_volume(metadata)
            if volume is not None and volume < min_daily_volume:
                skipped["low_volume"] += 1
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

    candidate_paths: List[Path] = []
    if config_path:
        try:
            candidate_paths.append(Path(config_path).expanduser())
        except TypeError:
            logger.warning(
                f"Invalid config path {config_path!r} supplied; skipping credential load."
            )
            return {}
    else:
        default_path = DEFAULT_CONFIG_PATH
        candidate_paths.append(default_path)
        cwd_candidate = Path.cwd() / "config" / "exchange_config.yaml"
        if cwd_candidate != default_path:
            candidate_paths.append(cwd_candidate)

    config_file: Optional[Path] = None
    for path in candidate_paths:
        if isinstance(path, str):  # pragma: no cover - defensive for unexpected types
            path = Path(path)
        if path.exists():
            config_file = path
            break
        logger.debug(f"Config path {path} does not exist; skipping credential load")

    if config_file is None:
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
    exchange: ExchangeConnection,
    routes: Sequence[TriangularRoute],
    *,
    price_cache: Dict[str, CachedPrice],
    markets: Dict[str, Dict[str, object]],
    fee_lookup: Dict[str, float],
    starting_amount: float,
    min_profit_percentage: float,
    max_route_length: Optional[int],
    max_executions: Optional[int],
    evaluation_interval: float,
    enable_execution: bool,
    price_max_age: float,
    pre_trade_slippage: bool,
    pre_trade_slippage_depth: int,
    slippage_action: str,
    max_slippage_percentage: Optional[float],
    slippage_order_book_depth: int,
    slippage_min_scale: float,
    slippage_scale_steps: int,
    slippage_scale_tolerance: float,
    slippage_usage_fraction: float,
    trigger_queue: "asyncio.Queue[str]",
    stop_event: asyncio.Event,
) -> None:
    """Evaluate cached prices and execute profitable opportunities."""

    last_execution: Optional[OpportunityExecution] = None
    execution_limit: Optional[int] = None
    if max_executions is not None and max_executions > 0:
        execution_limit = int(max_executions)
    executed_routes = 0

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

        raw_best = opportunities[0]
        logger.info(
            f"Best opportunity: route={' -> '.join(raw_best.route.symbols)} "
            f"profit={raw_best.profit:.6f} ({raw_best.profit_percentage:.4f}%) "
            f"profit_without_fees={raw_best.profit_without_fees:.6f} "
            f"fee_impact={raw_best.fee_impact:.6f} {raw_best.route.starting_currency}"
        )

        best = raw_best
        price_ages: Dict[str, float] = {}
        now_wall = time.time()
        for symbol in best.route.symbols:
            snapshot = fresh_prices.get(symbol)
            if snapshot is None:
                continue
            age = max(now_wall - snapshot.timestamp, 0.0)
            price_ages[symbol] = age
        if price_ages:
            min_age = min(price_ages.values())
            max_age = max(price_ages.values())
            avg_age = sum(price_ages.values()) / len(price_ages)
            formatted = ", ".join(
                f"{symbol}:{age * 1000.0:.1f}ms" for symbol, age in sorted(price_ages.items())
            )
            logger.info(
                "Price staleness for route %s: min=%.3fs avg=%.3fs max=%.3fs (%s)",
                " -> ".join(best.route.symbols),
                min_age,
                avg_age,
                max_age,
                formatted,
            )
        slippage_decision: Optional[SlippageDecision] = None
        order_books: Optional[Dict[str, OrderBookSnapshot]] = None
        precision_adapter: Optional[PrecisionAdapter] = None

        need_order_books = pre_trade_slippage or slippage_action != "ignore"
        if need_order_books:
            precision_adapter = PrecisionAdapter(
                amount_to_precision=exchange.amount_to_precision,
                cost_to_precision=exchange.cost_to_precision,
            )
            fetch_depth = 0
            if pre_trade_slippage:
                fetch_depth = max(fetch_depth, max(int(pre_trade_slippage_depth), 1))
            if slippage_action != "ignore":
                fetch_depth = max(fetch_depth, max(int(slippage_order_book_depth), 1))
            try:
                order_books = {}
                for symbol in raw_best.route.symbols:
                    order_books[symbol] = await asyncio.to_thread(
                        exchange.get_order_book,
                        symbol,
                        limit=fetch_depth,
                    )
            except Exception as exc:
                contexts: List[str] = []
                if pre_trade_slippage:
                    contexts.append("pre-trade slippage")
                if slippage_action != "ignore":
                    contexts.append("slippage")
                reason = " and ".join(contexts) if contexts else "slippage"
                logger.info(
                    "Skipping opportunity because order book retrieval failed while estimating %s for %s: %s",
                    reason,
                    " -> ".join(raw_best.route.symbols),
                    exc,
                )
                continue

        if pre_trade_slippage:
            assert order_books is not None  # for type-checkers
            try:
                pre_trade_simulation = simulate_opportunity_with_order_books(
                    raw_best,
                    order_books,
                    starting_amount=raw_best.starting_amount,
                    precision=precision_adapter,
                )
            except InsufficientLiquidityError as exc:
                logger.info(
                    "Skipping opportunity because pre-trade slippage estimation reported insufficient depth for %s: %s",
                    " -> ".join(raw_best.route.symbols),
                    exc,
                )
                continue
            except Exception as exc:
                logger.info(
                    "Skipping opportunity because pre-trade slippage estimation failed for %s: %s",
                    " -> ".join(raw_best.route.symbols),
                    exc,
                )
                continue
            expected_final = raw_best.final_amount
            actual_final = pre_trade_simulation.opportunity.final_amount
            total_slippage = (
                ((expected_final - actual_final) / expected_final) * 100.0
                if expected_final
                else 0.0
            )
            max_leg_slippage = max(
                (leg.slippage_pct for leg in pre_trade_simulation.legs),
                default=0.0,
            )
            max_output_slippage = max(
                (leg.output_slippage_pct for leg in pre_trade_simulation.legs),
                default=0.0,
            )
            logger.info(
                "Pre-trade slippage estimate for %s: total %.4f%% (max leg price %.4f%% / output %.4f%%); "
                "adjusted profit %.6f (%.4f%%)",
                " -> ".join(raw_best.route.symbols),
                total_slippage,
                max_leg_slippage,
                max_output_slippage,
                pre_trade_simulation.opportunity.profit,
                pre_trade_simulation.opportunity.profit_percentage,
            )
            best = pre_trade_simulation.opportunity

        if slippage_action != "ignore":
            if order_books is None or precision_adapter is None:
                precision_adapter = PrecisionAdapter(
                    amount_to_precision=exchange.amount_to_precision,
                    cost_to_precision=exchange.cost_to_precision,
                )
                try:
                    order_books = {}
                    depth = max(int(slippage_order_book_depth), 1)
                    for symbol in raw_best.route.symbols:
                        order_books[symbol] = await asyncio.to_thread(
                            exchange.get_order_book,
                            symbol,
                            limit=depth,
                        )
                except Exception as exc:
                    logger.info(
                        "Skipping opportunity because order book retrieval failed while estimating slippage for %s: %s",
                        " -> ".join(raw_best.route.symbols),
                        exc,
                    )
                    continue

            def _simulate_scale(scale: float) -> SlippageDecision:
                assert order_books is not None
                assert precision_adapter is not None
                simulation = simulate_opportunity_with_order_books(
                    raw_best,
                    order_books,
                    starting_amount=raw_best.starting_amount * scale,
                    precision=precision_adapter,
                )
                expected_final = raw_best.final_amount * scale
                actual_final = simulation.opportunity.final_amount
                total_slippage = (
                    ((expected_final - actual_final) / expected_final) * 100.0
                    if expected_final
                    else 0.0
                )
                max_leg_slippage = max(
                    (leg.slippage_pct for leg in simulation.legs),
                    default=0.0,
                )
                max_output_slippage = max(
                    (leg.output_slippage_pct for leg in simulation.legs),
                    default=0.0,
                )
                return SlippageDecision(
                    simulation,
                    total_slippage,
                    scale,
                    max_leg_slippage,
                    max_output_slippage,
                )

            def _decision_within_threshold(
                decision: SlippageDecision,
                threshold_pct: float,
            ) -> bool:
                epsilon = 1e-9
                limit = max(threshold_pct, 0.0)
                if limit == 0.0:
                    return (
                        decision.total_slippage_pct <= epsilon
                        and decision.max_leg_slippage_pct <= epsilon
                        and decision.max_leg_output_slippage_pct <= epsilon
                    )
                boundary = limit + epsilon
                return (
                    decision.total_slippage_pct <= boundary
                    and decision.max_leg_slippage_pct <= boundary
                    and decision.max_leg_output_slippage_pct <= boundary
                )

            initial_decision: Optional[SlippageDecision]
            initial_error: Optional[Exception]
            try:
                initial_decision = _simulate_scale(1.0)
            except InsufficientLiquidityError as exc:
                initial_decision = None
                initial_error = exc
            except Exception as exc:
                logger.info(
                    "Skipping opportunity after slippage simulation failed for scale 1.0: %s",
                    exc,
                )
                continue
            else:
                initial_error = None

            threshold = max_slippage_percentage if max_slippage_percentage is not None else 0.0

            if initial_decision is None:
                if slippage_action != "scale":
                    logger.info(
                        "Skipping opportunity because available depth could not satisfy the planned trade size: %s",
                        initial_error,
                    )
                    continue
            elif _decision_within_threshold(initial_decision, threshold):
                slippage_decision = initial_decision
            elif slippage_action == "reject":
                logger.info(
                    "Skipping opportunity because estimated slippage %.4f%% (max leg price %.4f%% / size %.4f%%) exceeds configured maximum %.4f%%.",
                    initial_decision.total_slippage_pct,
                    initial_decision.max_leg_slippage_pct,
                    initial_decision.max_leg_output_slippage_pct,
                    threshold,
                )
                continue

            if slippage_decision is None and slippage_action == "scale":
                min_scale = max(min(slippage_min_scale, 1.0), 0.0)
                tolerance = max(slippage_scale_tolerance, 1e-4)
                steps = max(slippage_scale_steps, 1)
                low = min_scale
                high = 1.0
                best_candidate: Optional[SlippageDecision] = None

                for _ in range(steps):
                    if high - low <= tolerance:
                        break
                    trial = (low + high) / 2.0
                    if trial <= 0:
                        break
                    try:
                        candidate = _simulate_scale(trial)
                    except InsufficientLiquidityError:
                        high = trial
                        continue
                    except Exception as exc:
                        logger.debug(
                            "Slippage simulation failed for scale %.4f: %s",
                            trial,
                            exc,
                        )
                        high = trial
                        continue
                    if _decision_within_threshold(candidate, threshold):
                        best_candidate = candidate
                        low = trial
                    else:
                        logger.debug(
                            "Rejected scale %.2f%%: total slippage %.4f%%, max leg price %.4f%%, max leg size %.4f%% (threshold %.4f%%)",
                            trial * 100.0,
                            candidate.total_slippage_pct,
                            candidate.max_leg_slippage_pct,
                            candidate.max_leg_output_slippage_pct,
                            threshold,
                        )
                        high = trial

                if best_candidate is None and min_scale > 0.0:
                    try:
                        candidate = _simulate_scale(min_scale)
                    except InsufficientLiquidityError:
                        candidate = None
                    except Exception as exc:
                        logger.debug(
                            "Slippage simulation failed for minimum scale %.4f: %s",
                            min_scale,
                            exc,
                        )
                        candidate = None
                    if candidate and _decision_within_threshold(candidate, threshold):
                        best_candidate = candidate

                if best_candidate is None:
                    reason = (
                        f"insufficient depth ({initial_error})"
                        if initial_decision is None
                        else (
                            "slippage total %.4f%% / max leg price %.4f%% / size %.4f%% > %.4f%%"
                            % (
                                initial_decision.total_slippage_pct,
                                initial_decision.max_leg_slippage_pct,
                                initial_decision.max_leg_output_slippage_pct,
                                threshold,
                            )
                        )
                    )
                    logger.info(
                        "Skipping opportunity because slippage remained above %.4f%% even after scaling down to %.2f%% of the starting amount (%s).",
                        threshold,
                        min_scale * 100,
                        reason,
                    )
                    continue

                slippage_decision = best_candidate

        if slippage_decision is not None:
            final_decision = slippage_decision
            worst_leg_slippage = max(
                (
                    leg.output_slippage_pct
                    for leg in final_decision.simulation.legs
                ),
                default=0.0,
            )
            if worst_leg_slippage > 0.0:
                worst_factor = max(0.0, 1.0 - worst_leg_slippage / 100.0)
                if worst_factor <= 0.0:
                    logger.info(
                        "Skipping opportunity because worst-leg slippage %.4f%% leaves no executable size.",
                        worst_leg_slippage,
                    )
                    continue
                worst_scale = final_decision.scale * worst_factor
                try:
                    worst_adjusted = _simulate_scale(worst_scale)
                except InsufficientLiquidityError:
                    logger.info(
                        "Skipping opportunity because the order book cannot satisfy the worst-leg slippage %.4f%% adjustment.",
                        worst_leg_slippage,
                    )
                    continue
                except Exception as exc:
                    logger.info(
                        "Skipping opportunity after applying worst-leg slippage %.4f%% scaling: %s",
                        worst_leg_slippage,
                        exc,
                    )
                    continue
                if not _decision_within_threshold(worst_adjusted, threshold):
                    logger.info(
                        "Skipping opportunity because worst-leg slippage %.4f%% scaling resulted in total slippage %.4f%% (max leg price %.4f%% / size %.4f%%) above %.4f%%.",
                        worst_leg_slippage,
                        worst_adjusted.total_slippage_pct,
                        worst_adjusted.max_leg_slippage_pct,
                        worst_adjusted.max_leg_output_slippage_pct,
                        threshold,
                    )
                    continue
                if worst_adjusted.scale <= 0.0:
                    logger.info(
                        "Skipping opportunity because worst-leg slippage %.4f%% reduced executable size to zero.",
                        worst_leg_slippage,
                    )
                    continue
                if worst_adjusted.scale < final_decision.scale - 1e-9:
                    logger.info(
                        "Worst-leg slippage %.4f%% reduced executable scale from %.2f%% to %.2f%%.",
                        worst_leg_slippage,
                        final_decision.scale * 100.0,
                        worst_adjusted.scale * 100.0,
                    )
                final_decision = worst_adjusted
                usage_fraction = max(min(slippage_usage_fraction, 1.0), 0.0)
                if usage_fraction <= 0.0:
                    logger.info(
                        "Skipping opportunity because configured slippage usage fraction is 0%% of the adjusted size.",
                    )
                    continue
                if usage_fraction < 1.0:
                    final_scale = final_decision.scale * usage_fraction
                    try:
                        usage_adjusted = _simulate_scale(final_scale)
                    except InsufficientLiquidityError:
                        logger.info(
                            "Skipping opportunity because the order book cannot satisfy the reserved usage fraction %.2f%%.",
                            usage_fraction * 100.0,
                        )
                        continue
                    except Exception as exc:
                        logger.info(
                            "Skipping opportunity after applying slippage usage fraction %.2f%%: %s",
                            usage_fraction * 100.0,
                            exc,
                        )
                        continue
                    if not _decision_within_threshold(usage_adjusted, threshold):
                        logger.info(
                            "Skipping opportunity because usage fraction %.2f%% resulted in slippage %.4f%% (max leg price %.4f%% / size %.4f%%) above the %.4f%% limit.",
                            usage_fraction * 100.0,
                            usage_adjusted.total_slippage_pct,
                            usage_adjusted.max_leg_slippage_pct,
                            usage_adjusted.max_leg_output_slippage_pct,
                            threshold,
                        )
                        continue
                    logger.info(
                        "Additional usage reservation reduced executable scale from %.2f%% to %.2f%% (usage fraction %.2f%%).",
                        final_decision.scale * 100.0,
                        usage_adjusted.scale * 100.0,
                        usage_fraction * 100.0,
                    )
                    final_decision = usage_adjusted

                slippage_decision = final_decision
                best = final_decision.simulation.opportunity
                logger.info(
                    "Slippage-adjusted plan: scale=%.2f%% final_amount=% .6f profit=% .6f (% .4f%%) total_slippage=% .4f%% (max leg price %.4f%% / size %.4f%%)",
                    slippage_decision.scale * 100.0,
                    best.final_amount,
                    best.profit,
                    best.profit_percentage,
                    slippage_decision.total_slippage_pct,
                    slippage_decision.max_leg_slippage_pct,
                    slippage_decision.max_leg_output_slippage_pct,
                )
                if not math.isclose(best.starting_amount, raw_best.starting_amount, rel_tol=0.0, abs_tol=1e-9):
                    logger.info(
                        "Starting amount reduced from %.6f to %.6f due to slippage adjustments.",
                        raw_best.starting_amount,
                        best.starting_amount,
                    )
                if slippage_decision.simulation.legs:
                    leg_details = ", ".join(
                        (
                            f"{leg.symbol}:price {leg.slippage_pct:.4f}% size {leg.output_slippage_pct:.4f}%"
                            f" expected_out {leg.expected_amount_out:.10f} -> actual_out {leg.actual_amount_out:.10f}"
                        )
                        for leg in slippage_decision.simulation.legs
                    )
                    logger.info("Per-leg slippage estimates: %s", leg_details)

        planning_ready_at = perf_counter()
        planning_latency = planning_ready_at - evaluation_started_at
        logger.info(
            "Planning latency for %s triggered by %s: %.3fs",
            " -> ".join(best.route.symbols),
            reason_summary,
            planning_latency,
        )

        if best.profit_percentage < min_profit_percentage:
            logger.info(
                "Opportunity rejected after slippage adjustment; profit %.4f%% below minimum %.4f%%.",
                best.profit_percentage,
                min_profit_percentage,
            )
            continue

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
        except ExchangeRequestTimeout as exc:
            logger.warning("Skipping execution due to exchange timeout: %s", exc)
            continue

        last_execution = OpportunityExecution(best.route, profit_signature)
        if orders:
            logger.info(f"Placed {len(orders)} order(s) for opportunity.")
            try:
                trigger_queue.put_nowait("post_trade")
            except asyncio.QueueFull:  # pragma: no cover - unbounded by default
                logger.debug("Trigger queue full; dropping post_trade notification")
        else:
            logger.info("Execution completed without order details from the exchange response.")

        executed_routes += 1
        if execution_limit is not None and executed_routes >= execution_limit:
            logger.info(
                f"Reached configured execution limit of {execution_limit} route(s); stopping runner for review."
            )
            stop_event.set()
            break


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

    credentials = ExchangeConnection._normalise_credentials(credentials)

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
    max_executions = args.max_executions
    if max_executions is not None and max_executions <= 0:
        max_executions = None

    if args.slippage_action != "ignore":
        if args.max_slippage_percentage is None:
            raise SystemExit(
                "--max-slippage-percentage must be provided when --slippage-action is not 'ignore'."
            )
        if args.max_slippage_percentage < 0:
            raise SystemExit("--max-slippage-percentage must be non-negative.")
        if args.slippage_order_book_depth <= 0:
            raise SystemExit("--slippage-order-book-depth must be positive.")
        if args.slippage_action == "scale":
            if not 0 <= args.slippage_scale_min <= 1:
                raise SystemExit("--slippage-scale-min must be between 0 and 1 (inclusive).")
            if args.slippage_scale_steps <= 0:
                raise SystemExit("--slippage-scale-steps must be positive.")
            if args.slippage_scale_tolerance <= 0:
                raise SystemExit("--slippage-scale-tolerance must be positive.")
        if not 0 <= args.slippage_usage_fraction <= 1:
            raise SystemExit("--slippage-usage-fraction must be between 0 and 1 (inclusive).")

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

    benchmark = BenchmarkRecorder(
        enabled=args.enable_benchmarking,
        interval=max(args.benchmark_interval, 0.1),
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
        min_daily_volume=max(float(args.min_daily_volume), 0.0),
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
    staggered_slippage = (
        args.staggered_slippage_assumption
        if getattr(args, "staggered_slippage_assumption", None)
        else [STAGGERED_SLIPPAGE_ASSUMPTION_DEFAULT]
    )

    if args.enable_execution:
        executor = TriangularArbitrageExecutor(
            exchange,
            dry_run=not args.live_trading,
            trade_log_path=trade_log_path,
            partial_fill_mode=args.partial_fill_mode,
            staggered_leg_delay=args.staggered_leg_delay,
            staggered_slippage_assumption=staggered_slippage,
        )
        if trade_log_path:
            logger.info(f"Logging executed trades to {trade_log_path}")

    symbols = sorted({symbol for route in routes for symbol in route.symbols})
    refreshed_fees: Dict[str, float] = {}
    if args.refresh_trading_fees:
        try:
            refreshed_fees = exchange.refresh_trading_fees(symbols)
        except Exception as exc:
            logger.warning(
                "Failed to refresh taker fees via exchange API; continuing with cached rates: %s",
                exc,
            )
        else:
            if refreshed_fees:
                min_refreshed = min(refreshed_fees.values())
                max_refreshed = max(refreshed_fees.values())
                avg_refreshed = sum(refreshed_fees.values()) / len(refreshed_fees)
                logger.info(
                    "Refreshed taker fees for %d symbol(s): min=%.4f%% max=%.4f%% avg=%.4f%%",
                    len(refreshed_fees),
                    min_refreshed * 100.0,
                    max_refreshed * 100.0,
                    avg_refreshed * 100.0,
                )
            else:
                logger.debug("Taker fee refresh completed without overrides from the exchange API.")

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

    await benchmark.start()

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
            exchange,
            routes,
            price_cache=price_cache,
            markets=markets,
            fee_lookup=fee_lookup,
            starting_amount=args.starting_amount,
            min_profit_percentage=args.min_profit_percentage,
            max_route_length=max_route_length,
            max_executions=max_executions,
            evaluation_interval=args.evaluation_interval,
            enable_execution=args.enable_execution,
            price_max_age=price_max_age,
            pre_trade_slippage=args.enable_pre_trade_slippage,
            pre_trade_slippage_depth=max(int(args.pre_trade_slippage_depth), 1),
            slippage_action=args.slippage_action,
            max_slippage_percentage=args.max_slippage_percentage,
            slippage_order_book_depth=args.slippage_order_book_depth,
            slippage_min_scale=args.slippage_scale_min,
            slippage_scale_steps=args.slippage_scale_steps,
            slippage_scale_tolerance=args.slippage_scale_tolerance,
            slippage_usage_fraction=args.slippage_usage_fraction,
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
        await benchmark.stop()
        summary = benchmark.summary()
        if summary:
            rss_avg_mib = summary["rss_avg"] / (1024**2)
            rss_max_mib = summary["rss_max"] / (1024**2)
            logical_cpus = int(summary.get("logical_cpus", 0)) or "unknown"
            physical_cpus = summary.get("physical_cpus")
            if physical_cpus is not None:
                physical_cpus = int(physical_cpus)
            logger.info(
                "Benchmark summary: duration=%.2fs samples=%d CPU avg=%.2f%% max=%.2f%% across %s logical core(s)%s; "
                "RSS avg=%.2f MiB max=%.2f MiB; threads avg=%.1f max=%d",
                summary["duration"],
                int(summary["samples"]),
                summary["cpu_avg"],
                summary["cpu_max"],
                logical_cpus,
                f" (physical {physical_cpus})" if physical_cpus else "",
                rss_avg_mib,
                rss_max_mib,
                summary["threads_avg"],
                int(summary["threads_max"]),
            )
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
        "--min-daily-volume",
        type=float,
        default=MIN_DAILY_VOLUME_DEFAULT,
        help=(
            "Minimum 24h volume required for markets to participate in route discovery. "
            "Values are interpreted in the exchange's reported units (typically quote currency)."
        ),
    )
    parser.add_argument(
        "--refresh-trading-fees",
        action=argparse.BooleanOptionalAction,
        default=REFRESH_TRADING_FEES_DEFAULT,
        help=(
            "Fetch account-specific taker fees from the exchange before evaluating opportunities. "
            "Disable to rely solely on cached market metadata."
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
        default=MIN_PROFIT_PERCENTAGE_DEFAULT,
        help="Minimum profit percentage required before executing an opportunity.",
    )
    parser.add_argument(
        "--max-route-length",
        type=int,
        default=MAX_ROUTE_LENGTH_DEFAULT,
        help="Optional maximum number of legs per route to evaluate.",
    )
    parser.add_argument(
        "--max-executions",
        type=int,
        default=MAX_EXECUTIONS_DEFAULT,
        help=(
            "Stop the runner after this many opportunities have been executed. "
            "Leave unset for no limit."
        ),
    )
    parser.add_argument(
        "--slippage-buffer",
        type=float,
        default=0.0,
        help="Fractional buffer applied after fees to account for slippage (e.g. 0.01 for 1%%).",
    )
    parser.add_argument(
        "--slippage-action",
        choices=["ignore", "reject", "scale"],
        default="ignore",
        help=(
            "Strategy to apply when order book depth indicates slippage: "
            "'ignore' keeps previous behaviour, 'reject' skips trades over the threshold, "
            "and 'scale' reduces the trade size until the slippage requirement is met."
        ),
    )
    parser.add_argument(
        "--max-slippage-percentage",
        type=float,
        default=None,
        help="Maximum acceptable aggregate slippage percentage when evaluating order book depth.",
    )
    parser.add_argument(
        "--slippage-order-book-depth",
        type=int,
        default=20,
        help="Number of order book levels to request when estimating slippage.",
    )
    parser.add_argument(
        "--enable-pre-trade-slippage",
        action=argparse.BooleanOptionalAction,
        default=PRE_TRADE_SLIPPAGE_ENABLED_DEFAULT,
        help=(
            "Estimate order book slippage before execution and adjust the expected profit "
            "accordingly before proceeding."
        ),
    )
    parser.add_argument(
        "--pre-trade-slippage-depth",
        type=int,
        default=PRE_TRADE_SLIPPAGE_DEPTH_DEFAULT,
        help="Order book depth (levels) to request when estimating pre-trade slippage.",
    )
    parser.add_argument(
        "--slippage-scale-min",
        type=float,
        default=0.1,
        help=(
            "Smallest fraction of the configured starting amount to consider when scaling trades "
            "to meet the slippage constraint."
        ),
    )
    parser.add_argument(
        "--slippage-scale-steps",
        type=int,
        default=8,
        help="Maximum iterations to use while searching for an acceptable scaled trade size.",
    )
    parser.add_argument(
        "--slippage-scale-tolerance",
        type=float,
        default=0.02,
        help="Stop scaling once the remaining search range falls below this fraction of the starting amount.",
    )
    parser.add_argument(
        "--slippage-usage-fraction",
        type=float,
        default=SLIPPAGE_USAGE_FRACTION_DEFAULT,
        help=(
            "Fraction of the slippage-adjusted trade size to actually execute to reserve depth for other market "
            "participants (e.g. 0.6 to only use 60%% of the depth that satisfies the slippage constraint)."
        ),
    )
    parser.add_argument(
        "--partial-fill-mode",
        choices=["wait", "progressive", "staggered"],
        default=PARTIAL_FILL_MODE_DEFAULT,
        help=(
            "Behaviour when an order leg is partially filled: 'wait' blocks until completion, "
            "'progressive' streams fills downstream, and 'staggered' submits each leg speculatively "
            "with a configurable delay and reconciles any residual once fills settle."
        ),
    )
    parser.add_argument(
        "--staggered-leg-delay",
        type=float,
        default=STAGGERED_LEG_DELAY_DEFAULT,
        help=(
            "Seconds to wait between submitting consecutive legs when using the 'staggered' "
            "partial-fill mode."
        ),
    )
    parser.add_argument(
        "--staggered-slippage-assumption",
        type=float,
        action="append",
        default=None,
        metavar="FRACTION",
        help=(
            "Assumed fractional slippage per leg when sizing staggered submissions (e.g. 0.01 for 1%%). "
            "Provide once to reuse for all legs or multiple times to set leg-specific values."
        ),
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
        "--enable-benchmarking",
        action=argparse.BooleanOptionalAction,
        default=ENABLE_BENCHMARKING_DEFAULT,
        help="Collect periodic CPU, memory, and thread usage metrics during execution.",
    )
    parser.add_argument(
        "--benchmark-interval",
        type=float,
        default=BENCHMARK_INTERVAL_DEFAULT,
        help="Seconds between benchmark samples when benchmarking is enabled.",
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
        help=(
            "Optional path to an exchange_config.yaml file containing API credentials."
        ),
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

