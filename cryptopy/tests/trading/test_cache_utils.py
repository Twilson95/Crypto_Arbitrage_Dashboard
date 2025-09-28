import pandas as pd
import pytest

from cryptopy.src.trading.cache_utils import PairAnalyticsCache
from cryptopy.src.trading.ArbitrageSimulator import ArbitrageSimulator


class DummyPortfolioManager:
    pass


def _store_spread(cache: PairAnalyticsCache, pair, date, value):
    spread_series = pd.Series([value], index=[pd.Timestamp(date)])
    cache.store(
        pair,
        pd.Timestamp(date),
        p_value=0.05,
        hedge_ratio=1.0,
        spread=spread_series,
    )


def test_get_spread_series_returns_recent_values(tmp_path):
    cache = PairAnalyticsCache(tmp_path)
    pair = ("BTC/USD", "ETH/USD")

    _store_spread(cache, pair, "2024-01-01", 1.0)
    _store_spread(cache, pair, "2024-01-02", 2.0)
    _store_spread(cache, pair, "2024-01-03", 3.0)

    series = cache.get_spread_series(pair, rolling_window=2)

    assert list(series.index) == [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")]
    assert list(series.values) == [2.0, 3.0]


def test_get_cached_spread_metrics_uses_cached_series(tmp_path):
    cache = PairAnalyticsCache(tmp_path)
    pair = ("BTC/USD", "ETH/USD")

    _store_spread(cache, pair, "2024-01-01", 1.0)
    _store_spread(cache, pair, "2024-01-02", 2.0)

    parameters = {
        "rolling_window": 2,
        "spread_threshold": 1.0,
        "spread_limit": 2.0,
        "expected_holding_days": 0,
    }

    simulator = ArbitrageSimulator(
        parameters=parameters,
        price_df=pd.DataFrame(),
        volume_df=pd.DataFrame(),
        portfolio_manager=DummyPortfolioManager(),
        pair_combinations=[pair],
        pair_analytics_cache=cache,
    )

    current_date = pd.Timestamp("2024-01-02")
    cached_spread = cache.load(pair, current_date)["spread"]

    metrics = simulator._get_cached_spread_metrics(pair, current_date, cached_spread)

    spread_mean = metrics["spread_mean"].dropna()
    assert not spread_mean.empty
    assert spread_mean.iloc[-1] == pytest.approx(1.5)
