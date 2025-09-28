import pandas as pd
import pytest

from cryptopy.src.trading.cache_utils import PairAnalyticsCache


def _store_spread(cache: PairAnalyticsCache, pair, date, value):
    spread_series = pd.Series([value], index=[pd.Timestamp(date)])
    cache.store(
        pair,
        pd.Timestamp(date),
        p_value=0.05,
        hedge_ratio=1.0,
        spread=spread_series,
    )


def test_get_spread_series_returns_empty_series(tmp_path):
    cache = PairAnalyticsCache(tmp_path)
    pair = ("BTC/USD", "ETH/USD")

    _store_spread(cache, pair, "2024-01-01", 1.0)

    series = cache.get_spread_series(pair, rolling_window=2)

    assert series.empty


def test_store_and_load_summary_values(tmp_path):
    cache = PairAnalyticsCache(tmp_path)
    pair = ("BTC/USD", "ETH/USD")
    current_date = pd.Timestamp("2024-01-02")

    spread_series = pd.Series([0.5, 0.6], index=[current_date - pd.Timedelta(days=1), current_date])
    cache.store(
        pair,
        current_date,
        p_value=0.12,
        hedge_ratio=1.5,
        spread=spread_series,
        coint_stat=-3.5,
        crit_values=(0.1, 0.2, 0.3),
    )

    cached = cache.load(pair, current_date)
    assert cached is not None
    assert cached["p_value"] == pytest.approx(0.12)
    assert cached["hedge_ratio"] == pytest.approx(1.5)
    assert cached["coint_stat"] == pytest.approx(-3.5)
    assert cached["crit_values"] == (0.1, 0.2, 0.3)


def test_load_summary_table_vectorises_date_normalisation(tmp_path):
    cache_dir = tmp_path / "analytics"
    cache_dir.mkdir()
    summary_path = cache_dir / "pair_analytics.csv"

    rows = [
        {
            "pair_key": "PAIR_A",
            "date_key": "2024-01-01 00:00:00",
            "p_value": 0.1,
            "hedge_ratio": 1.0,
            "coint_stat": 0.0,
            "crit_value_1": 0.0,
            "crit_value_2": 0.0,
            "crit_value_3": 0.0,
        },
        {
            "pair_key": "PAIR_A",
            "date_key": "2024-01-02 00:00:00+00:00",
            "p_value": 0.2,
            "hedge_ratio": 1.1,
            "coint_stat": 0.1,
            "crit_value_1": 0.1,
            "crit_value_2": 0.1,
            "crit_value_3": 0.1,
        },
        {
            "pair_key": "PAIR_B",
            "date_key": "20240103000000",
            "p_value": 0.3,
            "hedge_ratio": 1.2,
            "coint_stat": 0.2,
            "crit_value_1": 0.2,
            "crit_value_2": 0.2,
            "crit_value_3": 0.2,
        },
    ]

    pd.DataFrame(rows).to_csv(summary_path, index=False)

    cache = PairAnalyticsCache(cache_dir)

    expected_index = [
        ("PAIR_A", "2024-01-01 00:00:00"),
        ("PAIR_A", "2024-01-02 00:00:00+00:00"),
        ("PAIR_B", "2024-01-03 00:00:00"),
    ]

    assert list(cache._summary_df.index) == expected_index


def test_read_only_cache_does_not_persist_changes(tmp_path):
    cache_dir = tmp_path / "analytics"
    cache = PairAnalyticsCache(cache_dir)
    pair = ("BTC/USD", "ETH/USD")
    date = pd.Timestamp("2024-01-01")

    cache.store(
        pair,
        date,
        p_value=0.05,
        hedge_ratio=1.0,
        spread=pd.Series([0.1], index=[date]),
    )

    summary_path = cache.summary_path
    original_contents = summary_path.read_text()
    assert "BTC_USD__ETH_USD" in original_contents

    read_only_cache = PairAnalyticsCache(cache_dir, read_only=True)
    assert list(read_only_cache._summary_df.index) == [
        ("BTC_USD__ETH_USD", "2024-01-01 00:00:00")
    ]
    assert read_only_cache.load(pair, date) is not None

    read_only_cache.store(
        pair,
        pd.Timestamp("2024-01-02"),
        p_value=0.06,
        hedge_ratio=0.9,
        spread=pd.Series([0.2], index=[pd.Timestamp("2024-01-02")]),
    )

    assert summary_path.read_text() == original_contents
