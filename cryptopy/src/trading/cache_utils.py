from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import pandas as pd

from cryptopy.src.arbitrage.CointegrationCalculator import CointegrationCalculator
from cryptopy.scripts.simulations.simulation_helpers import filter_df


class PairAnalyticsCache:
    """Cache cointegration and spread analytics for pair/day combinations.

    Cached results are stored in tabular CSV files so the entire cache can be
    loaded in a single read and treated like a lookup table. One table stores
    the scalar metrics (p-values, hedge ratios, etc.) while another stores the
    spread time-series in long format.
    """

    _SUMMARY_COLUMNS = [
        "pair_key",
        "date_key",
        "p_value",
        "hedge_ratio",
        "coint_stat",
        "crit_value_1",
        "crit_value_2",
        "crit_value_3",
    ]

    _SPREAD_COLUMNS = ["pair_key", "date_key", "timestamp", "value"]

    def __init__(self, cache_dir: Optional[Path | str] = None):
        if cache_dir is None:
            cache_dir = Path(".cache") / "pair_analytics"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.summary_path = self.cache_dir / "pair_analytics.csv"
        self.spread_path = self.cache_dir / "pair_spreads.csv"

        self._summary_df = self._load_summary_table()
        self._spread_df = self._load_spread_table()

    @staticmethod
    def _pair_key(pair: Tuple[str, str]) -> str:
        return "__".join(symbol.replace("/", "_") for symbol in pair)

    @staticmethod
    def _date_key(current_date) -> str:
        timestamp = pd.Timestamp(current_date)
        if timestamp.tzinfo is not None:
            timestamp = timestamp.tz_convert(None)
        return timestamp.strftime("%Y%m%d%H%M%S")

    @staticmethod
    def _normalise_timestamp(value) -> pd.Timestamp:
        timestamp = pd.Timestamp(value)
        if timestamp.tzinfo is not None:
            timestamp = timestamp.tz_convert(None)
        return timestamp

    def _load_summary_table(self) -> pd.DataFrame:
        if self.summary_path.exists():
            df = pd.read_csv(self.summary_path)
        else:
            df = pd.DataFrame(columns=self._SUMMARY_COLUMNS)
        if df.empty:
            index = pd.MultiIndex(levels=[[], []], codes=[[], []], names=["pair_key", "date_key"])
            return pd.DataFrame(columns=self._SUMMARY_COLUMNS[2:], index=index)
        df = df[self._SUMMARY_COLUMNS]
        df = df.set_index(["pair_key", "date_key"])
        return df

    def _load_spread_table(self) -> pd.DataFrame:
        if self.spread_path.exists():
            df = pd.read_csv(self.spread_path)
        else:
            df = pd.DataFrame(columns=self._SPREAD_COLUMNS)
        if df.empty:
            return pd.DataFrame(columns=self._SPREAD_COLUMNS)
        df = df[self._SPREAD_COLUMNS]
        return df

    def _persist_summary(self) -> None:
        summary_to_write = self._summary_df.reset_index()
        summary_to_write.to_csv(self.summary_path, index=False)

    def _persist_spreads(self) -> None:
        self._spread_df.to_csv(self.spread_path, index=False)

    def _persist(self) -> None:
        self._persist_summary()
        self._persist_spreads()

    def _get_spread_series(self, pair_key: str, date_key: str) -> Optional[pd.Series]:
        mask = (self._spread_df["pair_key"] == pair_key) & (
            self._spread_df["date_key"] == date_key
        )
        spread_rows = self._spread_df.loc[mask]
        if spread_rows.empty:
            return None
        spread_rows = spread_rows.sort_values("timestamp")
        index = pd.to_datetime(spread_rows["timestamp"].tolist())
        values = spread_rows["value"].astype(float).tolist()
        return pd.Series(values, index=index)

    def load(self, pair: Tuple[str, str], current_date) -> Optional[Dict[str, Any]]:
        pair_key = self._pair_key(pair)
        date_key = self._date_key(current_date)
        if (pair_key, date_key) not in self._summary_df.index:
            return None

        summary_row = self._summary_df.loc[(pair_key, date_key)]
        spread_series = self._get_spread_series(pair_key, date_key)
        if spread_series is None:
            return None

        crit_values = summary_row[["crit_value_1", "crit_value_2", "crit_value_3"]].tolist()
        crit_values = [value for value in crit_values if pd.notna(value)]
        crit_values_tuple = tuple(crit_values) if crit_values else None

        return {
            "p_value": summary_row.get("p_value"),
            "hedge_ratio": summary_row.get("hedge_ratio"),
            "coint_stat": summary_row.get("coint_stat"),
            "crit_values": crit_values_tuple,
            "spread": spread_series,
        }

    def store(
        self,
        pair: Tuple[str, str],
        current_date,
        *,
        p_value: float,
        hedge_ratio: float,
        spread: pd.Series,
        coint_stat: Optional[float] = None,
        crit_values: Optional[Tuple[float, float, float]] = None,
    ) -> None:
        pair_key = self._pair_key(pair)
        date_key = self._date_key(current_date)

        summary_data = {
            "pair_key": pair_key,
            "date_key": date_key,
            "p_value": None if p_value is None else float(p_value),
            "hedge_ratio": None if hedge_ratio is None else float(hedge_ratio),
            "coint_stat": None if coint_stat is None else float(coint_stat),
            "crit_value_1": None,
            "crit_value_2": None,
            "crit_value_3": None,
        }

        if crit_values is not None:
            crit_values = tuple(float(value) for value in crit_values)
            for idx, value in enumerate(crit_values[:3]):
                summary_data[f"crit_value_{idx + 1}"] = value

        new_summary_row = pd.DataFrame([summary_data]).set_index(["pair_key", "date_key"])
        if (pair_key, date_key) in self._summary_df.index:
            self._summary_df = self._summary_df.drop(index=(pair_key, date_key))
        self._summary_df = pd.concat([self._summary_df, new_summary_row])

        spread = spread.sort_index()
        spread_df = pd.DataFrame(
            {
                "pair_key": pair_key,
                "date_key": date_key,
                "timestamp": [self._normalise_timestamp(idx).isoformat() for idx in spread.index],
                "value": [float(value) if pd.notna(value) else float("nan") for value in spread.values],
            }
        )

        mask = (self._spread_df["pair_key"] == pair_key) & (
            self._spread_df["date_key"] == date_key
        )
        if mask.any():
            self._spread_df = self._spread_df.loc[~mask].copy()
        self._spread_df = pd.concat([self._spread_df, spread_df], ignore_index=True)

        self._persist()

    def ensure(
        self,
        pair: Tuple[str, str],
        current_date,
        price_df_filtered: pd.DataFrame,
    ) -> Optional[Dict[str, Any]]:
        cached = self.load(pair, current_date)
        if cached is not None:
            return cached

        coint_stat, p_value, crit_values = CointegrationCalculator.test_cointegration(
            price_df_filtered, pair
        )
        if p_value is None:
            return None

        spread, hedge_ratio = CointegrationCalculator.calculate_spread(
            price_df_filtered, pair
        )
        self.store(
            pair,
            current_date,
            p_value=p_value,
            hedge_ratio=hedge_ratio,
            spread=spread,
            coint_stat=coint_stat,
            crit_values=crit_values,
        )
        return {
            "p_value": p_value,
            "hedge_ratio": hedge_ratio,
            "coint_stat": coint_stat,
            "crit_values": crit_values,
            "spread": spread,
        }


def precalculate_pair_analytics(
    price_df: pd.DataFrame,
    pair_combinations: list[Tuple[str, str]],
    days_back: int,
    cache: PairAnalyticsCache,
) -> None:
    """Pre-compute analytics for every pair/day combination."""

    for current_date in price_df.index[days_back:]:
        filtered_prices = filter_df(price_df, current_date, days_back)
        for pair in pair_combinations:
            cache.ensure(pair, current_date, filtered_prices)
