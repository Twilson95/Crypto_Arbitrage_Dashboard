from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import math
import pandas as pd

from cryptopy.src.arbitrage.CointegrationCalculator import CointegrationCalculator
from cryptopy.scripts.simulations.simulation_helpers import filter_df


class PairAnalyticsCache:
    """Cache cointegration and spread analytics for pair/day combinations."""

    _SUMMARY_COLUMNS = [
        "pair_key",
        "date_key",
        "p_value",
        "hedge_ratio",
        "coint_stat",
        "crit_value_1",
        "crit_value_2",
        "crit_value_3",
        "spread_timestamp",
        "spread_value",
    ]

    _SPREAD_COLUMNS = ["pair_key", "date_key", "timestamp", "value"]

    def __init__(self, cache_dir: Optional[Path | str] = None):
        if cache_dir is None:
            cache_dir = Path(".cache") / "pair_analytics"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.summary_path = self.cache_dir / "pair_analytics.csv"
        self._summary_df = self._load_summary_table()
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
            index = pd.MultiIndex(
                levels=[[], []], codes=[[], []], names=["pair_key", "date_key"]
            )
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

    def _persist(self) -> None:
        self._persist_summary()

    @staticmethod
    def _timestamp_from_date_key(date_key: str) -> pd.Timestamp:
        try:
            return pd.to_datetime(date_key, format="%Y%m%d%H%M%S")
        except (ValueError, TypeError):
            return pd.Timestamp(date_key)

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
        spread_value = summary_row.get("spread_value")
        if pd.isna(spread_value):
            return None

        timestamp_value = summary_row.get("spread_timestamp")
        if pd.isna(timestamp_value):
            timestamp = self._timestamp_from_date_key(date_key)
        else:
            timestamp = pd.Timestamp(timestamp_value)

        spread_series = self._get_spread_series(pair_key, date_key)
        if spread_series is None:
            return None

        crit_values = summary_row[
            ["crit_value_1", "crit_value_2", "crit_value_3"]
        ].tolist()
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
            "spread_timestamp": None,
            "spread_value": None,
        }

        if crit_values is not None:
            crit_values = tuple(float(value) for value in crit_values)
            for idx, value in enumerate(crit_values[:3]):
                summary_data[f"crit_value_{idx + 1}"] = value

        spread = spread.sort_index()
        valid_spread = spread.dropna()
        if not valid_spread.empty:
            timestamp = self._normalise_timestamp(valid_spread.index[-1])
            summary_data["spread_timestamp"] = timestamp.isoformat()
            summary_data["spread_value"] = float(valid_spread.iloc[-1])

        new_summary_row = pd.DataFrame([summary_data]).set_index(
            ["pair_key", "date_key"]
        )
        if (pair_key, date_key) in self._summary_df.index:
            self._summary_df = self._summary_df.drop(index=(pair_key, date_key))
        self._summary_df = pd.concat([self._summary_df, new_summary_row])

        spread = spread.sort_index()
        spread_df = pd.DataFrame(
            {
                "pair_key": pair_key,
                "date_key": date_key,
                "timestamp": [
                    self._normalise_timestamp(idx).isoformat() for idx in spread.index
                ],
                "value": [
                    float(value) if pd.notna(value) else float("nan")
                    for value in spread.values
                ],
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
        if not self._has_complete_window(price_df_filtered, current_date):
            return None

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

    @staticmethod
    def precalculate_pair_analytics(
        price_df: pd.DataFrame,
        pair_combinations: list[Tuple[str, str]],
        days_back: int,
        cache: PairAnalyticsCache,
    ) -> None:
        """Pre-compute analytics for every pair/day combination."""

        index_slice = price_df.index[days_back:]
        total_days = len(index_slice)
        if total_days == 0:
            return

        # progress_interval = max(1, math.ceil(total_days / 20))
        progress_interval = 1

        for idx, current_date in enumerate(index_slice, start=1):
            start_time = time.time()
            filtered_prices = filter_df(price_df, current_date, days_back)
            if not cache._has_complete_window(filtered_prices, current_date, days_back):
                continue
            for pair in pair_combinations:
                cache.ensure(pair, current_date, filtered_prices)

            if idx == 1 or idx % progress_interval == 0 or idx == total_days:
                end_time = time.time()
                time_taken = end_time - start_time
                print(
                    "[PairAnalyticsCache] "
                    f"Processed {idx}/{total_days} dates (latest: {current_date}) time: {time_taken:.2f}"
                )

    @staticmethod
    def _has_complete_window(
        price_df_filtered: pd.DataFrame,
        current_date,
        days_back: Optional[int] = None,
    ) -> bool:
        if days_back is None:
            days_back = 0

        if days_back <= 0:
            return not price_df_filtered.empty

        if price_df_filtered.empty:
            return False

        current_timestamp = PairAnalyticsCache._normalise_timestamp(current_date)
        earliest_timestamp = PairAnalyticsCache._normalise_timestamp(
            price_df_filtered.index.min()
        )

        expected_start = current_timestamp - pd.Timedelta(days=days_back)
        return earliest_timestamp <= expected_start
