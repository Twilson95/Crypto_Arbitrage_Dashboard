from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import warnings

import math
import time
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
    ]

    def __init__(
        self,
        cache_dir: Optional[Path | str] = None,
        *,
        read_only: bool = False,
    ):
        if cache_dir is None:
            cache_dir = Path(".cache") / "pair_analytics"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.summary_path = self.cache_dir / "pair_analytics.csv"
        self._summary_df = self._load_summary_table()
        self._read_only = read_only

    def _append_summary_rows(self, rows: pd.DataFrame) -> None:
        if rows is None or rows.empty:
            return
        rows = rows[self._SUMMARY_COLUMNS]
        header = not self.summary_path.exists() or self.summary_path.stat().st_size == 0
        rows.to_csv(self.summary_path, mode="a", header=header, index=False)

    def _rewrite_summary_csv(self) -> None:
        summary_to_write = self._summary_df.reset_index()
        summary_to_write = summary_to_write[self._SUMMARY_COLUMNS]
        summary_to_write.to_csv(self.summary_path, index=False)

    @staticmethod
    def _pair_key(pair: Tuple[str, str]) -> str:
        return "__".join(symbol.replace("/", "_") for symbol in pair)

    @staticmethod
    def _date_key(current_date) -> str:
        return PairAnalyticsCache._format_timestamp_for_storage(current_date)

    @staticmethod
    def _normalise_timestamp(value) -> pd.Timestamp:
        timestamp = pd.Timestamp(value)
        if timestamp.tzinfo is not None:
            timestamp = timestamp.tz_convert(None)
        return timestamp

    @staticmethod
    def _format_timestamp_for_storage(value) -> str:
        timestamp = pd.Timestamp(value)
        if pd.isna(timestamp):
            return ""
        if timestamp.tzinfo is not None:
            return timestamp.isoformat(sep=" ")
        return timestamp.strftime("%Y-%m-%d %H:%M:%S")

    def _load_summary_table(self) -> pd.DataFrame:
        if self.summary_path.exists():
            print(f"Loading summary table")
            df = pd.read_csv(self.summary_path)
        else:
            df = pd.DataFrame(columns=self._SUMMARY_COLUMNS)
        if df.empty:
            index = pd.MultiIndex(
                levels=[[], []], codes=[[], []], names=["pair_key", "date_key"]
            )
            empty_df = pd.DataFrame(columns=self._SUMMARY_COLUMNS[2:], index=index)
            empty_df.sort_index(inplace=True)
            return empty_df
        df = df[[col for col in self._SUMMARY_COLUMNS if col in df.columns]].copy()
        for column in self._SUMMARY_COLUMNS:
            if column not in df.columns:
                df[column] = pd.NA
        df = df[self._SUMMARY_COLUMNS]

        date_series = df["date_key"].astype("string")
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=(
                    "In a future version of pandas, parsing datetimes with mixed time "
                    "zones will raise an error unless `utc=True`."
                ),
                category=FutureWarning,
            )
            parsed_dates = pd.to_datetime(
                date_series, format="ISO8601", errors="coerce"
            )

        legacy_mask = parsed_dates.isna() & date_series.notna()
        if legacy_mask.any():
            legacy_parsed = pd.to_datetime(
                date_series.loc[legacy_mask],
                format="%Y%m%d%H%M%S",
                errors="coerce",
            )
            parsed_dates.loc[legacy_mask] = legacy_parsed

        formatted_dates = pd.Series(
            (
                PairAnalyticsCache._format_timestamp_for_storage(timestamp)
                if not pd.isna(timestamp)
                else date_value
            )
            for timestamp, date_value in zip(parsed_dates, date_series)
        )
        formatted_dates = formatted_dates.astype("string")
        missing_mask = formatted_dates.isna() | (formatted_dates == "")
        if missing_mask.any():
            formatted_dates.loc[missing_mask] = date_series.loc[missing_mask]

        df["date_key"] = formatted_dates
        df = df.set_index(["pair_key", "date_key"])
        df.sort_index(inplace=True)
        return df

    def _persist(
        self,
        *,
        new_summary_rows: Optional[pd.DataFrame] = None,
        rewrite_summary: bool = False,
    ) -> None:
        if self._read_only:
            return

        if rewrite_summary:
            self._rewrite_summary_csv()
        elif new_summary_rows is not None:
            self._append_summary_rows(new_summary_rows)

    @staticmethod
    def _timestamp_from_date_key(date_key) -> pd.Timestamp:
        timestamp = pd.to_datetime(date_key, errors="coerce")
        if pd.isna(timestamp):
            timestamp = pd.to_datetime(date_key, format="%Y%m%d%H%M%S", errors="coerce")
        if pd.isna(timestamp):
            return pd.Timestamp(date_key)
        return timestamp

    def load(self, pair: Tuple[str, str], current_date) -> Optional[Dict[str, Any]]:
        pair_key = self._pair_key(pair)
        date_key = self._date_key(current_date)
        try:
            summary_row = self._summary_df.loc[(pair_key, date_key)]
        except KeyError:
            return None

        if isinstance(summary_row, pd.DataFrame):
            summary_row = summary_row.iloc[-1]

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
        if self._read_only:
            return

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

        new_summary_row = pd.DataFrame([summary_data]).set_index(
            ["pair_key", "date_key"]
        )
        try:
            self._summary_df.loc[(pair_key, date_key)]
        except KeyError:
            summary_exists = False
        else:
            summary_exists = True
            self._summary_df = self._summary_df.drop(index=(pair_key, date_key))

        self._summary_df = pd.concat([self._summary_df, new_summary_row])
        self._summary_df.sort_index(inplace=True)

        new_summary_rows = new_summary_row.reset_index()
        self._persist(
            new_summary_rows=None if summary_exists else new_summary_rows,
            rewrite_summary=summary_exists,
        )

    def get_spread_series(
        self, pair: Tuple[str, str], rolling_window: Optional[int]
    ) -> pd.Series:
        """Return cached spread values.

        Spread values are no longer persisted, so this method returns an empty
        series. It is kept for backwards compatibility with older simulator
        code paths that may still invoke it conditionally.
        """

        return pd.Series(dtype="float64")

    def _resume_checkpoint(
        self, required_pairs: int
    ) -> Optional[Tuple[pd.Timestamp, bool]]:
        if required_pairs <= 0 or self._summary_df.empty:
            return None

        summary_index = self._summary_df.reset_index()[["pair_key", "date_key"]]
        if summary_index.empty:
            return None

        summary_index["timestamp"] = summary_index["date_key"].apply(
            self._timestamp_from_date_key
        )
        summary_index = summary_index.sort_values("timestamp")
        pair_counts = summary_index.groupby("timestamp")["pair_key"].nunique()
        if pair_counts.empty:
            return None

        incomplete_dates = pair_counts[pair_counts < required_pairs]
        if not incomplete_dates.empty:
            resume_timestamp = PairAnalyticsCache._normalise_timestamp(
                incomplete_dates.index.min()
            )
            return resume_timestamp, True

        resume_timestamp = PairAnalyticsCache._normalise_timestamp(
            pair_counts.index.max()
        )
        return resume_timestamp, False

    def ensure(
        self,
        pair: Tuple[str, str],
        current_date,
        price_df_filtered: pd.DataFrame,
    ) -> Optional[Dict[str, Any]]:
        if not self._has_complete_window(price_df_filtered, current_date):
            return None

        cached = self.load(pair, current_date)
        if cached is not None or self._read_only:
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
        }

    @staticmethod
    def precalculate_pair_analytics(
        price_df: pd.DataFrame,
        pair_combinations: list[Tuple[str, str]],
        days_back: int,
        cache: PairAnalyticsCache,
    ) -> None:
        """Pre-compute analytics for every pair/day combination."""

        if getattr(cache, "_read_only", False):
            raise ValueError("Cannot precalculate analytics using a read-only cache.")

        index_slice = price_df.index[days_back:]
        if len(index_slice) == 0:
            return

        resume_checkpoint = cache._resume_checkpoint(len(pair_combinations))
        if resume_checkpoint is not None:
            resume_timestamp, inclusive = resume_checkpoint
            normalised_index = pd.Index(
                [
                    PairAnalyticsCache._normalise_timestamp(value)
                    for value in index_slice
                ]
            )
            if inclusive:
                mask = normalised_index >= resume_timestamp
            else:
                mask = normalised_index > resume_timestamp
            index_slice = index_slice[mask]
            if len(index_slice) == 0:
                return

        total_days = len(index_slice)
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
