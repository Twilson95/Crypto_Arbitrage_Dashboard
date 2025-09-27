from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

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
        "spread_value",
    ]

    _LEGACY_SPREAD_COLUMNS = ["pair_key", "date_key", "timestamp", "value"]

    def __init__(self, cache_dir: Optional[Path | str] = None):
        if cache_dir is None:
            cache_dir = Path(".cache") / "pair_analytics"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.summary_path = self.cache_dir / "pair_analytics.csv"
        self._summary_df = self._load_summary_table()
        self._pending_summary_rows: list[pd.DataFrame] = []
        self._summary_needs_rewrite = bool(
            getattr(self, "_summary_missing_columns_on_load", [])
        )
        legacy_spread_df = self._load_legacy_spread_table()
        if legacy_spread_df is not None:
            self._merge_legacy_spread_table(legacy_spread_df)

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
            df = pd.read_csv(self.summary_path)
        else:
            df = pd.DataFrame(columns=self._SUMMARY_COLUMNS)
        if df.empty:
            missing_columns = [
                column for column in self._SUMMARY_COLUMNS if column not in df.columns
            ]
            self._summary_missing_columns_on_load = missing_columns
            index = pd.MultiIndex(
                levels=[[], []], codes=[[], []], names=["pair_key", "date_key"]
            )
            return pd.DataFrame(columns=self._SUMMARY_COLUMNS[2:], index=index)
        # migrate legacy spread_json column into a simple scalar value
        if "spread_json" in df.columns and "spread_value" not in df.columns:
            df["spread_value"] = df["spread_json"].apply(
                PairAnalyticsCache._extract_spread_value_from_json
            )

        df = df[[col for col in self._SUMMARY_COLUMNS if col in df.columns]].copy()
        missing_columns = [
            column for column in self._SUMMARY_COLUMNS if column not in df.columns
        ]
        for column in missing_columns:
            df[column] = pd.NA
        df = df[self._SUMMARY_COLUMNS]
        self._summary_missing_columns_on_load = missing_columns
        df = df.set_index(["pair_key", "date_key"])
        pair_values = df.index.get_level_values("pair_key")
        date_values = df.index.get_level_values("date_key")
        formatted_dates = [
            PairAnalyticsCache._format_timestamp_for_storage(
                PairAnalyticsCache._timestamp_from_date_key(value)
            )
            for value in date_values
        ]
        df.index = pd.MultiIndex.from_arrays(
            [pair_values, formatted_dates], names=["pair_key", "date_key"]
        )
        return df

    def _load_legacy_spread_table(self) -> Optional[pd.DataFrame]:
        legacy_path = self.cache_dir / "pair_spreads.csv"
        if not legacy_path.exists():
            return None

        df = pd.read_csv(legacy_path)
        if df.empty:
            return pd.DataFrame(columns=self._LEGACY_SPREAD_COLUMNS)

        df = df[
            [col for col in self._LEGACY_SPREAD_COLUMNS if col in df.columns]
        ].copy()
        for column in self._LEGACY_SPREAD_COLUMNS:
            if column not in df.columns:
                df[column] = pd.NA

        df = df[self._LEGACY_SPREAD_COLUMNS]
        df["date_key"] = df["date_key"].apply(
            lambda value: PairAnalyticsCache._format_timestamp_for_storage(
                PairAnalyticsCache._timestamp_from_date_key(value)
            )
        )
        df["timestamp"] = df["timestamp"].apply(
            lambda value: PairAnalyticsCache._format_timestamp_for_storage(
                PairAnalyticsCache._timestamp_from_date_key(value)
            )
        )
        return df

    def _merge_legacy_spread_table(self, spread_df: pd.DataFrame) -> None:
        if spread_df.empty:
            return

        updates_performed = False
        grouped = spread_df.groupby(["pair_key", "date_key"], dropna=False)
        for (pair_key, date_key), group in grouped:
            if pd.isna(pair_key) or pd.isna(date_key):
                continue

            group = group.sort_values("timestamp")
            spread_value_series = group["value"].dropna()
            spread_value = (
                float(spread_value_series.iloc[-1])
                if not spread_value_series.empty
                else None
            )

            summary_index = (pair_key, date_key)

            if summary_index not in self._summary_df.index:
                summary_data = {column: pd.NA for column in self._SUMMARY_COLUMNS[2:]}
                if spread_value is not None:
                    summary_data["spread_value"] = spread_value
                new_row = pd.DataFrame(
                    [summary_data],
                    index=pd.MultiIndex.from_tuples(
                        [summary_index], names=["pair_key", "date_key"]
                    ),
                )
                self._summary_df = pd.concat([self._summary_df, new_row])
                updates_performed = True
                continue

            existing_row = self._summary_df.loc[summary_index]
            if spread_value is not None and (
                pd.isna(existing_row.get("spread_value"))
                or existing_row.get("spread_value") != spread_value
            ):
                self._summary_df.loc[summary_index, "spread_value"] = spread_value
                updates_performed = True

        if updates_performed:
            self._summary_df = self._summary_df.sort_index()
            self._summary_needs_rewrite = True

    def _persist(
        self,
        *,
        new_summary_rows: Optional[pd.DataFrame] = None,
        rewrite_summary: bool = False,
    ) -> None:
        if rewrite_summary:
            self._rewrite_summary_csv()
        elif new_summary_rows is not None:
            self._append_summary_rows(new_summary_rows)

    def flush(self) -> None:
        """Persist any buffered updates to disk."""

        if self._pending_summary_rows:
            new_summary_rows = pd.concat(
                self._pending_summary_rows, ignore_index=True
            )
        else:
            new_summary_rows = None

        self._persist(
            new_summary_rows=new_summary_rows,
            rewrite_summary=self._summary_needs_rewrite,
        )

        self._pending_summary_rows.clear()
        self._summary_needs_rewrite = False

    @staticmethod
    def _timestamp_from_date_key(date_key) -> pd.Timestamp:
        timestamp = pd.to_datetime(date_key, errors="coerce")
        if pd.isna(timestamp):
            timestamp = pd.to_datetime(date_key, format="%Y%m%d%H%M%S", errors="coerce")
        if pd.isna(timestamp):
            return pd.Timestamp(date_key)
        return timestamp

    @staticmethod
    def _extract_spread_value_from_json(value: Any) -> Optional[float]:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return None
        if isinstance(value, str):
            try:
                records = json.loads(value)
            except Exception:  # pragma: no cover - defensive for corrupted rows
                return None
        else:
            records = value

        if not isinstance(records, list):
            return None

        last_value = None
        for record in records:
            if not isinstance(record, dict):
                continue
            value_obj = record.get("value")
            if value_obj is None or pd.isna(value_obj):
                continue
            try:
                last_value = float(value_obj)
            except (TypeError, ValueError):
                continue

        return last_value

    def load(self, pair: Tuple[str, str], current_date) -> Optional[Dict[str, Any]]:
        pair_key = self._pair_key(pair)
        date_key = self._date_key(current_date)
        if (pair_key, date_key) not in self._summary_df.index:
            return None

        summary_row = self._summary_df.loc[(pair_key, date_key)]
        spread_value = summary_row.get("spread_value")
        if pd.isna(spread_value):
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
            "spread_value": spread_value,
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
            "spread_value": None,
        }

        if crit_values is not None:
            crit_values = tuple(float(value) for value in crit_values)
            for idx, value in enumerate(crit_values[:3]):
                summary_data[f"crit_value_{idx + 1}"] = value

        spread = spread.sort_index()
        valid_spread = spread.dropna()
        if not valid_spread.empty:
            summary_data["spread_value"] = float(valid_spread.iloc[-1])

        new_summary_row = pd.DataFrame([summary_data]).set_index(
            ["pair_key", "date_key"]
        )
        summary_exists = (pair_key, date_key) in self._summary_df.index
        if summary_exists:
            self._summary_df = self._summary_df.drop(index=(pair_key, date_key))
        self._summary_df = pd.concat([self._summary_df, new_summary_row])

        new_summary_rows = new_summary_row.reset_index()
        self._pending_summary_rows.append(new_summary_rows)
        if summary_exists:
            self._summary_needs_rewrite = True

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
        *,
        auto_flush: bool = True,
    ) -> Optional[Dict[str, Any]]:
        if not self._has_complete_window(price_df_filtered, current_date):
            return None

        cached = self.load(pair, current_date)
        if cached is not None:
            hedge_ratio = cached.get("hedge_ratio")
            if hedge_ratio is None or pd.isna(hedge_ratio):
                cached = None
            else:
                spread, _ = CointegrationCalculator.calculate_spread(
                    price_df_filtered, pair, hedge_ratio
                )
                cached["spread"] = spread
                if auto_flush:
                    self.flush()
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
        if auto_flush:
            self.flush()
        return {
            "p_value": p_value,
            "hedge_ratio": hedge_ratio,
            "coint_stat": coint_stat,
            "crit_values": crit_values,
            "spread": spread,
            "spread_value": float(spread.dropna().iloc[-1]) if not spread.dropna().empty else None,
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
        if len(index_slice) == 0:
            return

        resume_checkpoint = cache._resume_checkpoint(len(pair_combinations))
        if resume_checkpoint is not None:
            resume_timestamp, inclusive = resume_checkpoint
            normalised_index = pd.Index(
                [PairAnalyticsCache._normalise_timestamp(value) for value in index_slice]
            )
            if inclusive:
                mask = normalised_index >= resume_timestamp
            else:
                mask = normalised_index > resume_timestamp
            index_slice = index_slice[mask]
            if len(index_slice) == 0:
                return

        total_days = len(index_slice)
        progress_interval = max(1, math.ceil(total_days / 20))

        try:
            for idx, current_date in enumerate(index_slice, start=1):
                start_time = time.time()
                filtered_prices = filter_df(price_df, current_date, days_back)
                if not cache._has_complete_window(
                    filtered_prices, current_date, days_back
                ):
                    continue
                for pair in pair_combinations:
                    cache.ensure(pair, current_date, filtered_prices, auto_flush=False)

                cache.flush()

                if idx == 1 or idx % progress_interval == 0 or idx == total_days:
                    end_time = time.time()
                    time_taken = end_time - start_time
                    print(
                        "[PairAnalyticsCache] "
                        f"Processed {idx}/{total_days} dates (latest: {current_date}) time: {time_taken:.2f}"
                    )
        finally:
            cache.flush()

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
