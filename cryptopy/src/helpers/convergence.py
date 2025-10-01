import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class ConvergenceForecast:
    """Container for convergence forecast outputs."""

    expected_exit_spread: pd.Series
    expected_exit_mean: pd.Series
    spread_paths: pd.DataFrame
    mean_paths: pd.DataFrame
    half_life: Optional[float]
    confidence: float
    decay_factor: Optional[float]
    phi: Optional[float]
    intercept: Optional[float]

    def to_dict(self) -> dict:
        return {
            "expected_exit_spread": self.expected_exit_spread,
            "expected_exit_mean": self.expected_exit_mean,
            "spread_paths": self.spread_paths,
            "mean_paths": self.mean_paths,
            "half_life": self.half_life,
            "confidence": self.confidence,
            "decay_factor": self.decay_factor,
            "phi": self.phi,
            "intercept": self.intercept,
        }


class ConvergenceForecaster:
    """Reusable AR(1)-style convergence forecaster."""

    def __init__(
        self, rolling_window: int, holding_period: int, lookback: Optional[int] = None
    ):
        self.rolling_window = max(int(rolling_window), 1)
        self.holding_period = max(int(holding_period), 0)
        self.lookback = lookback if lookback is None else max(int(lookback), 1)

    def forecast(self, spread: pd.Series) -> ConvergenceForecast:
        spread = spread.copy()
        spread_mean = spread.rolling(window=self.rolling_window).mean()
        params = self._estimate_parameters(spread)

        if self.holding_period <= 0:
            empty_df = pd.DataFrame(index=spread.index)
            expected_exit_spread = spread.copy()
            expected_exit_mean = spread_mean.copy().fillna(expected_exit_spread)
            return ConvergenceForecast(
                expected_exit_spread=expected_exit_spread,
                expected_exit_mean=expected_exit_mean,
                spread_paths=empty_df,
                mean_paths=empty_df,
                half_life=params.get("half_life") if params else None,
                confidence=params.get("confidence") if params else 0.0,
                decay_factor=1.0,
                phi=params.get("phi") if params else None,
                intercept=params.get("intercept") if params else None,
            )

        if not params:
            empty_df = pd.DataFrame(index=spread.index)
            fallback = spread_mean.fillna(spread)
            return ConvergenceForecast(
                expected_exit_spread=fallback,
                expected_exit_mean=fallback,
                spread_paths=empty_df,
                mean_paths=empty_df,
                half_life=None,
                confidence=0.0,
                decay_factor=None,
                phi=None,
                intercept=None,
            )

        phi = params["phi"]
        intercept = params["intercept"]
        half_life = params.get("half_life")
        confidence = params.get("confidence", 0.0)
        decay_factor = (
            float(np.power(phi, self.holding_period)) if np.isfinite(phi) else None
        )

        spread_paths = self._build_spread_paths(spread, spread_mean, phi)
        mean_paths = self._build_mean_paths(spread, spread_paths)

        if spread_paths.empty:
            expected_exit_spread = spread_mean.fillna(spread)
        else:
            expected_exit_spread = spread_paths.iloc[:, -1].copy()
            expected_exit_spread = expected_exit_spread.where(
                ~expected_exit_spread.isna(), spread_mean
            )
            expected_exit_spread = expected_exit_spread.fillna(spread)

        if mean_paths.empty:
            expected_exit_mean = spread_mean.fillna(expected_exit_spread)
        else:
            expected_exit_mean = mean_paths.iloc[:, -1].copy()
            expected_exit_mean = expected_exit_mean.where(
                ~expected_exit_mean.isna(), expected_exit_spread
            )
            expected_exit_mean = expected_exit_mean.fillna(spread_mean)

        expected_exit_mean = expected_exit_mean.fillna(expected_exit_spread)

        return ConvergenceForecast(
            expected_exit_spread=expected_exit_spread,
            expected_exit_mean=expected_exit_mean,
            spread_paths=spread_paths,
            mean_paths=mean_paths,
            half_life=half_life,
            confidence=confidence,
            decay_factor=decay_factor,
            phi=phi,
            intercept=intercept,
        )

    def plot_forecast(
        self,
        spread: pd.Series,
        forecast: ConvergenceForecast,
        *,
        show: bool = True,
        save_path: Optional[str] = None,
        ax=None,
        threshold_multiplier: Optional[float] = None,
        limit_multiplier: Optional[float] = None,
    ):
        """Visualise the historical spread, rolling mean, and forecast trajectories.

        Parameters
        ----------
        spread:
            Historical spread series used to generate the forecast.
        forecast:
            Convergence forecast generated via :meth:`forecast`.
        show:
            Whether to display the plot via ``plt.show()``. Defaults to ``True``.
        save_path:
            Optional path to persist the plot image. If provided the figure will be
            written using ``plt.savefig``.
        ax:
            Optional matplotlib axes to draw on. When omitted a new figure/axes pair
            is created and returned.

        Returns
        -------
        matplotlib.axes.Axes
            The axes containing the rendered plot.
        """

        import matplotlib.pyplot as plt

        spread = spread.sort_index()
        if isinstance(spread.index, pd.PeriodIndex):
            spread = spread.copy()
            spread.index = spread.index.to_timestamp()
        spread_mean = spread.rolling(window=self.rolling_window).mean()
        spread_std = spread.rolling(window=self.rolling_window).std()

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        else:
            fig = ax.figure

        spread.plot(ax=ax, label="Spread", color="black", linewidth=1.2)
        spread_mean.plot(
            ax=ax,
            label="Rolling mean",
            color="tab:blue",
            linestyle="--",
            linewidth=1.0,
        )

        if threshold_multiplier is not None and np.isfinite(threshold_multiplier):
            upper_threshold = spread_mean + threshold_multiplier * spread_std
            lower_threshold = spread_mean - threshold_multiplier * spread_std
            upper_threshold.plot(
                ax=ax,
                label="Upper threshold",
                color="tab:orange",
                linestyle=":",
            )
            lower_threshold.plot(
                ax=ax,
                label="Lower threshold",
                color="tab:orange",
                linestyle=":",
            )

        if limit_multiplier is not None and np.isfinite(limit_multiplier):
            upper_limit = spread_mean + limit_multiplier * spread_std
            lower_limit = spread_mean - limit_multiplier * spread_std
            upper_limit.plot(
                ax=ax,
                label="Upper limit",
                color="tab:red",
                linestyle="-.",
            )
            lower_limit.plot(
                ax=ax,
                label="Lower limit",
                color="tab:red",
                linestyle="-.",
            )
        spread_mean.plot(
            ax=ax, label="Rolling mean", color="tab:blue", linestyle="--", linewidth=1.0
        )

        def _build_future_index(current_index: pd.Index, steps: int) -> pd.Index:
            if steps <= 0:
                return current_index[:0]

            last_value = current_index[-1]

            datetime_index: Optional[pd.DatetimeIndex] = None

            if isinstance(current_index, pd.DatetimeIndex):
                datetime_index = current_index
            elif (
                np.issubdtype(current_index.dtype, np.datetime64)
                or current_index.inferred_type in {"datetime64", "datetime64tz", "date"}
            ):
                datetime_index = pd.DatetimeIndex(current_index)

            if datetime_index is not None:
                freq = datetime_index.freq or datetime_index.inferred_freq

                if freq is not None:
                    start = last_value + pd.tseries.frequencies.to_offset(freq)
                    return pd.date_range(start=start, periods=steps, freq=freq)

                if len(datetime_index) >= 2:
                    delta = datetime_index[-1] - datetime_index[-2]
                    if isinstance(delta, pd.Timedelta) and delta != pd.Timedelta(0):
                        values = [last_value + delta * (i + 1) for i in range(steps)]
                        return pd.DatetimeIndex(values)

                values = [last_value + pd.Timedelta(days=i + 1) for i in range(steps)]
                return pd.DatetimeIndex(values)

            if np.issubdtype(current_index.dtype, np.number):
                if len(current_index) >= 2:
                    step = current_index[-1] - current_index[-2]
                    if step == 0:
                        step = 1
                else:
                    step = 1

                values = [last_value + step * (i + 1) for i in range(steps)]
                return pd.Index(values)

            values = np.arange(len(current_index), len(current_index) + steps)
            return pd.Index(values)

        def _plot_continuation(
            base_series: pd.Series,
            continuation_values: pd.Series,
            label: str,
            color: str,
            linestyle: str,
            alpha: float,
            linewidth: float,
        ):
            base_series_valid = base_series.dropna()
            if base_series_valid.empty:
                return

            base_value = base_series_valid.iloc[-1]
            base_index = base_series_valid.index[-1]

            extension = continuation_values.dropna()
            if extension.empty:
                return

            steps = len(extension)
            future_index = _build_future_index(base_series_valid.index, steps)
            if len(future_index) < steps:
                return

            future_index = future_index[:steps]
            combined_index = pd.Index([base_index]).append(future_index)
            combined_values = pd.Series(
                np.concatenate([[base_value], extension.to_numpy()]),
                index=combined_index,
            )
            combined_values.plot(
                ax=ax,
                label=label,
                color=color,
                linestyle=linestyle,
                linewidth=linewidth,
                alpha=alpha,
            )

        if not forecast.spread_paths.empty:
            spread_forecast = forecast.spread_paths.iloc[-1]
            _plot_continuation(
                spread,
                spread_forecast,
                label="Forecast spread path",
                color="tab:orange",
                linestyle=":",
                alpha=0.7,
                linewidth=1.1,
            )

        if not forecast.mean_paths.empty:
            mean_forecast = forecast.mean_paths.iloc[-1]
            _plot_continuation(
                spread_mean,
                mean_forecast,
                label="Forecast mean path",
                color="tab:green",
                linestyle="-.",
                alpha=0.7,
                linewidth=1.1,
            )

        if (
            forecast.expected_exit_spread is not None
            and not forecast.spread_paths.empty
        ):
            exit_spread = forecast.expected_exit_spread.iloc[-1]
            if pd.notna(exit_spread):
                spread_forecast = forecast.spread_paths.iloc[-1].dropna()
                if not spread_forecast.empty:
                    future_index = _build_future_index(
                        spread.index, len(spread_forecast)
                    )
                    if len(future_index) >= len(spread_forecast):
                        ax.scatter(
                            future_index[len(spread_forecast) - 1],
                            exit_spread,
                            color="tab:orange",
                            label="Expected exit spread",
                        )

        if forecast.expected_exit_mean is not None and not forecast.mean_paths.empty:
            exit_mean = forecast.expected_exit_mean.iloc[-1]
            if pd.notna(exit_mean):
                mean_forecast = forecast.mean_paths.iloc[-1].dropna()
                if not mean_forecast.empty:
                    future_index = _build_future_index(spread.index, len(mean_forecast))
                    if len(future_index) >= len(mean_forecast):
                        ax.scatter(
                            future_index[len(mean_forecast) - 1],
                            exit_mean,
                            color="tab:green",
                            label="Expected exit mean",
                        )

        ax.set_title("Convergence Forecast")
        ax.set_xlabel("Date")
        ax.set_ylabel("Spread")
        ax.legend(loc="best")
        ax.grid(True, linestyle=":", alpha=0.3)

        if save_path:
            fig.savefig(save_path, bbox_inches="tight")

        if show:
            plt.show()

        return ax

    def _estimate_parameters(self, spread: pd.Series):
        spread_clean = spread.dropna()

        if self.lookback is not None and len(spread_clean) > self.lookback:
            spread_clean = spread_clean.iloc[-self.lookback :]

        if len(spread_clean) < max(5, self.rolling_window):
            return None

        lagged = spread_clean.shift(1).dropna()
        current = spread_clean.loc[lagged.index]

        if len(current) < 5:
            return None

        X = np.column_stack([np.ones(len(lagged)), lagged.values])
        y = current.values

        try:
            coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        except np.linalg.LinAlgError:
            return None

        intercept, phi = coeffs

        if not np.isfinite(intercept) or not np.isfinite(phi):
            return None

        y_hat = X @ coeffs
        residuals = y - y_hat
        ss_res = float(np.sum(residuals**2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))

        if math.isclose(ss_tot, 0.0):
            r_squared = 0.0
        else:
            r_squared = max(0.0, min(1.0, 1 - ss_res / ss_tot))

        if abs(phi) < 1 and abs(phi) > 1e-8:
            half_life = float(-np.log(2) / np.log(abs(phi)))
        elif abs(phi) < 1e-8:
            half_life = 0.0
        else:
            half_life = None

        return {
            "intercept": float(intercept),
            "phi": float(phi),
            "half_life": half_life,
            "confidence": r_squared,
        }

    def _build_spread_paths(
        self, spread: pd.Series, spread_mean: pd.Series, phi: float
    ) -> pd.DataFrame:
        if self.holding_period <= 0:
            return pd.DataFrame(index=spread.index)

        horizons = range(1, self.holding_period + 1)
        columns = {
            f"horizon_{step}": spread_mean + (spread - spread_mean) * (phi**step)
            for step in horizons
        }
        return pd.DataFrame(columns)

    def _build_mean_paths(
        self, spread: pd.Series, spread_paths: pd.DataFrame
    ) -> pd.DataFrame:
        if self.holding_period <= 0 or spread_paths.empty:
            return pd.DataFrame(index=spread.index)

        window = self.rolling_window
        horizons = spread_paths.shape[1]

        mean_paths = pd.DataFrame(
            np.nan,
            index=spread.index,
            columns=spread_paths.columns,
        )

        if len(spread) < window:
            return mean_paths

        spread_values = spread.to_numpy(dtype=float)
        forecast_values = spread_paths.to_numpy(dtype=float)

        sliding_windows = np.lib.stride_tricks.sliding_window_view(
            spread_values, window_shape=window
        )
        window_valid = ~np.isnan(sliding_windows).any(axis=1)
        window_sums = np.where(window_valid, np.sum(sliding_windows, axis=1), np.nan)

        min_window_horizon = min(window, horizons)
        if min_window_horizon:
            window_prefix = np.cumsum(sliding_windows[:, :min_window_horizon], axis=1)
        else:
            window_prefix = np.empty((sliding_windows.shape[0], 0))

        forecast_sub = forecast_values[window - 1 :, :]
        if forecast_sub.size == 0:
            return mean_paths

        forecast_valid = ~np.isnan(forecast_sub)
        cumulative_valid = np.cumprod(forecast_valid, axis=1).astype(bool)
        forecast_filled = np.where(forecast_valid, forecast_sub, 0.0)
        forecast_prefix = np.cumsum(forecast_filled, axis=1)

        window_sums_expanded = np.broadcast_to(window_sums[:, None], forecast_sub.shape)
        forecast_prefix_expanded = forecast_prefix

        drop_initial = np.zeros_like(forecast_prefix_expanded)
        if min_window_horizon:
            drop_initial[:, :min_window_horizon] = window_prefix
        if horizons > window:
            drop_initial[:, window:] = window_sums_expanded[:, window:]

        forecast_removals = np.zeros_like(forecast_prefix_expanded)
        if horizons > window:
            forecast_removals[:, window:] = forecast_prefix[:, :-window]

        updated_sums = (
            window_sums_expanded
            + forecast_prefix_expanded
            - drop_initial
            - forecast_removals
        )
        updated_means = updated_sums / window

        invalid_window_rows = ~window_valid
        if np.any(invalid_window_rows):
            updated_means[invalid_window_rows, :] = np.nan
        updated_means[~cumulative_valid] = np.nan

        mean_paths.iloc[window - 1 :, :] = updated_means

        return mean_paths
