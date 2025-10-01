from cryptopy import StatisticalArbitrage
from cryptopy.src.helpers.convergence import ConvergenceForecaster
import os
import pandas as pd
import json
import datetime
import glob
import numpy as np


def read_historic_data_long_term(pair, historic_data_folder):
    pair_filename = pair.replace("/", "_")  # Replace "/" with "_"
    file_path = f"{historic_data_folder}{pair_filename}.csv"
    if os.path.exists(file_path):
        return pd.read_csv(file_path, index_col="datetime", parse_dates=True)
    else:
        raise FileNotFoundError(f"File for {pair} not found.")


def filter_df(df, current_date, days_back):
    start_date = current_date - pd.Timedelta(days=days_back)
    return df[(df.index >= start_date) & (df.index <= current_date)]


def filter_list(list_data, date):
    """Safely extract data for a specific date from different container types."""

    if isinstance(list_data, (pd.Series, pd.DataFrame)):
        return list_data.loc[date] if date in list_data.index else None

    if isinstance(list_data, dict):
        return list_data.get(date)

    if np.isscalar(list_data):
        return list_data

    return None


def compute_spread_metrics(
    parameters, spread, current_date=None, trade_open=False
):
    rolling_window = parameters["rolling_window"]
    spread_mean = spread.rolling(window=rolling_window).mean()
    spread_std = spread.rolling(window=rolling_window).std()
    spread_threshold = parameters["spread_threshold"]

    upper_threshold = spread_mean + spread_threshold * spread_std
    lower_threshold = spread_mean - spread_threshold * spread_std

    upper_spread_threshold = parameters["spread_limit"]
    upper_spread_limit = spread_mean + upper_spread_threshold * spread_std
    lower_spread_limit = spread_mean - upper_spread_threshold * spread_std

    spread_std_safe = spread_std.replace(0, np.nan)
    spread_deviation = (spread - spread_mean).abs() / spread_std_safe
    trade_candidate_mask = (spread_deviation >= spread_threshold) & spread_deviation.notna()

    holding_period = max(int(round(parameters.get("expected_holding_days", 10))), 0)
    convergence_window = parameters.get("convergence_lookback", rolling_window * 3)
    forecaster = ConvergenceForecaster(
        rolling_window, holding_period, convergence_window
    )
    forecast = forecaster.forecast(spread)
    if parameters.get("plot_forecast"):
        forecaster.plot_forecast(spread, forecast)

    expected_exit_mean = forecast.expected_exit_mean
    expected_exit_spread = forecast.expected_exit_spread
    decay_factor = forecast.decay_factor
    convergence_half_life = forecast.half_life
    convergence_confidence = forecast.confidence
    convergence_phi = forecast.phi
    convergence_intercept = forecast.intercept
    spread_paths = forecast.spread_paths
    mean_paths = forecast.mean_paths

    if isinstance(expected_exit_mean, pd.Series):
        expected_exit_mean = expected_exit_mean.where(trade_candidate_mask)
    if isinstance(expected_exit_spread, pd.Series):
        expected_exit_spread = expected_exit_spread.where(trade_candidate_mask)

    if isinstance(spread_paths, pd.DataFrame) and not spread_paths.empty:
        mask_series = trade_candidate_mask.reindex(spread_paths.index).fillna(False)
        mask_values = mask_series.to_numpy(dtype=bool)[:, None]
        broadcast_mask = np.broadcast_to(mask_values, spread_paths.shape)
        spread_paths = spread_paths.where(broadcast_mask)

    if isinstance(mean_paths, pd.DataFrame) and not mean_paths.empty:
        mask_series = trade_candidate_mask.reindex(mean_paths.index).fillna(False)
        mask_values = mask_series.to_numpy(dtype=bool)[:, None]
        broadcast_mask = np.broadcast_to(mask_values, mean_paths.shape)
        mean_paths = mean_paths.where(broadcast_mask)

    return {
        "spread_mean": spread_mean,
        "spread_std": spread_std,
        "upper_threshold": upper_threshold,
        "lower_threshold": lower_threshold,
        "upper_limit": upper_spread_limit,
        "lower_limit": lower_spread_limit,
        "expected_mean_at_exit": expected_exit_mean,
        "expected_spread_at_exit": expected_exit_spread,
        "convergence_half_life": convergence_half_life,
        "convergence_confidence": convergence_confidence,
        "convergence_decay_factor": decay_factor,
        "convergence_phi": convergence_phi,
        "convergence_intercept": convergence_intercept,
        "forecasted_spread_path": spread_paths,
        "forecasted_mean_path": mean_paths,
        "spread_deviation": spread_deviation,
        "trade_candidate_mask": trade_candidate_mask,
    }


def get_todays_spread_data(
    parameters, spread, current_date, spread_metrics=None, trade_open=False
):
    if spread_metrics is None:
        spread_metrics = compute_spread_metrics(
            parameters, spread, current_date=current_date, trade_open=trade_open
        )

    todays_spread = filter_list(spread, current_date)
    todays_spread_mean = filter_list(spread_metrics["spread_mean"], current_date)
    todays_spread_std = filter_list(spread_metrics["spread_std"], current_date)
    todays_spread_deviation = filter_list(
        spread_metrics.get("spread_deviation"), current_date
    )

    def _sanitize(value):
        if value is None:
            return None
        if isinstance(value, (float, np.floating)) and np.isnan(value):
            return None
        return value

    todays_spread_std = _sanitize(todays_spread_std)
    todays_spread_mean = _sanitize(todays_spread_mean)
    todays_spread = _sanitize(todays_spread)
    todays_spread_deviation = _sanitize(todays_spread_deviation)

    if (
        todays_spread_deviation is None
        and todays_spread is not None
        and todays_spread_mean is not None
        and todays_spread_std not in (None, 0)
    ):
        todays_spread_deviation = abs(todays_spread - todays_spread_mean) / todays_spread_std

    expected_exit_mean = filter_list(
        spread_metrics["expected_mean_at_exit"], current_date
    )
    expected_exit_spread = filter_list(
        spread_metrics["expected_spread_at_exit"], current_date
    )
    expected_exit_mean = _sanitize(expected_exit_mean)
    expected_exit_spread = _sanitize(expected_exit_spread)

    forecast_spread_path = spread_metrics.get("forecasted_spread_path")
    forecast_mean_path = spread_metrics.get("forecasted_mean_path")
    todays_spread_forecast = None
    todays_mean_forecast = None
    if (
        isinstance(forecast_spread_path, pd.DataFrame)
        and current_date in forecast_spread_path.index
    ):
        todays_spread_forecast = (
            forecast_spread_path.loc[current_date].dropna().to_dict()
        )
    if (
        isinstance(forecast_mean_path, pd.DataFrame)
        and current_date in forecast_mean_path.index
    ):
        todays_mean_forecast = forecast_mean_path.loc[current_date].dropna().to_dict()
    forecast_diff = None
    if todays_spread_forecast and todays_mean_forecast:
        forecast_diff = {
            key: todays_spread_forecast.get(key) - todays_mean_forecast.get(key, np.nan)
            for key in todays_spread_forecast
        }

    spread_threshold = parameters.get("spread_threshold", 0)
    consider_for_trade = (
        todays_spread_deviation is not None
        and todays_spread_std not in (None, 0)
        and todays_spread_deviation >= spread_threshold
    )

    if not consider_for_trade:
        expected_exit_mean = None
        expected_exit_spread = None
        todays_spread_forecast = None
        todays_mean_forecast = None
        forecast_diff = None
    else:
        if expected_exit_mean is None:
            expected_exit_mean = todays_spread_mean
        if expected_exit_spread is None:
            expected_exit_spread = todays_spread_mean

    return {
        "date": current_date,
        "spread": todays_spread,
        "spread_mean": todays_spread_mean,
        "spread_std": todays_spread_std,
        "upper_threshold": filter_list(spread_metrics["upper_threshold"], current_date),
        "upper_limit": filter_list(spread_metrics["upper_limit"], current_date),
        "lower_threshold": filter_list(spread_metrics["lower_threshold"], current_date),
        "lower_limit": filter_list(spread_metrics["lower_limit"], current_date),
        "expected_spread_mean_at_exit": expected_exit_mean,
        "expected_exit_spread": expected_exit_spread,
        "convergence_half_life": spread_metrics.get("convergence_half_life"),
        "convergence_confidence": spread_metrics.get("convergence_confidence"),
        "convergence_decay_factor": spread_metrics.get("convergence_decay_factor"),
        "convergence_phi": spread_metrics.get("convergence_phi"),
        "convergence_intercept": spread_metrics.get("convergence_intercept"),
        "forecasted_spread_path": todays_spread_forecast,
        "forecasted_mean_path": todays_mean_forecast,
        "forecast_spread_minus_mean": forecast_diff,
        "spread_deviation": todays_spread_deviation,
        "trade_considered": consider_for_trade,
    }


def get_trade_profit(
    open_event,
    close_event,
    pair,
    currency_fees,
    df_filtered,
    trade_amount,
):
    arbitrage = StatisticalArbitrage.statistical_arbitrage_iteration(
        entry=(
            open_event["date"],
            open_event["spread_data"]["spread"],
            open_event["direction"],
            open_event.get("expected_exit_spread"),
        ),
        exit=(close_event["date"], close_event["spread_data"]["spread"]),
        pairs=pair,
        currency_fees=currency_fees,  # Example transaction cost
        price_df=df_filtered,
        usd_start=trade_amount,
        hedge_ratio=open_event["hedge_ratio"],
        exchange="test",
        borrow_rate_per_day=open_event.get("borrow_rate_per_day", 0.0),
        expected_holding_days=open_event.get("expected_holding_days", 0.0),
    )
    if arbitrage:
        profit = arbitrage.get("summary_header", {}).get("total_profit", 0)
        print(
            f"Pair: {pair}\n"
            f"Live Dates: {open_event['date']} to {close_event['date']}\n"
            f"Close Reason: {close_event['reason']}\n"
            f"Profit {profit:.2f}"
        )
        return profit


def get_combined_df_of_data(folder_path, field="close"):
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    dfs = []
    for file in csv_files:
        file_name = os.path.basename(file).replace(".csv", "")
        new_column_name = file_name.replace("_", "/")
        df = pd.read_csv(file, index_col=0)
        df = df[[field]].rename(columns={field: new_column_name})
        dfs.append(df)
    combined_df = pd.concat(dfs, axis=1, join="outer")
    combined_df.index = pd.to_datetime(combined_df.index)
    combined_df.index = combined_df.index.date
    return combined_df


def get_avg_price_difference(df, pair, hedge_ratio):
    mean_prices1 = df[pair[0]].mean()
    mean_prices2 = df[pair[1]].mean()

    return mean_prices1 / (mean_prices2 * hedge_ratio)


def calculate_expected_profit(
    pair,
    open_event,
    position_size,
    currency_fees,
    borrow_rate_per_day=None,
    expected_holding_days=0.0,
    short_notional=None,
):
    spread_data = open_event["spread_data"]
    spread = spread_data["spread"]
    expected_exit = spread_data.get("expected_exit_spread")
    if expected_exit is None or pd.isna(expected_exit):
        expected_exit = spread_data.get("expected_spread_mean_at_exit")
    if expected_exit is None or pd.isna(expected_exit):
        expected_exit = spread_data.get("spread_mean")
    if spread is None or expected_exit is None or pd.isna(spread):
        return 0.0

    fees = currency_fees[pair[0]]["taker"] * 2 + currency_fees[pair[1]]["taker"] * 2
    bought_amount = position_size["long_position"]["amount"]
    if open_event["direction"] == "short":
        bought_amount /= open_event["hedge_ratio"]

    expected_profit = bought_amount * abs(spread - expected_exit) * (1 - fees)

    if short_notional is None:
        short_notional = open_event.get("short_notional")

    if borrow_rate_per_day is None:
        borrow_rate_per_day = open_event.get("borrow_rate_per_day")

    borrow_rate = borrow_rate_per_day or 0.0
    holding_days = (
        expected_holding_days or open_event.get("expected_holding_days") or 0.0
    )

    if short_notional is None:
        short_entry_price = open_event.get("short_entry_price")
        if short_entry_price is not None:
            short_notional = (
                position_size["short_position"]["amount"] * short_entry_price
            )

    if short_notional is None:
        short_notional = 0.0

    expected_profit -= borrow_rate * holding_days * short_notional

    return expected_profit


def get_bought_and_sold_amounts(df, pair, open_event, current_date, trade_size=100):
    hedge_ratio = open_event["hedge_ratio"]

    if open_event["direction"] == "short":
        bought_coin = pair[1]
        sold_coin = pair[0]
        buy_coin_price = filter_list(df[bought_coin], current_date)
        adjusted_value = buy_coin_price
        bought_amount = trade_size / adjusted_value
        sold_amount = bought_amount / hedge_ratio
    elif open_event["direction"] == "long":
        bought_coin = pair[0]
        sold_coin = pair[1]
        buy_coin_price = filter_list(df[bought_coin], current_date)
        bought_amount = trade_size / buy_coin_price
        sold_amount = bought_amount * hedge_ratio
    else:
        return None

    trade_size = {
        "trade_amount_usd": trade_size,
        "long_position": {"coin": bought_coin, "amount": bought_amount},
        "short_position": {"coin": sold_coin, "amount": sold_amount},
    }

    return trade_size


def calculate_volume_spike(data, volume_period=30, volume_threshold=2):
    avg_volume = data.rolling(window=volume_period).mean().iloc[-1]
    current_volume = data.iloc[-1]
    return current_volume > avg_volume * volume_threshold, current_volume / avg_volume


def calculate_volatility_spike(data, volatility_period=30, volatility_threshold=1.5):
    returns = data.pct_change()
    avg_volatility = returns.rolling(window=volatility_period).std().iloc[-1]
    current_volatility = returns.iloc[-1]
    if avg_volatility < 0:
        return (
            current_volatility < avg_volatility * volatility_threshold,
            current_volatility / avg_volatility,
        )
    else:
        return (
            current_volatility > avg_volatility * volatility_threshold,
            current_volatility / avg_volatility,
        )


def is_volume_or_volatility_spike(price_data, volume_data, pair, parameters):
    is_spike = False
    volume_ratio, volatility_ratio = 0, 0
    for coin in pair:
        volumes = volume_data[coin]
        prices = price_data[coin]
        volume_spike, volume_ratio = calculate_volume_spike(
            volumes, parameters["volume_period"], parameters["volume_threshold"]
        )
        volatility_spike, volatility_ratio = calculate_volatility_spike(
            prices, parameters["volatility_period"], parameters["volatility_threshold"]
        )
        date = price_data.index[-1]
        if volume_spike:
            print(f"Trade entry skipped due to high volume spike {coin} on {date}.")
            is_spike = True
        elif volatility_spike:
            print(f"Trade entry skipped due to high volatility spike {coin} on {date}.")
            is_spike = True

    return is_spike, volume_ratio, volatility_ratio
