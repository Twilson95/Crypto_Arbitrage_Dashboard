# === CONFIGURABLE STRATEGY PARAMETERS === #
CONFIG = {
    "z_entry": 2.0,
    "z_exit": 0.2,
    "risk_per_trade": 0.03,
    "max_active_trades": 10,
    "max_correlation_threshold": 0.8,
    "initial_cash": 100_000,
    "lookback": 100,
    "min_rows": 500,
    "max_hold_days": 10,
    "stop_loss_z": 3.5,
}

import os
import numpy as np
import pandas as pd
from glob import glob
from statsmodels.tsa.stattools import coint
from statsmodels.regression.linear_model import OLS
import statsmodels.api as sm
from matplotlib import pyplot as plt


# === DATA LOADER === #
def load_log_prices(data_folder, min_rows):
    files = sorted(glob(os.path.join(data_folder, "*.csv")))
    price_data = {}
    for file in files:
        df = pd.read_csv(file)
        df["datetime"] = pd.to_datetime(df["datetime"])
        df.set_index("datetime", inplace=True)
        if df["close"].count() >= min_rows:
            symbol = os.path.basename(file).replace(".csv", "")
            price_data[symbol] = np.log(df["close"])
    log_prices = pd.concat(price_data.values(), axis=1, join="inner")
    log_prices.columns = list(price_data.keys())
    return log_prices.dropna()


# === SIGNAL ENGINE === #
def compute_pair_spread(a, b):
    model = OLS(a, sm.add_constant(b)).fit()
    hedge = model.params.iloc[1]
    spread = a - hedge * b
    return spread, hedge


def calculate_zscore(series, window):
    mean = series.rolling(window).mean()
    std = series.rolling(window).std()
    return (series - mean) / std


def is_cointegrated(series_a, series_b, pval_threshold=0.05):
    try:
        score, pval, _ = coint(series_a, series_b)
        return pval < pval_threshold
    except:
        return False


def is_strongly_cointegrated(a, b, threshold=0.05):
    score, pval, _ = coint(a, b)
    model = OLS(a, sm.add_constant(b)).fit()
    r2 = model.rsquared
    return pval < threshold and r2 > 0.9


# === BACKTEST ENGINE === #
def backtest_strategy(log_prices, config):
    portfolio_value = config["initial_cash"]
    active_trades = []
    pnl_curve = []

    all_pairs = [
        (a, b)
        for i, a in enumerate(log_prices.columns)
        for b in log_prices.columns[i + 1 :]
    ]

    dates = log_prices.index

    for t in range(config["lookback"] + 1, len(dates)):
        print(f"days {t}")
        today = dates[t]
        today_pnl = 0

        # Update P&L of active trades
        new_active = []
        for trade in active_trades:
            a, b = trade["pair"]
            price_a_now = log_prices[a].iloc[t]
            price_b_now = log_prices[b].iloc[t]
            pnl = trade["direction"] * (price_a_now - trade["price_a"]) - trade[
                "direction"
            ] * trade["hedge"] * (price_b_now - trade["price_b"])
            spread_series = (
                log_prices[a].iloc[t - config["lookback"] : t]
                - trade["hedge"] * log_prices[b].iloc[t - config["lookback"] : t]
            )
            z = calculate_zscore(spread_series, config["lookback"]).iloc[-1]

            max_hold = t - trade["entry_index"] > config["max_hold_days"]
            stop_loss = abs(z) > config["stop_loss_z"]
            exit_signal = (trade["direction"] == 1 and z >= -config["z_exit"]) or (
                trade["direction"] == -1 and z <= config["z_exit"]
            )

            if exit_signal or max_hold or stop_loss:
                today_pnl += pnl * trade["capital"]
            else:
                new_active.append(trade)

        active_trades = new_active

        # Evaluate new entries
        if len(active_trades) < config["max_active_trades"]:
            for a, b in all_pairs:
                if any(trade["pair"] == (a, b) for trade in active_trades):
                    continue
                series_a = log_prices[a].iloc[:t]
                series_b = log_prices[b].iloc[:t]
                if (
                    len(series_a) < config["lookback"]
                    or len(series_b) < config["lookback"]
                ):
                    continue

                if not is_strongly_cointegrated(series_a, series_b):
                    continue

                spread, hedge = compute_pair_spread(series_a, series_b)
                zscore = calculate_zscore(spread, config["lookback"]).iloc[-1]
                if abs(zscore) > config["z_entry"]:
                    direction = -1 if zscore > 0 else 1
                    price_a = log_prices[a].iloc[t]
                    price_b = log_prices[b].iloc[t]
                    capital = portfolio_value * config["risk_per_trade"]
                    active_trades.append(
                        {
                            "pair": (a, b),
                            "direction": direction,
                            "price_a": price_a,
                            "price_b": price_b,
                            "hedge": hedge,
                            "capital": capital,
                            "entry_index": t,
                        }
                    )
                    if len(active_trades) >= config["max_active_trades"]:
                        break

        pnl_curve.append(today_pnl)
        portfolio_value += today_pnl

    return pd.Series(np.cumsum(pnl_curve), index=dates[config["lookback"] + 1 :])


# === USAGE === #
data_folder = r"C:\Users\thoma\PycharmProjects\CryptoDashboard\data\historical_data\Kraken_long_history"
log_prices = load_log_prices(data_folder, CONFIG["min_rows"])
equity_curve = backtest_strategy(log_prices, CONFIG)
equity_curve.plot(title="Cumulative P&L")
plt.show()
