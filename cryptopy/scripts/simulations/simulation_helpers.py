from cryptopy import StatisticalArbitrage
import os
import pandas as pd
import json
import datetime
import glob


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
    todays_data = list_data.loc[date] if date in list_data.index else None
    return todays_data


def get_todays_spread_data(parameters, spread, current_date):
    rolling_window = parameters["rolling_window"]
    spread_mean = spread.rolling(window=rolling_window).mean()
    spread_std = spread.rolling(window=rolling_window).std()
    spread_threshold = parameters["spread_threshold"]

    upper_threshold = spread_mean + spread_threshold * spread_std
    lower_threshold = spread_mean - spread_threshold * spread_std

    upper_spread_threshold = parameters["spread_limit"]
    upper_spread_limit = spread_mean + upper_spread_threshold * spread_std
    lower_spread_limit = spread_mean - upper_spread_threshold * spread_std

    todays_spread = filter_list(spread, current_date)
    todays_spread_mean = filter_list(spread_mean, current_date)
    todays_spread_std = filter_list(spread_std, current_date)
    return {
        "date": current_date,
        "spread": todays_spread,
        "spread_mean": todays_spread_mean,
        "spread_std": todays_spread_std,
        "upper_threshold": filter_list(upper_threshold, current_date),
        "upper_limit": filter_list(upper_spread_limit, current_date),
        "lower_threshold": filter_list(lower_threshold, current_date),
        "lower_limit": filter_list(lower_spread_limit, current_date),
        "spread_deviation": abs(todays_spread - todays_spread_mean) / todays_spread_std,
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
        ),
        exit=(close_event["date"], close_event["spread_data"]["spread"]),
        pairs=pair,
        currency_fees=currency_fees,  # Example transaction cost
        price_df=df_filtered,
        usd_start=trade_amount,
        hedge_ratio=open_event["hedge_ratio"],
        exchange="test",
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


def calculate_expected_profit(pair, open_event, position_size, currency_fees):
    spread_data = open_event["spread_data"]
    spread = spread_data["spread"]
    spread_mean = spread_data["spread_mean"]

    fees = currency_fees[pair[0]]["taker"] * 2 + currency_fees[pair[1]]["taker"] * 2
    bought_amount = position_size["long_position"]["amount"]
    if open_event["direction"] == "short":
        bought_amount /= open_event["hedge_ratio"]
    return bought_amount * abs(spread - spread_mean) * (1 - fees)


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
