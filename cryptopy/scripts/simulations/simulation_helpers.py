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


def filter_list_to_current_date(list_data, current_date):
    todays_data = (
        list_data.loc[current_date] if current_date in list_data.index else None
    )
    return todays_data


def get_todays_spread_data(parameters, spread, current_date):
    rolling_window = parameters["rolling_window"]
    spread_mean = spread.rolling(window=rolling_window).mean()
    spread_std = spread.rolling(window=rolling_window).std()
    spread_threshold = parameters["spread_threshold"]
    upper_threshold = spread_mean + spread_threshold * spread_std
    lower_threshold = spread_mean - spread_threshold * spread_std

    return {
        "date": current_date,
        "spread": filter_list_to_current_date(spread, current_date),
        "spread_mean": filter_list_to_current_date(spread_mean, current_date),
        "spread_std": filter_list_to_current_date(spread_std, current_date),
        "upper_threshold": filter_list_to_current_date(upper_threshold, current_date),
        "lower_threshold": filter_list_to_current_date(lower_threshold, current_date),
    }


def get_trade_profit(
    open_event,
    close_event,
    pair,
    currency_fees,
    df_filtered,
    hedge_ratio,
    trade_amount,
):
    arbitrage = StatisticalArbitrage.statistical_arbitrage_iteration(
        entry=(open_event["date"], open_event["spread"], open_event["direction"]),
        exit=(close_event["date"], close_event["spread"]),
        pairs=pair,
        currency_fees=currency_fees,  # Example transaction cost
        price_df=df_filtered,
        usd_start=trade_amount,
        hedge_ratio=hedge_ratio,
        exchange="test",
    )
    if arbitrage:
        profit = arbitrage.get("summary_header", {}).get("total_profit", 0)
        print(
            f"{pair}, date: {open_event['date']} to {close_event['date']}, close_reason: {close_event['reason']}: profit {profit:.2f}"
        )
        return profit


def get_combined_df_of_prices(folder_path):
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    dfs = []
    for file in csv_files:
        file_name = os.path.basename(file).replace(".csv", "")
        new_column_name = file_name.replace("_", "/")
        df = pd.read_csv(file, index_col=0)
        df = df[["close"]].rename(columns={"close": new_column_name})
        dfs.append(df)
    combined_df = pd.concat(dfs, axis=1, join="outer")
    return combined_df


def get_avg_price_difference(df, pair, hedge_ratio):
    mean_prices1 = df[pair[0]].mean()
    mean_prices2 = df[pair[1]].mean()

    return mean_prices1 / (mean_prices2 * hedge_ratio)


def calculate_expected_profit(pair, todays_data, currency_fees, position_sizes):
    spread = todays_data["spread"]
    spread_mean = todays_data["spread_mean"]

    fees = currency_fees[pair[0]]["taker"] * 4
    bought_amount = position_sizes["long_position"]["amount"]
    return bought_amount * abs(spread - spread_mean) * (1 - fees)


def get_bought_and_sold_amounts(df, pair, open_event, todays_data, trade_size=100):
    current_date = todays_data["date"]
    hedge_ratio = open_event["hedge_ratio"]

    if open_event["direction"] == "short":
        bought_coin = pair[1]
        sold_coin = pair[0]
        buy_coin_price = filter_list_to_current_date(df[bought_coin], current_date)
        adjusted_value = buy_coin_price * hedge_ratio
        bought_amount = trade_size / adjusted_value
        sold_amount = bought_amount / hedge_ratio
    elif open_event["direction"] == "long":
        bought_coin = pair[0]
        sold_coin = pair[1]
        buy_coin_price = filter_list_to_current_date(df[bought_coin], current_date)
        bought_amount = trade_size / buy_coin_price
        sold_amount = trade_size * hedge_ratio
    else:
        return None

    trade_size = {
        "long_position": {"coin": bought_coin, "amount": bought_amount},
        "short_position": {"coin": sold_coin, "amount": sold_amount},
    }

    return trade_size
