from cryptopy import StatisticalArbitrage
import os
import pandas as pd
import json
import datetime
import glob


def check_for_opening_event(
    current_date,
    todays_spread,
    p_value,
    p_value_open_threshold,
    upper_threshold,
    lower_threshold,
    avg_price_ratio,
):
    if p_value < p_value_open_threshold:
        if todays_spread > upper_threshold:
            return current_date, todays_spread, "short", avg_price_ratio
        elif todays_spread < lower_threshold:
            return current_date, todays_spread, "long", avg_price_ratio
    return None


def check_for_closing_event(
    current_date,
    todays_spread,
    p_value,
    p_value_close_threshold,
    spread_mean,
    open_event,
    expiry_days_threshold,
    hedge_ratio,
):
    if p_value > p_value_close_threshold:
        return (current_date, todays_spread), "p_value"

    if hedge_ratio < 0:
        return (current_date, todays_spread), "negative_hedge_ratio"

    if open_event and (current_date - open_event[0]).days > expiry_days_threshold:
        return (current_date, todays_spread), "expired"

    if open_event and open_event[2] == "short" and todays_spread < spread_mean:
        return (current_date, todays_spread), "crossed_mean"
    elif open_event and open_event[2] == "long" and todays_spread > spread_mean:
        return (current_date, todays_spread), "crossed_mean"

    return None, None


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


def get_todays_data(list_data, current_date):
    todays_data = (
        list_data.loc[current_date] if current_date in list_data.index else None
    )
    return todays_data


def get_trade_profit(
    open_event,
    close_event,
    pair,
    currency_fees,
    df_filtered,
    hedge_ratio,
    close_reason,
    trade_amount,
):
    arbitrage = StatisticalArbitrage.statistical_arbitrage_iteration(
        entry=open_event[0:3],
        exit=close_event,
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
            f"{pair}, date: {open_event[0]} to {close_event[0]}, close_reason: {close_reason}: profit {profit:.2f}"
        )
        return profit


def json_serial(obj):
    if isinstance(obj, (datetime.datetime, datetime.date)):
        return (
            obj.isoformat()
        )  # Convert to ISO format string (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS)
    raise TypeError(f"Type {type(obj)} not serializable")


def save_to_json(data, filename):
    directory = os.path.dirname(filename)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

    with open(filename, "w") as json_file:
        json.dump(data, json_file, indent=4, default=json_serial)


def read_from_json(filename):
    with open(filename, "r") as json_file:
        return json.load(json_file)


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


def calculate_expected_profit(
    current_date,
    df,
    pair,
    hedge_ratio,
    open_event,
    todays_spread,
    todays_spread_mean,
    currency_fees,
):
    if open_event[2] == "short":
        buy_coin_price = get_todays_data(df[pair[1]], current_date)
        adjusted_value = buy_coin_price * hedge_ratio
        bought_amount = 100 / adjusted_value
    elif open_event[2] == "long":
        buy_coin_price = get_todays_data(df[pair[0]], current_date)
        bought_amount = 100 / buy_coin_price
    else:
        return None

    fees = currency_fees[pair[0]]["taker"] * 2
    return bought_amount * abs(todays_spread - todays_spread_mean) * (1 - fees)
