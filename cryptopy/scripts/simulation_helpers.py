from cryptopy import StatisticalArbitrage
import os
import pandas as pd
import json
import datetime


def check_for_opening_event(
    current_date,
    todays_spread,
    p_value,
    p_value_open_threshold,
    upper_threshold,
    lower_threshold,
):
    if p_value < p_value_open_threshold:
        if todays_spread > upper_threshold:
            return current_date, todays_spread, "short"
        elif todays_spread < lower_threshold:
            return current_date, todays_spread, "long"
    return None


def check_for_closing_event(
    current_date,
    todays_spread,
    p_value,
    p_value_close_threshold,
    spread_mean,
    open_event,
    expiry_days_threshold,
):
    if p_value > p_value_close_threshold:
        return (current_date, todays_spread), "p_value"

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
    open_event, close_event, pair, currency_fees, df_filtered, hedge_ratio, close_reason
):
    arbitrage = StatisticalArbitrage.statistical_arbitrage_iteration(
        entry=open_event,
        exit=close_event,
        pairs=pair,
        currency_fees=currency_fees,  # Example transaction cost
        price_df=df_filtered,
        usd_start=100,
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
