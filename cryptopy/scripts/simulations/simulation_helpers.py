from cryptopy import StatisticalArbitrage
import os
import pandas as pd
import json
import datetime
import glob


def check_for_opening_event(
    todays_data, p_value, parameters, avg_price_ratio, hedge_ratio
):
    current_date = todays_data["date"]
    upper_threshold = todays_data["upper_threshold"]
    lower_threshold = todays_data["lower_threshold"]
    spread = todays_data["spread"]
    spread_mean = todays_data["spread_mean"]

    if hedge_ratio < 0 and parameters["hedge_ratio_positive"]:
        return None

    if avg_price_ratio > parameters["max_coin_price_ratio"] or avg_price_ratio < 0:
        return None

    if p_value < parameters["p_value_open_threshold"]:
        spread_distance = abs(spread - spread_mean)
        if spread > upper_threshold:
            short_stop_loss = (
                spread + spread_distance * parameters["stop_loss_multiplier"]
            )

            return {
                "date": current_date,
                "spread": spread,
                "direction": "short",
                "avg_price_ratio": avg_price_ratio,
                "stop_loss": short_stop_loss,
            }
        elif spread < lower_threshold:
            long_stop_loss = (
                spread - spread_distance * parameters["stop_loss_multiplier"]
            )
            return {
                "date": current_date,
                "spread": spread,
                "direction": "long",
                "avg_price_ratio": avg_price_ratio,
                "stop_loss": long_stop_loss,
            }
    return None


def check_for_closing_event(
    todays_data,
    p_value,
    parameters,
    open_event,
    hedge_ratio,
):
    current_date = todays_data["date"]
    spread = todays_data["spread"]
    spread_mean = todays_data["spread_mean"]

    if p_value > parameters["p_value_close_threshold"]:
        return {"date": current_date, "spread": spread, "reason": "p_value"}

    if hedge_ratio < 0:
        return {
            "date": current_date,
            "spread": spread,
            "reason": "negative_hedge_ratio",
        }

    if (current_date - open_event["date"]).days > parameters["expiry_days_threshold"]:
        return {"date": current_date, "spread": spread, "reason": "expired"}

    if open_event["direction"] == "short" and spread < spread_mean:
        return {"date": current_date, "spread": spread, "reason": "crossed_mean"}
    elif open_event["direction"] == "long" and spread > spread_mean:
        return {"date": current_date, "spread": spread, "reason": "crossed_mean"}

    spread_distance = abs(open_event["spread"] - spread_mean)
    short_stop_loss = open_event["spread"] + spread_distance
    long_stop_loss = open_event["spread"] - spread_distance
    if open_event["direction"] == "short" and spread > short_stop_loss:
        return {"date": current_date, "spread": spread, "reason": "stop_loss"}
    elif open_event["direction"] == "long" and spread < long_stop_loss:
        return {"date": current_date, "spread": spread, "reason": "stop_loss"}

    return None


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


def get_todays_data(parameters, spread, current_date):
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
    df, pair, hedge_ratio, open_event, todays_data, currency_fees
):
    current_date = todays_data["date"]
    spread = todays_data["spread"]
    spread_mean = todays_data["spread_mean"]

    if open_event["direction"] == "short":
        buy_coin_price = filter_list_to_current_date(df[pair[1]], current_date)
        adjusted_value = buy_coin_price * hedge_ratio
        bought_amount = 100 / adjusted_value
    elif open_event["direction"] == "long":
        buy_coin_price = filter_list_to_current_date(df[pair[0]], current_date)
        bought_amount = 100 / buy_coin_price
    else:
        return None

    fees = currency_fees[pair[0]]["taker"] * 2
    return bought_amount * abs(spread - spread_mean) * (1 - fees)
