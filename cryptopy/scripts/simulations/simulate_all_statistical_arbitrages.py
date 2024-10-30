import pandas as pd

from cryptopy import CointegrationCalculator
from cryptopy.scripts.simulations.simulation_helpers import (
    get_trade_profit,
    get_todays_spread_data,
    check_for_closing_event,
    check_for_opening_event,
    calculate_expected_profit,
    filter_df,
    save_to_json,
    get_avg_price_difference,
    get_combined_df_of_prices,
)

simulation_name = "max_expected_profit"
exchange_name = "Kraken"
historic_data_folder = f"../../../data/historical_data/{exchange_name}_300_days/"
cointegration_pairs_path = f"../../../data/historical_data/cointegration_pairs.csv"
simulation_path = f"../../../data/simulations/all_trades/{simulation_name}.json"

trade_results = []

# Simulation parameters
parameters = {
    "days_back": 100,  # hedge ratio and p_value based off this
    "rolling_window": 30,  # controls moving avg for mean and thresholds
    "p_value_open_threshold": 0.01,
    "p_value_close_threshold": 1,
    "expiry_days_threshold": 30,
    "spread_threshold": 2,
    "hedge_ratio_positive": True,
    "stop_loss_multiplier": 1.5,  # ratio of expected trade distance to use as stop loss location
    "max_coin_price_ratio": 5,
    "max_concurrent_trades": 10,
    "min_expected_profit": 0.005,
    "max_expected_profit": 0.05,
}

df = get_combined_df_of_prices(historic_data_folder)

pair_combinations_df = pd.read_csv(cointegration_pairs_path)
pair_combinations = list(pair_combinations_df.itertuples(index=False, name=None))

# pair_combinations = [("LTC/USD", "KSM/USD")]
# ]
df.index = pd.to_datetime(df.index)
df.index = df.index.date
for pair in sorted(pair_combinations, key=lambda x: x[0]):
    print(pair)
    open_position = False
    open_event = None
    currency_fees = {pair[0]: {"taker": 0.004}, pair[1]: {"taker": 0.004}}

    days_back = parameters["days_back"]
    rolling_window = parameters["rolling_window"]
    for current_date in df.index[days_back:]:
        df_filtered = filter_df(df, current_date, days_back)

        coint_stat, p_value, crit_values = CointegrationCalculator.test_cointegration(
            df_filtered, pair
        )
        if p_value is None:
            continue

        if not open_position:
            hedge_ratio = None
        else:
            hedge_ratio = open_event["hedge_ratio"]

        spread, hedge_ratio = CointegrationCalculator.calculate_spread(
            df_filtered, pair, hedge_ratio
        )

        todays_spread_data = get_todays_spread_data(parameters, spread, current_date)

        close_event = None
        if open_position:
            close_event = check_for_closing_event(
                todays_spread_data, p_value, parameters, open_event, hedge_ratio
            )
            if close_event:
                profit = get_trade_profit(
                    open_event,
                    close_event,
                    pair,
                    currency_fees,
                    df_filtered,
                    hedge_ratio,
                    trade_amount=100,
                )
                close_event["hedge_ratio"] = hedge_ratio
                close_event["spread_data"] = todays_spread_data
                trade_results.append(
                    {
                        "pair": pair,
                        "open_event": open_event,
                        "close_event": close_event,
                        "profit": profit,
                    }
                )
                open_position = False

        avg_price_ratio = get_avg_price_difference(df_filtered, pair, hedge_ratio)

        if open_position:
            continue

        open_event = check_for_opening_event(
            todays_spread_data, p_value, parameters, avg_price_ratio, hedge_ratio
        )
        if open_event:
            expected_profit = calculate_expected_profit(
                df,
                pair,
                hedge_ratio,
                open_event,
                todays_spread_data,
                currency_fees,
            )
            print(f"{pair} expected profit: {expected_profit:.2f}")
            if (
                expected_profit < parameters["min_expected_profit"] * 1000
                or expected_profit > parameters["max_expected_profit"] * 1000
            ):
                continue

            open_event["expected_profit"] = expected_profit
            open_event["hedge_ratio"] = hedge_ratio
            open_event["spread_data"] = todays_spread_data
            open_position = True

total_profit = sum(result["profit"] for result in trade_results)
number_of_trades = len(trade_results)
positive_trades = len([trade for trade in trade_results if trade["profit"] > 0])
successful_trades = len(
    [
        trade
        for trade in trade_results
        if trade["close_event"]["reason"] == "crossed_mean"
    ]
)

print(f"Total Expected Profit: {total_profit:.2f}")
simulation_data = {
    "parameters": parameters,
    "stats": {
        "total_profit": total_profit,
        "success_rate": successful_trades / number_of_trades,
        "positive_results": positive_trades / number_of_trades,
        "number_of_trades": number_of_trades,
    },
    "trade_events": trade_results,
}
save_to_json(simulation_data, simulation_path)
