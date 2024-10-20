import pandas as pd
from cryptopy import CointegrationCalculator
from cryptopy.scripts.simulation_helpers import (
    get_trade_profit,
    get_todays_data,
    check_for_closing_event,
    check_for_opening_event,
    read_historic_data_long_term,
    filter_df,
)

exchange_name = "Kraken"
historic_data_folder = f"../../data/historical_data/{exchange_name}_300_days/"
trade_results = []

# Simulation parameters
days_back = 100  # hedge ratio and p_value based off this
rolling_window = 30  # controls moving avg for mean and thresholds
p_value_open_threshold = 0.05
p_value_close_threshold = 0.2
expiry_days_threshold = 14
spread_threshold = 2


pair_combinations = [
    ("LTC/USD", "KSM/USD")
    # ("BTC/USD", "DOGE/USD"),
]

for pair in pair_combinations:
    open_position = False
    open_event = None
    currency_fees = {pair[0]: {"taker": 0.004}, pair[1]: {"taker": 0.004}}

    df_pair_1 = read_historic_data_long_term(pair[0], historic_data_folder)
    df_pair_2 = read_historic_data_long_term(pair[1], historic_data_folder)

    df = pd.concat(
        [df_pair_1["close"], df_pair_2["close"]], axis=1, keys=[pair[0], pair[1]]
    ).dropna()

    for current_date in df.index[days_back:]:
        df_filtered = filter_df(df, current_date, days_back)

        coint_stat, p_value, crit_values = CointegrationCalculator.test_cointegration(
            df_filtered, pair
        )
        spread, hedge_ratio = CointegrationCalculator.calculate_spread(
            df_filtered, pair
        )
        spread_mean = spread.rolling(window=rolling_window).mean()
        spread_std = spread.rolling(window=rolling_window).std()
        upper_threshold = spread_mean + spread_threshold * spread_std
        lower_threshold = spread_mean - spread_threshold * spread_std

        todays_spread = get_todays_data(spread, current_date)
        todays_spread_mean = get_todays_data(spread_mean, current_date)
        todays_spread_std = get_todays_data(spread_std, current_date)
        todays_upper_threshold = get_todays_data(upper_threshold, current_date)
        todays_lower_threshold = get_todays_data(lower_threshold, current_date)

        close_event = None
        if open_position:
            close_event, close_reason = check_for_closing_event(
                current_date,
                todays_spread,
                p_value,
                p_value_close_threshold,
                todays_spread_mean,
                open_event,
                expiry_days_threshold,
            )
            if close_event:
                profit = get_trade_profit(
                    open_event,
                    close_event,
                    pair,
                    currency_fees,
                    df_filtered,
                    hedge_ratio,
                    close_reason,
                )
                trade_results.append((open_event, close_event, profit))
                open_position = False

        if not open_position:
            open_event = check_for_opening_event(
                current_date,
                todays_spread,
                p_value,
                p_value_open_threshold,
                todays_upper_threshold,
                todays_lower_threshold,
            )
            if open_event:
                open_position = True
