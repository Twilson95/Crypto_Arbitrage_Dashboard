import pandas as pd

from cryptopy import CointegrationCalculator, PortfolioManager
from cryptopy.scripts.simulations.simulation_helpers import (
    get_trade_profit,
    get_todays_data,
    check_for_closing_event,
    check_for_opening_event,
    read_historic_data_long_term,
    filter_df,
    save_to_json,
    get_combined_df_of_prices,
)

simulation_name = "portfolio_managed_sim"
exchange_name = "Kraken"
historic_data_folder = f"../../data/historical_data/{exchange_name}_300_days/"
cointegration_pairs_path = f"../../../data/historical_data/cointegration_pairs.csv"
simulation_path = f"../../data/simulations/{simulation_name}.json"

trade_results = []
cumulative_profit = 0

# Simulation parameters
days_back = 100  # hedge ratio and p_value based off this
rolling_window = 30  # controls moving avg for mean and thresholds
p_value_open_threshold = 0.05
p_value_close_threshold = 1
expiry_days_threshold = 14
spread_threshold = 2

folder_path = "../../../data/historical_data/Kraken_300_days"
df = get_combined_df_of_prices(folder_path)
print("historic_data_combined")

pair_combinations_df = pd.read_csv(cointegration_pairs_path)
pair_combinations = list(pair_combinations_df.itertuples(index=False, name=None))

portfolio_manager = PortfolioManager()
print(df)

df.index = pd.to_datetime(df.index)
df.index = df.index.date
for current_date in df.index[days_back:]:
    print(current_date, portfolio_manager.traded_pairs, cumulative_profit)
    # in future we can sort these pairs based on profitability from other simulations
    for pair in sorted(pair_combinations, key=lambda x: x[0]):
        currency_fees = {pair[0]: {"taker": 0.004}, pair[1]: {"taker": 0.004}}

        df_filtered = filter_df(df, current_date, days_back)

        coint_stat, p_value, crit_values = CointegrationCalculator.test_cointegration(
            df_filtered, pair
        )
        if p_value is None:
            continue

        # print(f"{pair}, p_value {p_value}")
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
        # if open position exists check for close signal
        open_event = portfolio_manager.get_open_event(pair)
        if open_event:
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
                portfolio_manager.on_closing_trade(pair)
                cumulative_profit += profit
                trade_results.append(
                    {
                        "pair": pair,
                        "open_date": open_event[0],
                        "open_spread": open_event[1],
                        "open_direction": open_event[2],
                        "close_date": close_event[0],
                        "close_spread": close_event[1],
                        "close_reason": close_reason,
                        "profit": profit,
                    }
                )
        open_event = portfolio_manager.get_open_event(pair)
        if open_event is not None:
            # print(f"{pair} already has open position")
            continue
        if portfolio_manager.is_at_max_trades():
            # print(f"portfolio is at max open positions")
            continue
        if portfolio_manager.is_pair_traded(pair):
            # print(f"{pair} one of coins is already in an open position")
            continue

        open_event = check_for_opening_event(
            current_date,
            todays_spread,
            p_value,
            p_value_open_threshold,
            todays_upper_threshold,
            todays_lower_threshold,
        )
        if open_event:
            portfolio_manager.on_opening_trade(pair, open_event)

total_profit = sum(result["profit"] for result in trade_results)
print(f"Total Expected Profit: {total_profit:.2f}")
simulation_data = {
    "parameters": {
        "days_back": days_back,
        "rolling_window": rolling_window,
        "p_value_open_threshold": p_value_open_threshold,
        "p_value_close_threshold": p_value_close_threshold,
        "expiry_days_threshold": expiry_days_threshold,
        "spread_threshold": spread_threshold,
    },
    "trade_events": trade_results,
    "total_profit": total_profit,
}
save_to_json(simulation_data, simulation_path)
