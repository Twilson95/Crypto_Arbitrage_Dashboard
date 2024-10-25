import pandas as pd

from cryptopy import CointegrationCalculator, PortfolioManager
from cryptopy.scripts.simulations.simulation_helpers import (
    get_trade_profit,
    get_todays_data,
    check_for_closing_event,
    check_for_opening_event,
    filter_df,
    save_to_json,
    get_combined_df_of_prices,
    get_avg_price_difference,
    calculate_expected_profit,
)

simulation_name = "portfolio_sim_p_0.01_mpr_5"
exchange_name = "Kraken"
historic_data_folder = f"../../../data/historical_data/{exchange_name}_300_days/"
cointegration_pairs_path = f"../../../data/historical_data/cointegration_pairs.csv"
simulation_path = f"../../../data/simulations/{simulation_name}.json"

trade_results = []
cumulative_profit = 0

# Simulation parameters
parameters = {
    "days_back": 100,  # hedge ratio and p_value based off this
    "rolling_window": 30,  # controls moving avg for mean and thresholds
    "p_value_open_threshold": 0.01,
    "p_value_close_threshold": 1,
    "expiry_days_threshold": 14,
    "spread_threshold": 2,
    "hedge_ratio_positive": True,
    "max_coin_price_ratio": 5,
    "max_concurrent_trades": 10,
    "min_expected_profit": 5,
}

folder_path = "../../../data/historical_data/Kraken_300_days"
df = get_combined_df_of_prices(folder_path)
print("historic_data_combined")

pair_combinations_df = pd.read_csv(cointegration_pairs_path)
pair_combinations = list(pair_combinations_df.itertuples(index=False, name=None))

portfolio_manager = PortfolioManager(parameters["max_concurrent_trades"])
print(df.head())

df.index = pd.to_datetime(df.index)
df.index = df.index.date
days_back = parameters["days_back"]
for current_date in df.index[days_back:]:
    print(f"{current_date}, {portfolio_manager.traded_pairs}, {cumulative_profit:.2f}")
    # in future we can sort these pairs based on profitability from other simulations
    for pair in sorted(pair_combinations, key=lambda x: x[0]):
        if "XRP/USD" in pair:
            continue
        currency_fees = {pair[0]: {"taker": 0.004}, pair[1]: {"taker": 0.004}}

        df_filtered = filter_df(df, current_date, days_back)
        coint_stat, p_value, crit_values = CointegrationCalculator.test_cointegration(
            df_filtered, pair
        )
        if p_value is None:
            continue

        spread, hedge_ratio = CointegrationCalculator.calculate_spread(
            df_filtered, pair
        )
        rolling_window = parameters["rolling_window"]
        spread_mean = spread.rolling(window=rolling_window).mean()
        spread_std = spread.rolling(window=rolling_window).std()
        spread_threshold = parameters["spread_threshold"]
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
                parameters["p_value_close_threshold"],
                todays_spread_mean,
                open_event,
                parameters["expiry_days_threshold"],
                hedge_ratio,
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
                        "price_ratio": open_event[3],
                    }
                )
        open_event = portfolio_manager.get_open_event(pair)
        if open_event is not None:
            continue
        if portfolio_manager.is_at_max_trades():
            continue
        if portfolio_manager.is_pair_traded(pair):
            continue
        if hedge_ratio < 0 and parameters["hedge_ratio_positive"]:
            continue
        avg_price_ratio = get_avg_price_difference(df_filtered, pair, hedge_ratio)
        if avg_price_ratio > parameters["max_coin_price_ratio"] or avg_price_ratio < 0:
            continue

        open_event = check_for_opening_event(
            current_date,
            todays_spread,
            p_value,
            parameters["p_value_open_threshold"],
            todays_upper_threshold,
            todays_lower_threshold,
            avg_price_ratio,
        )
        if open_event:
            expected_profit = calculate_expected_profit(
                current_date,
                df,
                pair,
                hedge_ratio,
                open_event,
                todays_spread,
                todays_spread_mean,
                currency_fees,
            )
            print(f"{pair} expected profit: {expected_profit}")
            if expected_profit < parameters["min_expected_profit"]:
                continue

            portfolio_manager.on_opening_trade(pair, open_event)

total_profit = sum(result["profit"] for result in trade_results)
number_of_trades = len(trade_results)
positive_trades = len([trade for trade in trade_results if trade["profit"] > 0])
successful_trades = len(
    [trade for trade in trade_results if trade["close_reason"] == "crossed_mean"]
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
