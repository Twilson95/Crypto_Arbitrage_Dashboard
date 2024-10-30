import pandas as pd

from cryptopy import CointegrationCalculator, PortfolioManager
from cryptopy.scripts.simulations.simulation_helpers import (
    get_trade_profit,
    check_for_closing_event,
    check_for_opening_event,
    filter_df,
    save_to_json,
    get_combined_df_of_prices,
    get_avg_price_difference,
    calculate_expected_profit,
    get_todays_spread_data,
)

simulation_name = "trade_size_measured_at_entry_time"
exchange_name = "Kraken"
historic_data_folder = f"../../../data/historical_data/{exchange_name}_300_days/"
cointegration_pairs_path = f"../../../data/historical_data/cointegration_pairs.csv"
simulation_path = f"../../../data/simulations/portfolio_sim/{simulation_name}.json"

trade_results = []
cumulative_profit = 0

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
    "min_expected_profit": 0.005,  # must expect at least half a percent of the portfolio amount
    "max_expected_profit": 0.05,  # no more at risk as 5% percent of the portfolio amount
}

folder_path = "../../../data/historical_data/Kraken_300_days"
df = get_combined_df_of_prices(folder_path)
print("historic_data_combined")

pair_combinations_df = pd.read_csv(cointegration_pairs_path)
pair_combinations = list(pair_combinations_df.itertuples(index=False, name=None))

portfolio_manager = PortfolioManager(parameters["max_concurrent_trades"], funds=1000)
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

        open_event = portfolio_manager.get_open_event(pair)
        if open_event is None:
            hedge_ratio = None
        else:
            hedge_ratio = open_event["hedge_ratio"]

        spread, hedge_ratio = CointegrationCalculator.calculate_spread(
            df_filtered, pair, hedge_ratio
        )

        todays_spread_data = get_todays_spread_data(parameters, spread, current_date)

        close_event = None
        if open_event:
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
                    open_event["trade_amount"],
                )
                portfolio_manager.on_closing_trade(pair, profit)
                cumulative_profit += profit
                open_event["hedge_ratio"] = hedge_ratio
                open_event["spread_data"] = todays_spread_data
                trade_results.append(
                    {
                        "pair": pair,
                        "open_event": open_event,
                        "close_event": close_event,
                        "profit": profit,
                    }
                )
        open_event = portfolio_manager.get_open_event(pair)
        avg_price_ratio = get_avg_price_difference(df_filtered, pair, hedge_ratio)

        if open_event is not None:
            continue
        if portfolio_manager.is_at_max_trades():
            continue
        if portfolio_manager.is_pair_traded(pair):
            continue

        open_event = check_for_opening_event(
            todays_spread_data, p_value, parameters, avg_price_ratio, hedge_ratio
        )
        if open_event:
            current_funds = portfolio_manager.get_funds()
            trade_amount = current_funds * 0.1
            expected_profit = calculate_expected_profit(
                df,
                pair,
                hedge_ratio,
                open_event,
                todays_spread_data,
                currency_fees,
                trade_amount,
            )
            print(f"{pair} expected profit: {expected_profit:.2f}")
            if (
                expected_profit < parameters["min_expected_profit"] * current_funds
                or expected_profit > parameters["max_expected_profit"] * current_funds
            ):
                continue

            open_event["trade_amount"] = trade_amount
            open_event["expected_profit"] = expected_profit
            open_event["hedge_ratio"] = hedge_ratio
            open_event["spread_data"] = todays_spread_data

            portfolio_manager.on_opening_trade(pair, open_event)

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
