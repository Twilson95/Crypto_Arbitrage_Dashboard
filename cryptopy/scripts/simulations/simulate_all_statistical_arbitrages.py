import pandas as pd

from cryptopy import CointegrationCalculator, TradingOpportunities, JsonHelper
from cryptopy.scripts.simulations.simulation_helpers import (
    get_trade_profit,
    get_todays_spread_data,
    calculate_expected_profit,
    filter_df,
    get_avg_price_difference,
    get_combined_df_of_data,
    get_bought_and_sold_amounts,
    is_volume_or_volatility_spike,
)

simulation_name = "long_history_expiry_days_25"
exchange_name = "Kraken"
# historic_data_folder = f"../../../data/historical_data/{exchange_name}_300_days/"
historic_data_folder = f"../../../data/historical_data/{exchange_name}_long_history/"
cointegration_pairs_path = f"../../../data/historical_data/cointegration_pairs.csv"
simulation_path = f"../../../data/simulations/all_trades/{simulation_name}.json"

trade_results = []

# Simulation parameters
parameters = {
    "days_back": 100,  # hedge ratio and p_value based off this
    "rolling_window": 30,  # controls moving avg for mean and thresholds
    "p_value_open_threshold": 0.03,  # optimised, maximizes opportunities while keeping success rate
    "p_value_close_threshold": 1,  # optimised
    "expiry_days_threshold": 30,  # optimised, try 15 for portfolio to allow more trades
    "spread_threshold": 1.8,  # optimised 1.8 - 2
    "spread_limit": 3,  # optimised at 3-4
    "hedge_ratio_positive": True,
    "stop_loss_multiplier": 1.5,  # optimised 1.5-1.8
    "max_coin_price_ratio": 50,  # default 50
    "max_concurrent_trades": 8,
    "min_expected_profit": 0.004,  # must expect at least half a percent of the portfolio amount
    "max_expected_profit": 0.04,  # no more at risk as 5% percent of the portfolio amount
    "trade_size": 0.08,  # proportion of portfolio bought in each trade
    "volume_period": 30,
    "volume_threshold": 2,  # default 2
    "volatility_period": 30,
    "volatility_threshold": 1.5,  # default 1.5
    "max_each_coin": 2,
}

price_df = get_combined_df_of_data(historic_data_folder, "close")
volume_df = get_combined_df_of_data(historic_data_folder, "volume")

pair_combinations_df = pd.read_csv(cointegration_pairs_path)
pair_combinations = list(pair_combinations_df.itertuples(index=False, name=None))

for pair in sorted(pair_combinations, key=lambda x: x[0]):
    print(pair)
    open_position = False
    open_event = None
    currency_fees = {pair[0]: {"taker": 0.002}, pair[1]: {"taker": 0.002}}

    days_back = parameters["days_back"]
    rolling_window = parameters["rolling_window"]
    for current_date in price_df.index[days_back:]:
        price_df_filtered = filter_df(price_df, current_date, days_back)
        volume_df_filtered = filter_df(volume_df, current_date, days_back)

        coint_stat, p_value, crit_values = CointegrationCalculator.test_cointegration(
            price_df_filtered, pair
        )
        if p_value is None:
            continue

        if not open_position:
            hedge_ratio = None
        else:
            hedge_ratio = open_event["hedge_ratio"]

        spread, hedge_ratio = CointegrationCalculator.calculate_spread(
            price_df_filtered, pair, hedge_ratio
        )

        todays_spread_data = get_todays_spread_data(parameters, spread, current_date)

        close_event = None
        if open_position:
            close_event = TradingOpportunities.check_for_closing_event(
                todays_spread_data, p_value, parameters, open_event, hedge_ratio
            )
            if close_event:
                profit = get_trade_profit(
                    open_event,
                    close_event,
                    pair,
                    currency_fees,
                    price_df_filtered,
                    trade_amount=100,
                )
                close_event["hedge_ratio"] = hedge_ratio
                close_event["p_value"] = p_value
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

        avg_price_ratio = get_avg_price_difference(price_df_filtered, pair, hedge_ratio)

        if open_position:
            continue

        open_event = TradingOpportunities.check_for_opening_event(
            todays_spread_data,
            p_value,
            parameters,
            avg_price_ratio,
            hedge_ratio,
            current_date,
        )
        if open_event:
            position_sizes = get_bought_and_sold_amounts(
                price_df, pair, open_event, current_date, trade_size=100
            )

            expected_profit = calculate_expected_profit(
                pair, open_event, position_sizes, currency_fees
            )
            print(f"{pair} expected profit: {expected_profit:.2f}")
            if (
                expected_profit < parameters["min_expected_profit"] * 1000
                or expected_profit > parameters["max_expected_profit"] * 1000
            ):
                print("Not within expected profit range")
                continue

            is_spike, volume_ratio, volatility_ratio = is_volume_or_volatility_spike(
                price_df_filtered, volume_df_filtered, pair, parameters
            )

            if is_spike:
                continue

            open_event["p_value"] = p_value
            open_event["expected_profit"] = expected_profit
            open_event["spread_data"] = todays_spread_data
            open_event["volume_ratio"] = volume_ratio
            open_event["volatility_ratio"] = volatility_ratio
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
JsonHelper.save_to_json(simulation_data, simulation_path)
