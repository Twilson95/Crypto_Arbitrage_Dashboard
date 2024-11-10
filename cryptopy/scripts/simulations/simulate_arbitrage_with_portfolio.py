import pandas as pd

from cryptopy import (
    CointegrationCalculator,
    PortfolioManager,
    JsonHelper,
    TradingOpportunities,
)
from cryptopy.scripts.simulations.simulation_helpers import (
    get_trade_profit,
    filter_df,
    get_combined_df_of_data,
    get_avg_price_difference,
    calculate_expected_profit,
    get_todays_spread_data,
    get_bought_and_sold_amounts,
    is_volume_or_volatility_spike,
)

simulation_name = "expected_trading_strategy"
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
    "max_concurrent_trades": 8,
    "min_expected_profit": 0.008,  # must expect at least half a percent of the portfolio amount
    "max_expected_profit": 0.05,  # no more at risk as 5% percent of the portfolio amount
    "trade_size": 0.08,  # proportion of portfolio bought in each trade
}

folder_path = "../../../data/historical_data/Kraken_300_days"
price_df = get_combined_df_of_data(folder_path, "close")
volume_df = get_combined_df_of_data(folder_path, "volume")

print("historic_data_combined")

pair_combinations_df = pd.read_csv(cointegration_pairs_path)
pair_combinations = list(pair_combinations_df.itertuples(index=False, name=None))

portfolio_manager = PortfolioManager(parameters["max_concurrent_trades"], funds=1000)
print(price_df.head())

price_df.index = pd.to_datetime(price_df.index)
price_df.index = price_df.index.date
days_back = parameters["days_back"]
for current_date in price_df.index[days_back:]:
    print(f"{current_date}, {portfolio_manager.traded_pairs}, {cumulative_profit:.2f}")
    # in future we can sort these pairs based on profitability from other simulations
    for pair in sorted(pair_combinations, key=lambda x: x[0]):
        if "XRP/USD" in pair:
            continue
        currency_fees = {pair[0]: {"taker": 0.002}, pair[1]: {"taker": 0.002}}

        df_filtered = filter_df(price_df, current_date, days_back)
        coint_stat, p_value, crit_values = CointegrationCalculator.test_cointegration(
            df_filtered, pair
        )
        if p_value is None:
            continue

        open_event = portfolio_manager.get_open_trades(pair)
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
            close_event = TradingOpportunities.check_for_closing_event(
                todays_spread_data, p_value, parameters, open_event, hedge_ratio
            )
            if close_event:
                profit = get_trade_profit(
                    open_event,
                    close_event,
                    pair,
                    currency_fees,
                    df_filtered,
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
        open_event = portfolio_manager.get_open_trades(pair)
        avg_price_ratio = get_avg_price_difference(df_filtered, pair, hedge_ratio)

        if open_event is not None:
            continue
        if portfolio_manager.is_at_max_trades():
            continue
        if portfolio_manager.is_pair_traded(pair):
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
            current_funds = portfolio_manager.get_funds()
            trade_amount = current_funds * parameters["trade_size"]
            position_size = get_bought_and_sold_amounts(
                price_df, pair, open_event, current_date, trade_size=trade_amount
            )
            expected_profit = calculate_expected_profit(
                pair, todays_spread_data, currency_fees, position_size
            )
            print(f"{pair} expected profit: {expected_profit:.2f}")
            if (
                expected_profit < parameters["min_expected_profit"] * current_funds
                or expected_profit > parameters["max_expected_profit"] * current_funds
            ):
                print("Not within expected profit range")
                continue

            if is_volume_or_volatility_spike(price_df, volume_df, pair, parameters):
                continue

            open_event["position_size"] = position_size
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
JsonHelper.save_to_json(simulation_data, simulation_path)
