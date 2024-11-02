import yaml
import pandas as pd
from cryptopy import (
    DataManager,
    PortfolioManager,
    CointegrationCalculator,
    JsonHelper,
    TradingOpportunities,
)
from cryptopy.scripts.simulations.simulation_helpers import (
    get_todays_spread_data,
    get_trade_profit,
    get_avg_price_difference,
    calculate_expected_profit,
    get_bought_and_sold_amounts,
)

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

with open("cryptopy/config/trading_config.yaml", "r") as f:
    exchange_config = yaml.safe_load(f)

data_manager = DataManager(exchange_config, live_trades=False)
exchange_name = "Kraken"
data_fetcher = data_manager.get_exchange(exchange_name)
current_balance = data_fetcher.get_balance()
print(f"current_balance {current_balance}")
open_trades = data_fetcher.get_open_trades()
print(f"open_trades {open_trades}")


usd_balance = 0
historical_prices = data_manager.get_historical_prices_for_all_currencies("Kraken")
historical_prices.index = pd.to_datetime(historical_prices.index)
historical_prices.index = historical_prices.index.date
current_date = historical_prices.index[-1]

for symbol, balance in current_balance.items():
    if symbol == "ZUSD/USD":
        usd_balance += balance
    try:
        current_price = historical_prices[symbol].iloc[-1]
        usd_balance += balance * current_price
    except:
        pass
print("usd_balance", usd_balance)

trades_path = "data/portfolio_data/Kraken/trades.json"
portfolio_manager = PortfolioManager(
    max_trades=8, funds=usd_balance, trades_path=trades_path
)
portfolio_manager.read_open_events()
print(portfolio_manager.get_all_open_events())

cointegration_pairs_path = f"data/historical_data/cointegration_pairs.csv"
pair_combinations_df = pd.read_csv(cointegration_pairs_path)
pair_combinations = list(pair_combinations_df.itertuples(index=False, name=None))

trade_results = []
for pair in sorted(pair_combinations, key=lambda x: x[0]):

    if "XRP/USD" in pair:
        continue
    currency_fees = {pair[0]: {"taker": 0.002}, pair[1]: {"taker": 0.002}}

    coint_stat, p_value, crit_values = CointegrationCalculator.test_cointegration(
        historical_prices, pair
    )
    if p_value is None:
        continue

    open_event = portfolio_manager.get_open_event(pair)
    if open_event is None:
        hedge_ratio = None
    else:
        hedge_ratio = open_event.get("hedge_ratio")

    spread, hedge_ratio = CointegrationCalculator.calculate_spread(
        historical_prices, pair, hedge_ratio
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
                historical_prices,
                hedge_ratio,
                open_event["trade_amount"],
            )
            portfolio_manager.on_closing_trade(pair, profit)
            open_event["hedge_ratio"] = hedge_ratio
            open_event["spread_data"] = todays_spread_data

            # delete trade result for this open event and append this
            trade_results.append(
                {
                    "pair": pair,
                    "open_event": open_event,
                    "close_event": close_event,
                    "profit": profit,
                }
            )
            # close trade

    open_event = portfolio_manager.get_open_event(pair)
    avg_price_ratio = get_avg_price_difference(historical_prices, pair, hedge_ratio)

    if open_event is not None:
        continue
    if portfolio_manager.is_at_max_trades():
        continue
    if portfolio_manager.is_pair_traded(pair):
        continue

    open_event = TradingOpportunities.check_for_opening_event(
        todays_spread_data, p_value, parameters, avg_price_ratio, hedge_ratio
    )
    if open_event:
        current_funds = portfolio_manager.get_funds()
        trade_amount = current_funds * 0.05
        position_size = get_bought_and_sold_amounts(
            historical_prices,
            pair,
            open_event,
            todays_spread_data,
            trade_size=trade_amount,
        )
        expected_profit = calculate_expected_profit(
            pair, todays_spread_data, currency_fees, position_size
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
        # make trade

simulation_data = {
    "trade_events": trade_results,
}
print(simulation_data)
# JsonHelper.save_to_json(simulation_data, trades_path)
