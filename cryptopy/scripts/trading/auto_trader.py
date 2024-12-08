import yaml
import pandas as pd
from cryptopy import (
    DataManager,
    PortfolioManager,
    CointegrationCalculator,
    JsonHelper,
    TradingOpportunities,
    TradeManager,
    TradeManagerKraken,
)
from cryptopy.scripts.simulations.simulation_helpers import (
    get_todays_spread_data,
    get_trade_profit,
    get_avg_price_difference,
    calculate_expected_profit,
    get_bought_and_sold_amounts,
    is_volume_or_volatility_spike,
)

parameters = {
    "days_back": 100,  # hedge ratio and p_value based off this
    "rolling_window": 30,  # controls moving avg for mean and thresholds
    "p_value_open_threshold": 0.01,
    "p_value_close_threshold": 1,
    "expiry_days_threshold": 30,
    "spread_threshold": 2,
    "spread_limit": 5,
    "hedge_ratio_positive": True,
    "stop_loss_multiplier": 1.5,  # ratio of expected trade distance to use as stop loss location
    "max_coin_price_ratio": 5,
    "max_concurrent_trades": 10,
    "trade_size": 0.06,  # amount of portfolio to buy during each trade
    "min_expected_profit": 0.006,  # must expect at least half a percent of the portfolio amount
    "max_expected_profit": 0.030,  # no more at risk as 5% percent of the portfolio amount
    "volume_period": 30,
    "volume_threshold": 2,
    "volatility_period": 30,
    "volatility_threshold": 1.5,
    "max_each_coin": 2,
}

with open("cryptopy/config/trading_config.yaml", "r") as f:
    exchange_config = yaml.safe_load(f)

exchange_name = "Kraken"
write_output = True
make_trades = False

data_manager = DataManager(exchange_config, live_trades=False, use_cache=False)
# trading_manager = TradeManager(exchange_config, exchange_name, make_trades)
trading_manager = TradeManagerKraken(exchange_config, exchange_name, make_trades)

data_fetcher = data_manager.get_exchange(exchange_name)
current_balance = data_fetcher.get_balance()
print(f"current_balance {current_balance}")
open_trades = data_fetcher.get_open_trades()
print(f"open_trades {open_trades}")


usd_balance = 0
historical_prices = data_manager.get_historical_data_for_all_currencies("Kraken")
historical_prices.index = pd.to_datetime(historical_prices.index)
historical_prices.index = historical_prices.index.date
current_date = historical_prices.index[-1]

historical_volume = data_manager.get_historical_data_for_all_currencies(
    "Kraken", "volume"
)
historical_volume.index = pd.to_datetime(historical_volume.index)
historical_volume.index = historical_volume.index.date

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
trade_results = portfolio_manager.get_all_trade_events()

print(portfolio_manager.get_all_open_events())

cointegration_pairs_path = f"data/historical_data/cointegration_pairs.csv"
pair_combinations_df = pd.read_csv(cointegration_pairs_path)
pair_combinations = list(pair_combinations_df.itertuples(index=False, name=None))

for pair in sorted(pair_combinations, key=lambda x: x[0]):

    if "XRP/USD" in pair:
        continue
    currency_fees = {pair[0]: {"taker": 0.004}, pair[1]: {"taker": 0.004}}

    coint_stat, p_value, crit_values = CointegrationCalculator.test_cointegration(
        historical_prices, pair
    )
    if p_value is None:
        continue

    open_trade = portfolio_manager.get_open_trades(pair)
    if open_trade is None:
        hedge_ratio = None
    else:
        hedge_ratio = open_trade["open_event"].get("hedge_ratio")

    spread, hedge_ratio = CointegrationCalculator.calculate_spread(
        historical_prices, pair, hedge_ratio
    )
    todays_spread_data = get_todays_spread_data(parameters, spread, current_date)

    close_event = None
    if open_trade:
        close_event = TradingOpportunities.check_for_closing_event(
            todays_spread_data,
            p_value,
            parameters,
            open_trade["open_event"],
            hedge_ratio,
        )
        if close_event:
            if "hedge_ratio" not in open_trade["open_event"]:
                open_trade["open_event"]["hedge_ratio"] = hedge_ratio

            profit = get_trade_profit(
                open_trade["open_event"],
                close_event,
                pair,
                currency_fees,
                historical_prices,
                open_trade["position_size"]["trade_amount_usd"],
            )
            portfolio_manager.on_closing_trade(pair, profit)
            # open_trade["close_event"]["hedge_ratio"] = hedge_ratio
            # open_trade["close_event"]["spread_data"] = todays_spread_data

            position_size = open_trade.get("position_size")
            if position_size is None:
                position_size = get_bought_and_sold_amounts(
                    historical_prices,
                    pair,
                    open_trade,
                    todays_spread_data,
                    trade_size=open_trade["position_size"]["trade_amount_usd"],
                )
            open_trade["close_event"] = close_event
            open_trade["position_size"] = position_size
            open_trade["profit"] = profit

            trading_manager.close_arbitrage_positions(position_size)
            # data_fetcher.close_arbitrage_positions_sync(position_size)
            # delete trade result for this open event and append this
            trade_results = [
                event
                for event in trade_results
                # compare sets so reversed pairs are taken care of
                if (event["pair"][0], event["pair"][1]) != pair
                or "close_event" in event
            ]
            trade_results.append(open_trade)

    open_trade = portfolio_manager.get_open_trades(pair)
    avg_price_ratio = get_avg_price_difference(historical_prices, pair, hedge_ratio)

    if open_trade is not None:
        # print("No open trade")
        continue
    if portfolio_manager.is_at_max_trades():
        # print("At max trades")
        continue
    if portfolio_manager.is_pair_traded(pair):
        # print("Pair already traded")
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
            historical_prices,
            pair,
            open_event,
            current_date,
            trade_size=trade_amount,
        )
        expected_profit = calculate_expected_profit(
            pair, open_event, position_size, currency_fees
        )
        print(f"{pair} expected profit: {expected_profit:.2f}")
        if (
            expected_profit < parameters["min_expected_profit"] * current_funds
            or expected_profit > parameters["max_expected_profit"] * current_funds
        ):
            print("Not within expected profit range")
            continue

        if is_volume_or_volatility_spike(
            historical_prices, historical_volume, pair, parameters
        ):
            continue

        if portfolio_manager.already_hold_coin_position(position_size):
            print("Already hold position in one of the coins")
            continue

        open_trade = {
            "pair": pair,
            "open_event": open_event,
            "position_size": position_size,
            "expected_profit": expected_profit,
        }
        print(f"open_trade {open_trade}")
        trading_manager.open_arbitrage_positions(open_trade["position_size"])
        # data_fetcher.open_arbitrage_positions_sync(open_trade["position_size"])
        trade_results.append(open_trade)
        portfolio_manager.on_opening_trade(pair, open_trade)

simulation_data = {
    "trade_events": trade_results,
}
print(simulation_data)

if write_output:
    JsonHelper.save_to_json(simulation_data, trades_path)
    print("saved trading data")

data_manager.shutdown_sync()
