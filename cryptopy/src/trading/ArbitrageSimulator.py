from cryptopy import CointegrationCalculator, TradingOpportunities, RiverPredictor

from cryptopy.scripts.simulations.simulation_helpers import (
    get_trade_profit,
    filter_df,
    get_avg_price_difference,
    calculate_expected_profit,
    get_bought_and_sold_amounts,
    filter_list,
)
from typing import Optional


class ArbitrageSimulator:
    def __init__(
        self,
        parameters,
        price_df,
        volume_df,
        portfolio_manager,
        pair_combinations,
        ml_model: Optional[RiverPredictor] = None,
        trades_before_prediction=100,
    ):
        self.parameters = parameters
        self.price_df = price_df
        self.volume_df = volume_df
        self.portfolio_manager = portfolio_manager
        self.pair_combinations = sorted(pair_combinations, key=lambda x: x[0])
        self.ml_model = ml_model
        self.trades_before_prediction = trades_before_prediction
        self.trade_results = []
        self.daily_trade_results = []
        self.open_opportunities = {}
        self.current_market_trend = None

    def run_simulation(self):
        days_back = self.parameters["days_back"]

        for current_date in self.price_df.index[days_back:]:
            self.simulate_day(current_date, days_back)
            cumulative_profit = self.portfolio_manager.get_cumulative_profit()

            print(
                f"\n-----------------------------------------------------\n"
                f"Date: {current_date} \n"
                f"Open Trades: {self.portfolio_manager.traded_pairs} \n"
                f"Total Profit {cumulative_profit:.2f} \n"
            )

        all_trades = self.portfolio_manager.get_all_trade_events()
        cumulative_profit = self.portfolio_manager.get_cumulative_profit()
        return all_trades, cumulative_profit
        # return self.all_trades, self.cumulative_profit

    def simulate_day(self, current_date, days_back):
        daily_opportunities, closed_trades = self.find_daily_opportunities(
            current_date, days_back
        )

        if self.ml_model and closed_trades:
            print("Training Model")
            self.ml_model.learn_from_data(closed_trades)

        predicted_success_trades = []
        for opp in daily_opportunities:
            if (
                self.ml_model
                and len(self.trade_results) > self.trades_before_prediction
            ):
                should_trade, probability_of_success = (
                    self.ml_model.predict_opportunity(opp)
                )
                if should_trade:
                    print("Successful opportunity predicted")
                    opp["probability_of_success"] = probability_of_success
                    predicted_success_trades.append(opp)
                else:
                    print("Should not trade")
            else:
                if not self.portfolio_manager.is_at_max_trades():
                    self.portfolio_manager.on_opening_trade(
                        opp["pair"], opp["open_event"]
                    )

        predicted_success_trades.sort(
            key=lambda x: x["probability_of_success"], reverse=True
        )

        for opp in predicted_success_trades:
            if not self.portfolio_manager.is_at_max_trades():
                self.portfolio_manager.on_opening_trade(opp["pair"], opp["open_event"])

    def find_daily_opportunities(self, current_date, days_back):
        daily_opportunities = []
        closed_trades = []

        price_df_filtered = filter_df(self.price_df, current_date, days_back)
        volume_df_filtered = filter_df(self.volume_df, current_date, days_back)

        trend_parameters = self.parameters["trend_parameters"]
        price_field = "BTC/USD"
        market_prices = price_df_filtered[[price_field]]
        market_trend = ArbitrageSimulator.get_current_trend(
            market_prices, price_field, trend_parameters
        )

        for pair in self.pair_combinations:
            if "XRP/USD" in pair:
                continue

            opportunity, closed_trade = self.review_pair(
                pair, current_date, price_df_filtered, volume_df_filtered
            )
            if closed_trade:
                closed_trades.append(closed_trade)
                del self.open_opportunities[pair]
            if opportunity:
                daily_opportunities.append(opportunity)
                self.open_opportunities[pair] = opportunity["open_event"]

                price_df_for_trend = filter_df(
                    self.price_df[[pair[0], pair[1]]],
                    current_date,
                    trend_parameters["long_window"],
                )

                coin_1_trend = ArbitrageSimulator.get_current_trend(
                    price_df_for_trend, pair[0], trend_parameters
                )
                coin_2_trend = ArbitrageSimulator.get_current_trend(
                    price_df_for_trend, pair[1], trend_parameters
                )

                opportunity["open_event"].update(
                    {
                        "market_trend": market_trend,
                        "coin_1_trend": coin_1_trend,
                        "coin_2_trend": coin_2_trend,
                    }
                )

        return daily_opportunities, closed_trades

    def review_pair(self, pair, current_date, price_df_filtered, volume_df_filtered):
        result = self.get_cointegration_and_spread_info(
            pair, price_df_filtered, current_date
        )
        if result is None:
            return None, None
        p_value, hedge_ratio, todays_spread_data, currency_fees = result

        # Attempt to close an existing trade if any
        closed_trade = self.attempt_closing_trade(
            pair,
            p_value,
            hedge_ratio,
            todays_spread_data,
            price_df_filtered,
            currency_fees,
        )

        # Check if we can open a new position after attempting to close
        opportunity = self.check_opening_conditions(
            pair,
            current_date,
            price_df_filtered,
            volume_df_filtered,
            p_value,
            hedge_ratio,
            todays_spread_data,
            currency_fees,
        )

        return opportunity, closed_trade

    @staticmethod
    def get_current_trend(price_df, price_field, trend_parameters):
        price_df = price_df.copy()
        short_window = trend_parameters["short_window"]
        long_window = trend_parameters["long_window"]
        change_threshold = trend_parameters["change_threshold"]

        price_df["short_MA"] = price_df[price_field].rolling(window=short_window).mean()
        price_df["long_MA"] = price_df[price_field].rolling(window=long_window).mean()

        sma_difference = abs(
            price_df["short_MA"].iloc[-1] - price_df["long_MA"].iloc[-1]
        )
        sma_percentage_difference = sma_difference / price_df["long_MA"].iloc[-1]

        if sma_percentage_difference < change_threshold:
            trend = "Flat"
        elif price_df["short_MA"].iloc[-1] > price_df["long_MA"].iloc[-1]:
            trend = "Uptrend"
        else:
            trend = "Downtrend"
        return trend

    def get_cointegration_and_spread_info(self, pair, price_df_filtered, current_date):
        currency_fees = {pair[0]: {"taker": 0.002}, pair[1]: {"taker": 0.002}}

        coint_stat, p_value, crit_values = CointegrationCalculator.test_cointegration(
            price_df_filtered, pair
        )
        if p_value is None:
            return None

        open_event = self.portfolio_manager.get_open_trades(pair)
        hedge_ratio = open_event["hedge_ratio"] if open_event else None

        spread, hedge_ratio = CointegrationCalculator.calculate_spread(
            price_df_filtered, pair, hedge_ratio
        )
        todays_spread_data = self.get_todays_spread_data(spread, current_date)

        return p_value, hedge_ratio, todays_spread_data, currency_fees

    def attempt_closing_trade(
        self,
        pair,
        p_value,
        hedge_ratio,
        todays_spread_data,
        price_df_filtered,
        currency_fees,
    ):
        # get all open events even those we aren't trading because we'll use these for training the model
        open_event = self.open_opportunities.get(pair, None)
        # open_event = self.portfolio_manager.get_open_trades(pair)
        if not open_event:
            return

        close_event = TradingOpportunities.check_for_closing_event(
            todays_spread_data, p_value, self.parameters, open_event, hedge_ratio
        )

        if close_event:
            profit = get_trade_profit(
                open_event,
                close_event,
                pair,
                currency_fees,
                price_df_filtered,
                open_event["trade_amount"],
            )
            # open_event["hedge_ratio"] = hedge_ratio
            # open_event["spread_data"] = todays_spread_data

            closing_trade = {
                "pair": pair,
                "open_event": open_event,
                "close_event": close_event,
                "profit": profit,
            }
            self.portfolio_manager.on_closing_trade(pair, closing_trade)
            self.trade_results.append(closing_trade)
            return closing_trade

    def check_opening_conditions(
        self,
        pair,
        current_date,
        price_df_filtered,
        volume_df_filtered,
        p_value,
        hedge_ratio,
        todays_spread_data,
        currency_fees,
    ):
        # Check if there's an open trade or if conditions prevent new trades
        # if self.portfolio_manager.get_open_trades(pair) is not None:
        if self.open_opportunities.get(pair) is not None:
            return
        if self.portfolio_manager.is_at_max_trades():
            return
        if self.portfolio_manager.is_pair_traded(pair):
            return

        avg_price_ratio = get_avg_price_difference(price_df_filtered, pair, hedge_ratio)
        new_open_event = TradingOpportunities.check_for_opening_event(
            todays_spread_data,
            p_value,
            self.parameters,
            avg_price_ratio,
            hedge_ratio,
            current_date,
        )

        if new_open_event is None:
            return

        # Validate potential trade
        current_funds = self.portfolio_manager.get_funds()
        trade_amount = current_funds * self.parameters["trade_size"]
        if (
            self.parameters["trade_size_same_risk"]
            and new_open_event["direction"] == "short"
        ):
            trade_amount /= new_open_event["hedge_ratio"]

        position_size = get_bought_and_sold_amounts(
            self.price_df, pair, new_open_event, current_date, trade_size=trade_amount
        )
        expected_profit = calculate_expected_profit(
            pair, new_open_event, position_size, currency_fees
        )

        print(f"{pair} expected profit: {expected_profit:.2f}")
        if not self.is_profit_in_range(expected_profit, current_funds):
            print("Not within expected profit range")
            return

        is_spike, volume_ratio, volatility_ratio = self.is_volume_or_volatility_spike(
            price_df_filtered, volume_df_filtered, pair
        )
        if is_spike:
            return

        if self.portfolio_manager.already_hold_coin_position(position_size):
            print("Already hold position in one of the coins")
            return

        # Set the necessary fields in the event
        new_open_event.update(
            {
                "p_value": p_value,
                "position_size": position_size,
                "trade_amount": trade_amount,
                "expected_profit": expected_profit,
                "hedge_ratio": hedge_ratio,
                "spread_data": todays_spread_data,
                "volume_ratio": volume_ratio,
                "volatility_ratio": volatility_ratio,
            }
        )

        return {"pair": pair, "open_event": new_open_event}

    def is_profit_in_range(self, expected_profit, current_funds):
        return (
            self.parameters["min_expected_profit"] * current_funds
            <= expected_profit
            <= self.parameters["max_expected_profit"] * current_funds
        )

    def is_volume_or_volatility_spike(self, price_data, volume_data, pair):
        parameters = self.parameters
        is_spike = False
        volume_ratio, volatility_ratio = 0, 0
        for coin in pair:
            volumes = volume_data[coin]
            prices = price_data[coin]
            volume_spike, volume_ratio = ArbitrageSimulator.calculate_volume_spike(
                volumes, parameters["volume_period"], parameters["volume_threshold"]
            )
            volatility_spike, volatility_ratio = (
                ArbitrageSimulator.calculate_volatility_spike(
                    prices,
                    parameters["volatility_period"],
                    parameters["volatility_threshold"],
                )
            )
            date = price_data.index[-1]
            if volume_spike:
                print(f"Trade entry skipped due to high volume spike {coin} on {date}.")
                is_spike = True
            elif volatility_spike:
                print(
                    f"Trade entry skipped due to high volatility spike {coin} on {date}."
                )
                is_spike = True

        return is_spike, volume_ratio, volatility_ratio

    @staticmethod
    def calculate_volume_spike(data, volume_period=30, volume_threshold=2):
        avg_volume = data.rolling(window=volume_period).mean().iloc[-1]
        current_volume = data.iloc[-1]
        return (
            current_volume > avg_volume * volume_threshold,
            current_volume / avg_volume,
        )

    @staticmethod
    def calculate_volatility_spike(
        data, volatility_period=30, volatility_threshold=1.5
    ):
        returns = data.pct_change()
        avg_volatility = returns.rolling(window=volatility_period).std().iloc[-1]
        current_volatility = returns.iloc[-1]
        if avg_volatility < 0:
            return (
                current_volatility < avg_volatility * volatility_threshold,
                current_volatility / avg_volatility,
            )
        else:
            return (
                current_volatility > avg_volatility * volatility_threshold,
                current_volatility / avg_volatility,
            )

    def get_todays_spread_data(self, spread, current_date):
        rolling_window = self.parameters["rolling_window"]
        spread_mean = spread.rolling(window=rolling_window).mean()
        spread_std = spread.rolling(window=rolling_window).std()
        spread_threshold = self.parameters["spread_threshold"]

        upper_threshold = spread_mean + spread_threshold * spread_std
        lower_threshold = spread_mean - spread_threshold * spread_std

        upper_spread_threshold = self.parameters["spread_limit"]
        upper_spread_limit = spread_mean + upper_spread_threshold * spread_std
        lower_spread_limit = spread_mean - upper_spread_threshold * spread_std

        todays_spread = filter_list(spread, current_date)
        todays_spread_mean = filter_list(spread_mean, current_date)
        todays_spread_std = filter_list(spread_std, current_date)
        return {
            "date": current_date,
            "spread": todays_spread,
            "spread_mean": todays_spread_mean,
            "spread_std": todays_spread_std,
            "upper_threshold": filter_list(upper_threshold, current_date),
            "upper_limit": filter_list(upper_spread_limit, current_date),
            "lower_threshold": filter_list(lower_threshold, current_date),
            "lower_limit": filter_list(lower_spread_limit, current_date),
            "spread_deviation": abs(todays_spread - todays_spread_mean)
            / todays_spread_std,
        }

    @staticmethod
    def extract_features_from_trade_data(daily_trade_data):
        x = []
        y = []
        for trade in daily_trade_data:
            open_event = trade["open_event"]
            close_event = trade["close_event"]
            # Build the feature dictionary
            features = {
                "coin_1": trade["pair"][0].split("/")[0],
                "coin_2": trade["pair"][1].split("/")[0],
                "p_value": open_event["p_value"],
                "spread_deviation": open_event["spread_data"]["spread_deviation"],
                "hedge_ratio": open_event["hedge_ratio"],
                "direction": 1 if open_event["direction"] == "long" else 0,
                "avg_price_ratio": open_event["avg_price_ratio"],
                "expected_profit": open_event["expected_profit"],
                "volume_ratio": open_event["volume_ratio"],
                "volatility_ratio": open_event["volatility_ratio"],
                # target stored internally, but here we just record features
                # "target": 1 if close_event["reason"] == "crossed_mean" else 0,
            }

            x.append(features)
            # Target: profit > 0 or not
            # y.append(1 if trade["profit"] > 0 else 0)
            y.append(1 if close_event["reason"] == "crossed_mean" else 0)

        return x, y
