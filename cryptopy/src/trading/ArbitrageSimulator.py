from cryptopy import CointegrationCalculator, TradingOpportunities

from cryptopy.scripts.simulations.simulation_helpers import (
    get_trade_profit,
    filter_df,
    get_avg_price_difference,
    calculate_expected_profit,
    get_bought_and_sold_amounts,
    filter_list,
)


class ArbitrageSimulator:
    def __init__(
        self,
        parameters,
        price_df,
        volume_df,
        portfolio_manager,
        pair_combinations,
        ml_model=None,
    ):
        self.parameters = parameters
        self.price_df = price_df
        self.volume_df = volume_df
        self.portfolio_manager = portfolio_manager
        self.pair_combinations = pair_combinations
        self.ml_model = ml_model
        self.trade_results = []
        self.cumulative_profit = 0.0

    def run_simulation(self):
        days_back = self.parameters["days_back"]

        for current_date in self.price_df.index[days_back:]:
            self.simulate_day(current_date, days_back)

            print(
                f"{current_date}, {self.portfolio_manager.traded_pairs}, {self.cumulative_profit:.2f}"
            )

        return self.trade_results, self.cumulative_profit

    def simulate_day(self, current_date, days_back):

        daily_opportunities = self.find_daily_opportunities(current_date, days_back)

        if self.ml_model and daily_opportunities:
            daily_opportunities = self.apply_ml_decision(daily_opportunities)

        for opp in daily_opportunities:
            self.portfolio_manager.on_opening_trade(opp["pair"], opp["open_event"])

    def find_daily_opportunities(self, current_date, days_back):
        daily_opportunities = []

        for pair in sorted(self.pair_combinations, key=lambda x: x[0]):
            if "XRP/USD" in pair:
                continue

            price_df_filtered = filter_df(self.price_df, current_date, days_back)
            volume_df_filtered = filter_df(self.volume_df, current_date, days_back)

            opportunity = self.review_pair_opportunity(
                pair, current_date, price_df_filtered, volume_df_filtered
            )
            if opportunity:
                daily_opportunities.append(opportunity)

        return daily_opportunities

    def review_pair_opportunity(
        self, pair, current_date, price_df_filtered, volume_df_filtered
    ):
        result = self.get_cointegration_and_spread_info(
            pair, price_df_filtered, current_date
        )
        if result is None:
            return
        p_value, hedge_ratio, todays_spread_data, currency_fees = result

        # Attempt to close an existing trade if any
        self.attempt_closing_trade(
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

        return opportunity

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
        open_event = self.portfolio_manager.get_open_trades(pair)
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
            self.portfolio_manager.on_closing_trade(pair, profit)
            self.cumulative_profit += profit
            open_event["hedge_ratio"] = hedge_ratio
            open_event["spread_data"] = todays_spread_data
            self.trade_results.append(
                {
                    "pair": pair,
                    "open_event": open_event,
                    "close_event": close_event,
                    "profit": profit,
                }
            )

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
        if self.portfolio_manager.get_open_trades(pair) is not None:
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

    def apply_ml_decision(self, opportunities):
        """Use the ML model to decide which of the daily opportunities to actually execute."""
        # Extract features from opportunities for ML model
        # This might vary depending on your ML model and feature engineering
        feature_matrix = []
        for opp in opportunities:
            oe = opp["open_event"]
            # Example: Construct a feature vector from event data
            features = [
                oe["p_value"],
                oe["spread_data"]["spread_deviation"],
                oe["hedge_ratio"],
                1 if oe["direction"] == "long" else 0,
                oe["avg_price_ratio"],
                oe["expected_profit"],
                oe["volume_ratio"],
                oe["volatility_ratio"],
            ]
            feature_matrix.append(features)

        predictions = self.ml_model.predict(feature_matrix)

        selected = []
        for opp, pred in zip(opportunities, predictions):
            if pred == 1:  # If model recommends to trade
                selected.append(opp)

        return selected
