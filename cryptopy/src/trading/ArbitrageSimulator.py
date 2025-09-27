import time

import math

import pandas as pd

from cryptopy import CointegrationCalculator, TradingOpportunities, RiverPredictor
from cryptopy.src.trading.cache_utils import (
    PairAnalyticsCache,
)

from cryptopy.scripts.simulations.simulation_helpers import (
    get_trade_profit,
    filter_df,
    get_avg_price_difference,
    calculate_expected_profit,
    get_bought_and_sold_amounts,
    filter_list,
    compute_spread_metrics,
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
        pair_analytics_cache: Optional[PairAnalyticsCache] = None,
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
        self._spread_metrics_cache = {}
        if pair_analytics_cache is not None:
            self._pair_analytics_cache = pair_analytics_cache
        elif self.parameters.get("use_pair_analytics_cache", True):
            cache_dir = self.parameters.get("analytics_cache_dir")
            self._pair_analytics_cache = PairAnalyticsCache(cache_dir)
        else:
            self._pair_analytics_cache = None
        self._trend_parameters = self.parameters.get("trend_parameters", {})

        short_window = self._trend_parameters.get("short_window")
        long_window = self._trend_parameters.get("long_window")

        if short_window and long_window:
            self._short_ma = self.price_df.rolling(window=short_window).mean()
            self._long_ma = self.price_df.rolling(window=long_window).mean()
        else:
            self._short_ma = None
            self._long_ma = None

    def _get_expected_holding_days(self):
        return self.parameters.get("expected_holding_days", 0.0)

    def _get_borrow_rate_per_day(self):
        return self.parameters.get("borrow_rate_per_day", 0.0)

    def _get_price_at(self, symbol, current_date):
        if symbol is None:
            return None

        try:
            price = self.price_df.at[current_date, symbol]
        except KeyError:
            return None

        try:
            return float(price)
        except (TypeError, ValueError):
            return None

    def _compute_short_position_metrics(self, position_size, current_date):
        short_symbol = position_size["short_position"].get("coin")
        short_price = self._get_price_at(short_symbol, current_date)

        if short_price is None:
            return 0.0, None

        short_amount = position_size["short_position"].get("amount", 0.0)
        short_notional = short_amount * short_price
        return short_notional, short_price

    def precalculate_pair_analytics(self):
        if self._pair_analytics_cache is None:
            return

        days_back = self.parameters.get("days_back", 0)
        PairAnalyticsCache.precalculate_pair_analytics(
            self.price_df, self.pair_combinations, days_back, self._pair_analytics_cache
        )

    def run_simulation(self):
        days_back = self.parameters["days_back"]

        for current_date in self.price_df.index[days_back:]:
            start_time = time.time()
            self.simulate_day(current_date, days_back)
            cumulative_profit = self.portfolio_manager.get_cumulative_profit()
            end_time = time.time()

            print(
                f"\n-----------------------------------------------------\n"
                f"Date: {current_date} \n"
                f"Open Trades: {self.portfolio_manager.traded_pairs} \n"
                f"Total Profit {cumulative_profit:.2f} \n"
                f"Time: {end_time - start_time:.2f}secs"
            )

        all_trades = self.portfolio_manager.get_all_trade_events()
        cumulative_profit = self.portfolio_manager.get_cumulative_profit()
        return all_trades, cumulative_profit

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

        price_field = "BTC/USD"
        market_trend = self.get_current_trend(price_field, current_date)

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

                coin_1_trend = self.get_current_trend(pair[0], current_date)
                coin_2_trend = self.get_current_trend(pair[1], current_date)

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

    def get_current_trend(self, price_field, current_date):
        if (
            self._short_ma is None
            or self._long_ma is None
            or price_field not in self.price_df.columns
        ):
            return None

        try:
            short_ma_value = self._short_ma.at[current_date, price_field]
            long_ma_value = self._long_ma.at[current_date, price_field]
        except KeyError:
            return None

        change_threshold = self._trend_parameters.get("change_threshold")
        if change_threshold is None:
            return None

        sma_difference = abs(short_ma_value - long_ma_value)
        sma_percentage_difference = sma_difference / long_ma_value

        if sma_percentage_difference < change_threshold:
            trend = "Flat"
        elif short_ma_value > long_ma_value:
            trend = "Uptrend"
        else:
            trend = "Downtrend"
        return trend

    def get_cointegration_and_spread_info(self, pair, price_df_filtered, current_date):
        currency_fees = {pair[0]: {"taker": 0.002}, pair[1]: {"taker": 0.002}}

        cached_analytics = None
        if self._pair_analytics_cache is not None:
            cached_analytics = self._pair_analytics_cache.ensure(
                pair, current_date, price_df_filtered
            )
            if cached_analytics is None:
                return None
            p_value = cached_analytics["p_value"]
            base_spread = cached_analytics["spread"]
            base_hedge_ratio = cached_analytics["hedge_ratio"]
        else:
            coint_stat, p_value, _ = CointegrationCalculator.test_cointegration(
                price_df_filtered, pair
            )
            if p_value is None:
                return None
            base_spread, base_hedge_ratio = CointegrationCalculator.calculate_spread(
                price_df_filtered, pair
            )

        open_event = self.portfolio_manager.get_open_trades(pair)
        if open_event:
            event_hedge_ratio = open_event.get("hedge_ratio")
            spread, hedge_ratio = CointegrationCalculator.calculate_spread(
                price_df_filtered, pair, event_hedge_ratio
            )
        else:
            spread = base_spread
            hedge_ratio = base_hedge_ratio

        spread_metrics = self._get_cached_spread_metrics(pair, current_date, spread)
        todays_spread_data = self.get_todays_spread_data(
            spread, current_date, spread_metrics
        )

        return p_value, hedge_ratio, todays_spread_data, currency_fees

    def _get_cached_spread_metrics(self, pair, current_date, spread):
        pair_key = tuple(pair)
        cache_key = (pair_key, current_date)
        cached_metrics = self._spread_metrics_cache.get(cache_key)
        if cached_metrics is not None:
            return cached_metrics

        metrics = compute_spread_metrics(self.parameters, spread)

        keys_to_remove = [
            key
            for key in list(self._spread_metrics_cache.keys())
            if key[0] == pair_key and key[1] != current_date
        ]
        for key in keys_to_remove:
            del self._spread_metrics_cache[key]

        self._spread_metrics_cache[cache_key] = metrics
        return metrics

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
            if close_event.get("reason") in {
                "forecast_degraded",
                "forecast_confidence_drop",
            }:
                forecasted_profit = close_event.get("forecasted_profit")
                forecast_exit = close_event.get("forecasted_exit_spread")
                forecast_confidence = close_event.get("forecasted_confidence")
                print(
                    f"{pair} closing early due to {close_event['reason']}: "
                    f"forecasted profit {forecasted_profit}, "
                    f"forecasted exit {forecast_exit}, confidence {forecast_confidence}"
                )
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
        short_notional, short_entry_price = self._compute_short_position_metrics(
            position_size, current_date
        )

        borrow_rate_per_day = self._get_borrow_rate_per_day()
        expected_holding_days = self._get_expected_holding_days()

        expected_profit = calculate_expected_profit(
            pair,
            new_open_event,
            position_size,
            currency_fees,
            borrow_rate_per_day=borrow_rate_per_day,
            expected_holding_days=expected_holding_days,
            short_notional=short_notional,
        )

        expected_exit_spread = new_open_event.get("expected_exit_spread_value")
        if expected_exit_spread is None:
            expected_exit_spread = new_open_event.get("expected_exit_spread")
        half_life = new_open_event.get("convergence_half_life")
        confidence = new_open_event.get("convergence_confidence")
        decay_factor = new_open_event.get("convergence_decay_factor")
        phi = new_open_event.get("convergence_phi")
        intercept = new_open_event.get("convergence_intercept")
        forecast_diff = new_open_event.get("forecast_spread_minus_mean")

        exit_str = "N/A"
        if expected_exit_spread is not None:
            try:
                exit_value = float(expected_exit_spread)
                if not math.isnan(exit_value):
                    exit_str = f"{exit_value:.6f}"
            except (TypeError, ValueError):
                pass

        half_life_str = "N/A"
        if half_life is not None:
            try:
                half_life_value = float(half_life)
                if not math.isnan(half_life_value):
                    half_life_str = f"{half_life_value:.2f}"
            except (TypeError, ValueError):
                pass

        confidence_str = "N/A"
        if confidence is not None:
            try:
                confidence_value = float(confidence)
                if not math.isnan(confidence_value):
                    confidence_str = f"{confidence_value:.2f}"
            except (TypeError, ValueError):
                pass

        decay_str = "N/A"
        if decay_factor is not None:
            try:
                decay_value = float(decay_factor)
                if not math.isnan(decay_value):
                    decay_str = f"{decay_value:.4f}"
            except (TypeError, ValueError):
                pass

        phi_str = "N/A"
        if phi is not None:
            try:
                phi_value = float(phi)
                if not math.isnan(phi_value):
                    phi_str = f"{phi_value:.4f}"
            except (TypeError, ValueError):
                pass

        intercept_str = "N/A"
        if intercept is not None:
            try:
                intercept_value = float(intercept)
                if not math.isnan(intercept_value):
                    intercept_str = f"{intercept_value:.6f}"
            except (TypeError, ValueError):
                pass

        print(
            f"{pair} convergence forecast -> expected exit spread: {exit_str}, "
            f"half-life: {half_life_str}, confidence: {confidence_str}, "
            f"decay factor: {decay_str}, phi: {phi_str}, intercept: {intercept_str}"
        )
        if forecast_diff:
            print(f"{pair} forecast spread minus mean: {forecast_diff}")
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
                "expected_holding_days": expected_holding_days,
                "borrow_rate_per_day": borrow_rate_per_day,
                "short_notional": short_notional,
                "short_entry_price": short_entry_price,
                "fee_rate": (
                    currency_fees[pair[0]]["taker"] * 2
                    + currency_fees[pair[1]]["taker"] * 2
                ),
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

    def get_todays_spread_data(self, spread, current_date, spread_metrics=None):
        if spread_metrics is None:
            spread_metrics = compute_spread_metrics(self.parameters, spread)

        todays_spread = filter_list(spread, current_date)
        todays_spread_mean = filter_list(spread_metrics["spread_mean"], current_date)
        todays_spread_std = filter_list(spread_metrics["spread_std"], current_date)
        expected_exit_mean = filter_list(
            spread_metrics["expected_mean_at_exit"], current_date
        )
        expected_exit_spread = filter_list(
            spread_metrics["expected_spread_at_exit"], current_date
        )
        if expected_exit_mean is None:
            expected_exit_mean = todays_spread_mean
        forecast_spread_path = spread_metrics.get("forecasted_spread_path")
        forecast_mean_path = spread_metrics.get("forecasted_mean_path")
        todays_spread_forecast = None
        todays_mean_forecast = None
        if (
            isinstance(forecast_spread_path, pd.DataFrame)
            and current_date in forecast_spread_path.index
        ):
            todays_spread_forecast = (
                forecast_spread_path.loc[current_date].dropna().to_dict()
            )
        if (
            isinstance(forecast_mean_path, pd.DataFrame)
            and current_date in forecast_mean_path.index
        ):
            todays_mean_forecast = (
                forecast_mean_path.loc[current_date].dropna().to_dict()
            )
        forecast_diff = None
        if todays_spread_forecast and todays_mean_forecast:
            forecast_diff = {
                key: todays_spread_forecast.get(key)
                - todays_mean_forecast.get(key, math.nan)
                for key in todays_spread_forecast
            }
        return {
            "date": current_date,
            "spread": todays_spread,
            "spread_mean": todays_spread_mean,
            "spread_std": todays_spread_std,
            "upper_threshold": filter_list(
                spread_metrics["upper_threshold"], current_date
            ),
            "upper_limit": filter_list(spread_metrics["upper_limit"], current_date),
            "lower_threshold": filter_list(
                spread_metrics["lower_threshold"], current_date
            ),
            "lower_limit": filter_list(spread_metrics["lower_limit"], current_date),
            "expected_spread_mean_at_exit": expected_exit_mean,
            "expected_exit_spread": expected_exit_spread,
            "convergence_half_life": spread_metrics.get("convergence_half_life"),
            "convergence_confidence": spread_metrics.get("convergence_confidence"),
            "convergence_decay_factor": spread_metrics.get("convergence_decay_factor"),
            "convergence_phi": spread_metrics.get("convergence_phi"),
            "convergence_intercept": spread_metrics.get("convergence_intercept"),
            "forecasted_spread_path": todays_spread_forecast,
            "forecasted_mean_path": todays_mean_forecast,
            "forecast_spread_minus_mean": forecast_diff,
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
