from cryptopy.src.trading.PortfolioManager import PortfolioManager


class TradingOpportunities:
    @staticmethod
    def check_for_opening_event(
        todays_spread_data,
        p_value,
        parameters,
        avg_price_ratio,
        hedge_ratio,
        current_date,
    ):
        upper_threshold = todays_spread_data["upper_threshold"]
        upper_limit = todays_spread_data["upper_limit"]
        lower_threshold = todays_spread_data["lower_threshold"]
        lower_limit = todays_spread_data["lower_limit"]
        spread = todays_spread_data["spread"]
        spread_mean = todays_spread_data["spread_mean"]

        if hedge_ratio < 0 and parameters["hedge_ratio_positive"]:
            return None

        if avg_price_ratio > parameters["max_coin_price_ratio"] or avg_price_ratio < 0:
            return None

        if p_value < parameters["p_value_open_threshold"]:
            spread_distance = abs(spread - spread_mean)
            if upper_limit > spread > upper_threshold:
                short_stop_loss = (
                    spread + spread_distance * parameters["stop_loss_multiplier"]
                )

                return {
                    "date": current_date,
                    "spread_data": todays_spread_data,
                    "hedge_ratio": hedge_ratio,
                    "direction": "short",
                    "avg_price_ratio": avg_price_ratio,
                    "stop_loss": short_stop_loss,
                    "expected_exit_spread": todays_spread_data.get(
                        "expected_spread_mean_at_exit"
                    ),
                    "expected_exit_spread_value": todays_spread_data.get(
                        "expected_exit_spread"
                    ),
                    "convergence_half_life": todays_spread_data.get(
                        "convergence_half_life"
                    ),
                    "convergence_confidence": todays_spread_data.get(
                        "convergence_confidence"
                    ),
                    "convergence_decay_factor": todays_spread_data.get(
                        "convergence_decay_factor"
                    ),
                    "convergence_phi": todays_spread_data.get("convergence_phi"),
                    "convergence_intercept": todays_spread_data.get(
                        "convergence_intercept"
                    ),
                    "forecasted_spread_path": todays_spread_data.get(
                        "forecasted_spread_path"
                    ),
                    "forecasted_mean_path": todays_spread_data.get(
                        "forecasted_mean_path"
                    ),
                    "forecast_spread_minus_mean": todays_spread_data.get(
                        "forecast_spread_minus_mean"
                    ),
                }
            elif lower_limit < spread < lower_threshold:
                long_stop_loss = (
                    spread - spread_distance * parameters["stop_loss_multiplier"]
                )
                return {
                    "date": current_date,
                    "spread_data": todays_spread_data,
                    "hedge_ratio": hedge_ratio,
                    "direction": "long",
                    "avg_price_ratio": avg_price_ratio,
                    "stop_loss": long_stop_loss,
                    "expected_exit_spread": todays_spread_data.get(
                        "expected_spread_mean_at_exit"
                    ),
                    "expected_exit_spread_value": todays_spread_data.get(
                        "expected_exit_spread"
                    ),
                    "convergence_half_life": todays_spread_data.get(
                        "convergence_half_life"
                    ),
                    "convergence_confidence": todays_spread_data.get(
                        "convergence_confidence"
                    ),
                    "convergence_decay_factor": todays_spread_data.get(
                        "convergence_decay_factor"
                    ),
                    "convergence_phi": todays_spread_data.get("convergence_phi"),
                    "convergence_intercept": todays_spread_data.get(
                        "convergence_intercept"
                    ),
                    "forecasted_spread_path": todays_spread_data.get(
                        "forecasted_spread_path"
                    ),
                    "forecasted_mean_path": todays_spread_data.get(
                        "forecasted_mean_path"
                    ),
                    "forecast_spread_minus_mean": todays_spread_data.get(
                        "forecast_spread_minus_mean"
                    ),
                }
        return None

    @staticmethod
    def check_for_closing_event(
        todays_data,
        p_value,
        parameters,
        open_event,
        hedge_ratio,
    ):
        current_date = todays_data["date"]
        spread = todays_data["spread"]
        spread_mean = todays_data["spread_mean"]

        if p_value > parameters["p_value_close_threshold"]:
            return {
                "date": current_date,
                "spread_data": todays_data,
                "reason": "p_value",
            }

        if hedge_ratio < 0:
            return {
                "date": current_date,
                "spread_data": todays_data,
                "reason": "negative_hedge_ratio",
            }

        if (current_date - open_event["date"]).days > parameters[
            "expiry_days_threshold"
        ]:
            return {
                "date": current_date,
                "spread_data": todays_data,
                "reason": "expired",
            }

        forecast_close = TradingOpportunities._evaluate_forecast_exit(
            todays_data, parameters, open_event
        )
        if forecast_close:
            return forecast_close

        if open_event["direction"] == "short" and spread < spread_mean:
            return {
                "date": current_date,
                "spread_data": todays_data,
                "reason": "crossed_mean",
            }
        elif open_event["direction"] == "long" and spread > spread_mean:
            return {
                "date": current_date,
                "spread_data": todays_data,
                "reason": "crossed_mean",
            }

        if open_event["direction"] == "short" and spread > open_event["stop_loss"]:
            return {
                "date": current_date,
                "spread_data": todays_data,
                "reason": "stop_loss",
            }
        elif open_event["direction"] == "long" and spread < open_event["stop_loss"]:
            return {
                "date": current_date,
                "spread_data": todays_data,
                "reason": "stop_loss",
            }

        if (
            open_event["direction"] == "short"
            and spread_mean > open_event["spread_data"]["spread"]
        ):
            return {
                "date": current_date,
                "spread_data": todays_data,
                "reason": "non-profitable",
            }
        elif (
            open_event["direction"] == "long"
            and spread_mean < open_event["spread_data"]["spread"]
        ):
            return {
                "date": current_date,
                "spread_data": todays_data,
                "reason": "non-profitable",
            }

        return None

    @staticmethod
    def _evaluate_forecast_exit(todays_data, parameters, open_event):
        current_date = todays_data["date"]
        spread = todays_data.get("spread")
        expected_exit = todays_data.get("expected_exit_spread")
        if expected_exit is None:
            expected_exit = todays_data.get("expected_spread_mean_at_exit")
        if expected_exit is None:
            expected_exit = todays_data.get("spread_mean")

        if spread is None or expected_exit is None:
            return None

        position_size = open_event.get("position_size")
        if not position_size:
            return None

        long_position = position_size.get("long_position", {})
        bought_amount = long_position.get("amount")
        if bought_amount is None:
            return None

        hedge_ratio = open_event.get("hedge_ratio", 1)
        if open_event.get("direction") == "short" and hedge_ratio:
            bought_amount /= hedge_ratio

        fee_rate = open_event.get("fee_rate", 0.0) or 0.0
        forecasted_profit = bought_amount * abs(spread - expected_exit) * (1 - fee_rate)

        min_expected_ratio = parameters.get(
            "min_expected_profit_to_hold", parameters.get("min_expected_profit", 0.0)
        )
        trade_amount = open_event.get("trade_amount", 0.0)
        threshold = trade_amount * min_expected_ratio

        confidence_floor = parameters.get("min_convergence_confidence", None)
        confidence = todays_data.get("convergence_confidence")

        if confidence_floor is not None and confidence is not None:
            if confidence < confidence_floor:
                return {
                    "date": current_date,
                    "spread_data": todays_data,
                    "reason": "forecast_confidence_drop",
                    "forecasted_profit": forecasted_profit,
                    "forecasted_exit_spread": expected_exit,
                    "forecasted_confidence": confidence,
                }

        if forecasted_profit < threshold:
            return {
                "date": current_date,
                "spread_data": todays_data,
                "reason": "forecast_degraded",
                "forecasted_profit": forecasted_profit,
                "forecasted_exit_spread": expected_exit,
                "forecasted_confidence": confidence,
            }

        return None
