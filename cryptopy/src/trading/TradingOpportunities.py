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
        lower_threshold = todays_spread_data["lower_threshold"]
        spread = todays_spread_data["spread"]
        spread_mean = todays_spread_data["spread_mean"]

        if hedge_ratio < 0 and parameters["hedge_ratio_positive"]:
            return None

        if avg_price_ratio > parameters["max_coin_price_ratio"] or avg_price_ratio < 0:
            return None

        if p_value < parameters["p_value_open_threshold"]:
            spread_distance = abs(spread - spread_mean)
            if spread > upper_threshold:
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
                }
            elif spread < lower_threshold:
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
            return {"date": current_date, "spread_data": spread, "reason": "p_value"}

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
