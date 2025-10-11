from datetime import date

from cryptopy.src.trading.TradingOpportunities import TradingOpportunities


def base_close_parameters():
    return {
        "p_value_close_threshold": 0.5,
        "expiry_days_threshold": 10,
        "stop_loss_triggered_immediately": True,
    }


def test_apply_stop_loss_cap_limits_short_losses():
    todays_data = {"spread": 12.0}
    open_event = {"direction": "short", "stop_loss": 11.0}

    TradingOpportunities._apply_stop_loss_cap(todays_data, open_event)

    assert todays_data["spread"] == 11.0


def test_apply_stop_loss_cap_preserves_better_short_exit():
    todays_data = {"spread": 10.5}
    open_event = {"direction": "short", "stop_loss": 11.0}

    TradingOpportunities._apply_stop_loss_cap(todays_data, open_event)

    assert todays_data["spread"] == 10.5


def test_apply_stop_loss_cap_limits_long_losses():
    todays_data = {"spread": -2.0}
    open_event = {"direction": "long", "stop_loss": -1.5}

    TradingOpportunities._apply_stop_loss_cap(todays_data, open_event)

    assert todays_data["spread"] == -1.5


def test_apply_stop_loss_cap_prefers_spread_override():
    todays_data = {"spread": 14.0}
    open_event = {
        "direction": "short",
        "stop_loss": 12.5,
        "spread_data": {"stop_loss_spread": 11.25},
    }

    TradingOpportunities._apply_stop_loss_cap(todays_data, open_event)

    assert todays_data["spread"] == 11.25


def test_apply_stop_loss_cap_handles_dict_stop_loss():
    todays_data = {"spread": -2.5}
    open_event = {"direction": "long", "stop_loss": {"spread": -1.75}}

    TradingOpportunities._apply_stop_loss_cap(todays_data, open_event)

    assert todays_data["spread"] == -1.75


def test_stop_loss_close_uses_override_value():
    todays_data = {
        "date": date(2024, 1, 2),
        "spread": 15.0,
        "spread_mean": 9.0,
    }
    open_event = {
        "date": date(2024, 1, 1),
        "direction": "short",
        "stop_loss": 12.5,
        "spread_data": {"spread": 9.5, "stop_loss_spread": 11.0},
    }

    event = TradingOpportunities.check_for_closing_event(
        todays_data,
        p_value=0.0,
        parameters=base_close_parameters(),
        open_event=open_event,
        hedge_ratio=1.0,
    )

    assert event is not None
    assert event["reason"] == "stop_loss"
    assert todays_data["spread"] == 11.0


def test_stop_loss_close_uses_numeric_value_when_no_override():
    todays_data = {
        "date": date(2024, 1, 2),
        "spread": -4.5,
        "spread_mean": -2.0,
    }
    open_event = {
        "date": date(2024, 1, 1),
        "direction": "long",
        "stop_loss": -3.5,
        "spread_data": {"spread": -2.5},
    }

    event = TradingOpportunities.check_for_closing_event(
        todays_data,
        p_value=0.0,
        parameters=base_close_parameters(),
        open_event=open_event,
        hedge_ratio=1.0,
    )

    assert event is not None
    assert event["reason"] == "stop_loss"
    assert todays_data["spread"] == -3.5
