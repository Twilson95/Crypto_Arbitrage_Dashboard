import sys
from pathlib import Path

import numpy as np
import pandas as pd
import unittest


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


from cryptopy.scripts.simulations import simulation_helpers as sim_helpers  # noqa: E402


class TestSimulationHelperForecastFiltering(unittest.TestCase):
    def setUp(self):
        self.parameters = {
            "rolling_window": 3,
            "spread_threshold": 1.0,
            "spread_limit": 3.0,
            "expected_holding_days": 2,
            "convergence_lookback": 6,
        }
        values = [
            0.1,
            0.2,
            0.15,
            2.5,
            2.6,
            2.8,
            -2.5,
            -2.7,
            -2.9,
            0.05,
        ]
        index = pd.date_range("2023-01-01", periods=len(values), freq="D")
        self.spread = pd.Series(values, index=index)
        self.low_deviation_day = pd.Timestamp("2023-01-05")
        self.high_deviation_day = pd.Timestamp("2023-01-06")
        self.metrics = sim_helpers.compute_spread_metrics(
            self.parameters,
            self.spread,
            current_date=self.high_deviation_day,
        )

    def test_expected_exit_only_for_trade_candidates(self):
        expected_exit_spread = self.metrics["expected_spread_at_exit"]
        self.assertTrue(np.isnan(expected_exit_spread.loc[self.low_deviation_day]))
        self.assertFalse(np.isnan(expected_exit_spread.loc[self.high_deviation_day]))

    def test_get_todays_spread_data_filters_forecast(self):
        low_day_data = sim_helpers.get_todays_spread_data(
            self.parameters,
            self.spread,
            self.low_deviation_day,
            spread_metrics=self.metrics,
        )
        high_day_data = sim_helpers.get_todays_spread_data(
            self.parameters,
            self.spread,
            self.high_deviation_day,
            spread_metrics=self.metrics,
        )

        self.assertFalse(low_day_data["trade_considered"])
        self.assertIsNone(low_day_data["expected_exit_spread"])
        self.assertIsNone(low_day_data["forecasted_spread_path"])
        self.assertIsNone(low_day_data["forecast_spread_minus_mean"])

        self.assertTrue(high_day_data["trade_considered"])
        self.assertIsNotNone(high_day_data["expected_exit_spread"])
        self.assertIsNotNone(high_day_data["forecasted_spread_path"])
        self.assertIsNotNone(high_day_data["forecast_spread_minus_mean"])

    def test_forecast_not_computed_without_candidates(self):
        parameters = dict(self.parameters)
        parameters["spread_threshold"] = 5.0  # ensure no candidates

        flat_spread = pd.Series(
            [0.1, 0.2, 0.05, -0.1, 0.0],
            index=pd.date_range("2023-02-01", periods=5, freq="D"),
        )

        metrics = sim_helpers.compute_spread_metrics(
            parameters,
            flat_spread,
            current_date=pd.Timestamp("2023-02-05"),
        )

        expected_exit = metrics["expected_spread_at_exit"]
        self.assertTrue(expected_exit.isna().all())
        self.assertTrue(metrics["forecasted_spread_path"].empty)
        self.assertTrue(metrics["forecasted_mean_path"].empty)
        self.assertIsNone(metrics["convergence_decay_factor"])
        self.assertIsNone(metrics["convergence_half_life"])
        self.assertIsNone(metrics["convergence_phi"])
        self.assertIsNone(metrics["convergence_intercept"])

    def test_forecast_retained_for_open_trade_without_threshold(self):
        metrics = sim_helpers.compute_spread_metrics(
            self.parameters,
            self.spread,
            current_date=self.low_deviation_day,
            trade_open=True,
        )

        low_day_data = sim_helpers.get_todays_spread_data(
            self.parameters,
            self.spread,
            self.low_deviation_day,
            spread_metrics=metrics,
            trade_open=True,
        )

        self.assertFalse(low_day_data["trade_considered"])
        self.assertIsNotNone(low_day_data["expected_exit_spread"])
        self.assertIsNotNone(low_day_data["forecasted_spread_path"])
        self.assertIsNotNone(low_day_data["forecast_spread_minus_mean"])


if __name__ == "__main__":
    unittest.main()
