import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest


ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from helpers.convergence import ConvergenceForecaster  # noqa: E402


def _get_forecast_line(ax):
    for line in ax.lines:
        if line.get_label() == "Forecast spread path":
            return line
    pytest.fail("Forecast spread path line was not rendered")


def test_plot_forecast_aligns_period_index():
    period_index = pd.period_range("2024-01-01", periods=48, freq="H")
    spread = pd.Series(np.sin(np.linspace(0, np.pi, len(period_index))), index=period_index)

    forecaster = ConvergenceForecaster(rolling_window=5, holding_period=6)
    forecast = forecaster.forecast(spread)

    ax = forecaster.plot_forecast(spread, forecast, show=False)

    try:
        forecast_line = _get_forecast_line(ax)
    finally:
        plt.close(ax.figure)

    xdata = forecast_line.get_xdata()

    assert len(xdata) > 0

    first_point = xdata[0]
    assert not isinstance(first_point, (int, np.integer)), "Forecast plotted on numeric axis"

    if isinstance(first_point, pd.Period):
        last_observed = spread.index[-1]
        assert first_point >= last_observed
    else:
        last_observed = spread.index[-1].to_timestamp()
        assert pd.Timestamp(first_point) >= last_observed
