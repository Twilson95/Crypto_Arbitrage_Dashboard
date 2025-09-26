import importlib.util
import sys
import types
from pathlib import Path

import pandas as pd
import pandas.testing as pdt
import unittest


def _load_module(module_name):
    module_path = Path(__file__).resolve().parents[2] / "src" / "arbitrage" / f"{module_name}.py"
    spec = importlib.util.spec_from_file_location(
        f"cryptopy.src.arbitrage.{module_name}", module_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_stat_module = _load_module("StatisticalArbitrage")
StatisticalArbitrage = _stat_module.StatisticalArbitrage

if "cryptopy" not in sys.modules:
    cryptopy_stub = types.ModuleType("cryptopy")
    cryptopy_stub.StatisticalArbitrage = StatisticalArbitrage
    sys.modules["cryptopy"] = cryptopy_stub
else:
    setattr(sys.modules["cryptopy"], "StatisticalArbitrage", StatisticalArbitrage)

sys.modules.setdefault("cryptopy.src", types.ModuleType("cryptopy.src"))
sys.modules.setdefault("cryptopy.src.arbitrage", types.ModuleType("cryptopy.src.arbitrage"))
setattr(sys.modules["cryptopy"], "src", sys.modules["cryptopy.src"])
setattr(sys.modules["cryptopy.src"], "arbitrage", sys.modules["cryptopy.src.arbitrage"])
sys.modules["cryptopy.src.arbitrage"].StatisticalArbitrage = _stat_module
sys.modules["cryptopy.src.arbitrage.StatisticalArbitrage"] = _stat_module

_coint_module = _load_module("CointegrationCalculator")
sys.modules["cryptopy.src.arbitrage.CointegrationCalculator"] = _coint_module
sys.modules["cryptopy.src.arbitrage"].CointegrationCalculator = _coint_module
CointegrationCalculator = _coint_module.CointegrationCalculator
setattr(sys.modules["cryptopy"], "CointegrationCalculator", CointegrationCalculator)


class TestStatisticalArbitrageIteration(unittest.TestCase):
    def setUp(self):
        self.pairs = ("BTC/USD", "ETH/USD")
        self.entry_time = pd.Timestamp("2023-01-01 00:00:00")
        self.exit_time = pd.Timestamp("2023-01-02 00:00:00")
        self.price_df = pd.DataFrame(
            {
                "BTC/USD": [100.0, 110.0],
                "ETH/USD": [200.0, 210.0],
            },
            index=[self.entry_time, self.exit_time],
        )
        self.currency_fees = {
            "BTC/USD": {"taker": 0.0},
            "ETH/USD": {"taker": 0.0},
        }

    def test_negative_hedge_ratio_long_entry_flips_legs(self):
        result = StatisticalArbitrage.statistical_arbitrage_iteration(
            entry=(self.entry_time, 0.0, "long"),
            exit=(self.exit_time, 0.0),
            pairs=self.pairs,
            currency_fees=self.currency_fees,
            price_df=self.price_df,
            usd_start=100.0,
            hedge_ratio=-2.0,
            exchange="test",
        )

        buy_instruction, short_instruction = result["instructions"][:2]

        self.assertEqual(buy_instruction["instruction"], "buy")
        self.assertEqual(buy_instruction["to_currency"], "ETH")
        self.assertEqual(short_instruction["instruction"], "sell short")
        self.assertEqual(short_instruction["from_currency"], "BTC")
        self.assertAlmostEqual(
            short_instruction["from_amount"],
            buy_instruction["to_amount"] * 2.0,
        )

    def test_negative_hedge_ratio_short_entry_flips_legs(self):
        result = StatisticalArbitrage.statistical_arbitrage_iteration(
            entry=(self.entry_time, 0.0, "short"),
            exit=(self.exit_time, 0.0),
            pairs=self.pairs,
            currency_fees=self.currency_fees,
            price_df=self.price_df,
            usd_start=100.0,
            hedge_ratio=-2.0,
            exchange="test",
        )

        buy_instruction, short_instruction = result["instructions"][:2]

        self.assertEqual(buy_instruction["instruction"], "buy")
        self.assertEqual(buy_instruction["to_currency"], "BTC")
        self.assertEqual(short_instruction["instruction"], "sell short")
        self.assertEqual(short_instruction["from_currency"], "ETH")
        self.assertAlmostEqual(
            short_instruction["from_amount"],
            buy_instruction["to_amount"] * 0.5,
        )


class TestCointegrationCalculator(unittest.TestCase):
    def test_calculate_spread_with_negative_cached_ratio(self):
        index = pd.date_range("2023-01-01", periods=3, freq="D")
        df = pd.DataFrame(
            {
                "BTC/USD": [100.0, 101.0, 102.0],
                "ETH/USD": [200.0, 198.0, 197.0],
            },
            index=index,
        )

        spread, hedge_ratio = CointegrationCalculator.calculate_spread(
            df, ("BTC/USD", "ETH/USD"), hedge_ratio=-2.0
        )

        expected_spread = df["ETH/USD"] - 2.0 * df["BTC/USD"]

        self.assertEqual(hedge_ratio, -2.0)
        pdt.assert_series_equal(spread, expected_spread)


if __name__ == "__main__":
    unittest.main()
