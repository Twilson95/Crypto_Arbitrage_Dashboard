import importlib.util
import sys
import types
from pathlib import Path

import pandas as pd
import pandas.testing as pdt
import unittest


def _load_module(module_name):
    module_path = (
        Path(__file__).resolve().parents[2] / "src" / "arbitrage" / f"{module_name}.py"
    )
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
sys.modules.setdefault(
    "cryptopy.src.arbitrage", types.ModuleType("cryptopy.src.arbitrage")
)
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

    def test_long_iteration_calculates_expected_profits_and_fees(self):
        entry_time = pd.Timestamp("2023-01-01 00:00:00")
        exit_time = pd.Timestamp("2023-01-02 00:00:00")
        price_df = pd.DataFrame(
            {
                "BTC/USD": [100.0, 110.0],
                "ETH/USD": [50.0, 45.0],
            },
            index=[entry_time, exit_time],
        )

        usd_start = 100.0
        hedge_ratio = 2.0
        btc_fee = 0.01
        eth_fee = 0.02

        result = StatisticalArbitrage.statistical_arbitrage_iteration(
            entry=(entry_time, 0.0632, "long"),
            exit=(exit_time, 0.06633),
            pairs=("BTC/USD", "ETH/USD"),
            currency_fees={
                "BTC/USD": {"taker": btc_fee},
                "ETH/USD": {"taker": eth_fee},
            },
            price_df=price_df,
            usd_start=usd_start,
            hedge_ratio=hedge_ratio,
            exchange="test",
        )

        amount_buy_btc = usd_start / price_df.loc[entry_time, "BTC/USD"]
        fees_buy_btc = amount_buy_btc * btc_fee
        net_buy_btc = amount_buy_btc - fees_buy_btc
        fees_sell_btc = net_buy_btc * btc_fee
        usd_after_sell_btc = (net_buy_btc - fees_sell_btc) * price_df.loc[exit_time, "BTC/USD"]
        coin1_profit = usd_after_sell_btc - usd_start
        coin1_fees_usd = (
            fees_buy_btc * price_df.loc[entry_time, "BTC/USD"]
            + fees_sell_btc * price_df.loc[exit_time, "BTC/USD"]
        )

        amount_sell_eth = amount_buy_btc * hedge_ratio
        fees_sell_eth = amount_sell_eth * eth_fee
        usd_after_sell_short = amount_sell_eth * price_df.loc[entry_time, "ETH/USD"]
        net_sell_eth = amount_sell_eth - fees_sell_eth
        fees_buy_cover_eth = net_sell_eth * eth_fee
        usd_after_cover_eth = (
            net_sell_eth - fees_buy_cover_eth
        ) * price_df.loc[exit_time, "ETH/USD"]
        coin2_profit = usd_after_sell_short - usd_after_cover_eth
        coin2_fees_usd = (
            fees_sell_eth * price_df.loc[entry_time, "ETH/USD"]
            + fees_buy_cover_eth * price_df.loc[exit_time, "ETH/USD"]
        )

        expected_total_profit = coin1_profit + coin2_profit
        expected_total_fees = coin1_fees_usd + coin2_fees_usd

        summary = result["summary_header"]
        waterfall = result["waterfall_data"]

        self.assertAlmostEqual(summary["total_profit"], expected_total_profit)
        self.assertAlmostEqual(waterfall["Coin-1 Fees"], -coin1_fees_usd)
        self.assertAlmostEqual(waterfall["Coin-2 Fees"], -coin2_fees_usd)
        self.assertAlmostEqual(
            waterfall["Potential Profit"],
            expected_total_profit + expected_total_fees,
        )

    def test_short_iteration_accounts_for_borrow_costs(self):
        entry_time = pd.Timestamp("2023-01-01 00:00:00")
        exit_time = pd.Timestamp("2023-01-03 00:00:00")
        price_df = pd.DataFrame(
            {
                "BTC/USD": [100.0, 90.0],
                "ETH/USD": [50.0, 55.0],
            },
            index=[entry_time, exit_time],
        )

        usd_start = 100.0
        borrow_rate = 0.05
        hedge_ratio = 2.0

        result = StatisticalArbitrage.statistical_arbitrage_iteration(
            entry=(entry_time, 0.0632, "short"),
            exit=(exit_time, 0.06633),
            pairs=("BTC/USD", "ETH/USD"),
            currency_fees={
                "BTC/USD": {"taker": 0.0},
                "ETH/USD": {"taker": 0.0},
            },
            price_df=price_df,
            usd_start=usd_start,
            hedge_ratio=hedge_ratio,
            exchange="test",
            borrow_rate_per_day=borrow_rate,
            expected_holding_days=2.0,
        )

        amount_buy_eth = usd_start / price_df.loc[entry_time, "ETH/USD"]
        hedge_ratio_short = 1 / hedge_ratio
        short_notional = (
            amount_buy_eth * hedge_ratio_short * price_df.loc[entry_time, "BTC/USD"]
        )
        expected_borrow_cost = borrow_rate * 2.0 * short_notional
        eth_profit = (
            amount_buy_eth * price_df.loc[exit_time, "ETH/USD"]
            - usd_start
        )
        btc_profit = (
            amount_buy_eth * hedge_ratio_short * price_df.loc[entry_time, "BTC/USD"]
            - amount_buy_eth * hedge_ratio_short * price_df.loc[exit_time, "BTC/USD"]
        )
        expected_total_profit = eth_profit + btc_profit - expected_borrow_cost

        summary = result["summary_header"]
        waterfall = result["waterfall_data"]

        self.assertAlmostEqual(summary["borrow_costs"], expected_borrow_cost)
        self.assertAlmostEqual(result["borrow_costs"], expected_borrow_cost)
        self.assertAlmostEqual(result["actual_holding_days"], 2.0)
        self.assertAlmostEqual(waterfall["Borrow/Funding Costs"], -expected_borrow_cost)
        self.assertAlmostEqual(summary["total_profit"], expected_total_profit)

    def test_short_iteration_applies_fees_to_correct_legs(self):
        entry_time = pd.Timestamp("2023-01-01 00:00:00")
        exit_time = pd.Timestamp("2023-01-02 00:00:00")
        price_df = pd.DataFrame(
            {
                "BTC/USD": [100.0, 95.0],
                "ETH/USD": [50.0, 55.0],
            },
            index=[entry_time, exit_time],
        )

        btc_fee = 0.03
        eth_fee = 0.01
        usd_start = 100.0
        hedge_ratio = 2.0

        result = StatisticalArbitrage.statistical_arbitrage_iteration(
            entry=(entry_time, 0.0632, "short"),
            exit=(exit_time, 0.06633),
            pairs=("BTC/USD", "ETH/USD"),
            currency_fees={
                "BTC/USD": {"taker": btc_fee},
                "ETH/USD": {"taker": eth_fee},
            },
            price_df=price_df,
            usd_start=usd_start,
            hedge_ratio=hedge_ratio,
            exchange="test",
        )

        amount_buy_eth = usd_start / price_df.loc[entry_time, "ETH/USD"]
        fees_buy_eth = amount_buy_eth * eth_fee
        net_buy_eth = amount_buy_eth - fees_buy_eth
        fees_sell_eth = net_buy_eth * eth_fee
        usd_after_sell_eth = (net_buy_eth - fees_sell_eth) * price_df.loc[
            exit_time, "ETH/USD"
        ]
        eth_profit = usd_after_sell_eth - usd_start
        eth_fees_usd = (
            fees_buy_eth * price_df.loc[entry_time, "ETH/USD"]
            + fees_sell_eth * price_df.loc[exit_time, "ETH/USD"]
        )

        hedge_ratio_short = 1 / hedge_ratio
        amount_sell_btc = amount_buy_eth * hedge_ratio_short
        fees_sell_btc = amount_sell_btc * btc_fee
        usd_after_sell_short = amount_sell_btc * price_df.loc[entry_time, "BTC/USD"]
        net_sell_btc = amount_sell_btc - fees_sell_btc
        fees_buy_cover_btc = net_sell_btc * btc_fee
        usd_after_cover_btc = (
            net_sell_btc - fees_buy_cover_btc
        ) * price_df.loc[exit_time, "BTC/USD"]
        btc_profit = usd_after_sell_short - usd_after_cover_btc
        btc_fees_usd = (
            fees_sell_btc * price_df.loc[entry_time, "BTC/USD"]
            + fees_buy_cover_btc * price_df.loc[exit_time, "BTC/USD"]
        )

        summary = result["summary_header"]
        waterfall = result["waterfall_data"]

        self.assertAlmostEqual(summary["total_profit"], eth_profit + btc_profit)
        self.assertAlmostEqual(waterfall["Coin-1 Fees"], -eth_fees_usd)
        self.assertAlmostEqual(waterfall["Coin-2 Fees"], -btc_fees_usd)
        self.assertEqual(summary["coins_used"], ["ETH", "BTC"])

    def test_short_iteration_simple_inputs_produce_expected_profit_and_fees(self):
        entry_time = pd.Timestamp("2023-01-01 00:00:00")
        exit_time = pd.Timestamp("2023-01-02 00:00:00")
        price_df = pd.DataFrame(
            {
                "BTC/USD": [200.0, 180.0],
                "ETH/USD": [100.0, 120.0],
            },
            index=[entry_time, exit_time],
        )

        usd_start = 100.0
        taker_fee = 0.10
        hedge_ratio = 1.0
        borrow_rate = 0.05

        result = StatisticalArbitrage.statistical_arbitrage_iteration(
            entry=(entry_time, 0.5, "short"),
            exit=(exit_time, 0.25),
            pairs=("BTC/USD", "ETH/USD"),
            currency_fees={
                "BTC/USD": {"taker": taker_fee},
                "ETH/USD": {"taker": taker_fee},
            },
            price_df=price_df,
            usd_start=usd_start,
            hedge_ratio=hedge_ratio,
            exchange="unit-test",
            borrow_rate_per_day=borrow_rate,
            expected_holding_days=1.0,
        )

        eth_entry_price = price_df.loc[entry_time, "ETH/USD"]
        eth_exit_price = price_df.loc[exit_time, "ETH/USD"]
        btc_entry_price = price_df.loc[entry_time, "BTC/USD"]
        btc_exit_price = price_df.loc[exit_time, "BTC/USD"]

        amount_eth_bought = usd_start / eth_entry_price
        eth_entry_fee = amount_eth_bought * taker_fee
        net_eth_after_entry = amount_eth_bought - eth_entry_fee
        eth_exit_fee = net_eth_after_entry * taker_fee
        usd_after_eth_exit = (net_eth_after_entry - eth_exit_fee) * eth_exit_price
        eth_profit = usd_after_eth_exit - usd_start
        eth_fees_usd = eth_entry_fee * eth_entry_price + eth_exit_fee * eth_exit_price

        amount_btc_shorted = amount_eth_bought * hedge_ratio
        btc_entry_fee = amount_btc_shorted * taker_fee
        net_btc_after_entry = amount_btc_shorted - btc_entry_fee
        btc_exit_fee = net_btc_after_entry * taker_fee
        usd_received_from_short = amount_btc_shorted * btc_entry_price
        usd_spent_covering_short = (
            net_btc_after_entry - btc_exit_fee
        ) * btc_exit_price
        btc_profit = usd_received_from_short - usd_spent_covering_short
        btc_fees_usd = btc_entry_fee * btc_entry_price + btc_exit_fee * btc_exit_price

        short_notional = amount_btc_shorted * btc_entry_price
        expected_borrow_cost = borrow_rate * 1.0 * short_notional

        expected_total_profit = eth_profit + btc_profit - expected_borrow_cost

        summary = result["summary_header"]
        waterfall = result["waterfall_data"]

        self.assertAlmostEqual(summary["total_profit"], expected_total_profit)
        self.assertAlmostEqual(summary["borrow_costs"], expected_borrow_cost)
        self.assertAlmostEqual(result["borrow_costs"], expected_borrow_cost)
        self.assertAlmostEqual(waterfall["Coin-1 Fees"], -eth_fees_usd)
        self.assertAlmostEqual(waterfall["Coin-2 Fees"], -btc_fees_usd)
        self.assertAlmostEqual(waterfall["Borrow/Funding Costs"], -expected_borrow_cost)


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
