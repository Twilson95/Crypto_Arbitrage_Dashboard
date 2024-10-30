import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint
from dataclasses import dataclass, field
from typing import Optional, Dict
from cryptopy import StatisticalArbitrage
from datetime import timedelta


@dataclass
class CointegrationData:
    pair: tuple
    p_value: float
    spread: Optional[pd.Series]
    hedge_ratio: Optional[float]
    trade_details: Dict = field(default_factory=dict)

    def update_latest_spread(self, price1, price2):
        self.spread.iloc[-1] = price1 - self.hedge_ratio * price2

    def update_trade_details(self):
        self.trade_details = StatisticalArbitrage.get_statistical_arbitrage_trades(
            self.spread
        )

    def is_open_opportunity(self):
        open_opportunity = False
        if self.trade_details.get("trade_status") == "open":
            current_spread = self.spread.iloc[-1]
            upper_threshold = self.trade_details["upper_threshold"].iloc[-1]
            lower_threshold = self.trade_details["lower_threshold"].iloc[-1]
            if current_spread > upper_threshold or current_spread < lower_threshold:
                open_opportunity = True

        return open_opportunity


class CointegrationCalculator:

    @staticmethod
    def find_cointegration_pairs(df, precalculated_pairs):
        # Start with the pre-existing cointegration pairs
        cointegration_pairs = precalculated_pairs

        for column_1 in df.columns:
            for column_2 in df.columns:
                if column_1 == column_2:
                    continue
                # pair = tuple(sorted([column_1, column_2]))
                pair = tuple(
                    sorted(
                        [column_1, column_2], key=lambda x: df[x].mean(), reverse=True
                    )
                )
                if pair in cointegration_pairs:
                    continue

                coint_stat, p_value, crit_values = (
                    CointegrationCalculator.test_cointegration(df, pair)
                )
                spread, hedge_ratio = None, None
                if p_value < 0.05:
                    spread, hedge_ratio = CointegrationCalculator.calculate_spread(
                        df, pair
                    )

                cointegration_pairs[pair] = CointegrationData(
                    pair=pair, p_value=p_value, spread=spread, hedge_ratio=hedge_ratio
                )

        return cointegration_pairs

    # @staticmethod
    # def find_cointegration_pairs(df, precalculated_pairs, run_date=None, days_back=100):
    #     if run_date is None:
    #         run_date = pd.Timestamp.today()
    #     else:
    #         # Convert to pd.Timestamp if the user provides a string or date object
    #         run_date = pd.to_datetime(run_date)
    #
    #     start_date = run_date - pd.Timedelta(days=days_back)
    #     df_filtered = df[(df.index >= start_date) & (df.index <= run_date)]
    #
    #     cointegration_pairs = precalculated_pairs
    #
    #     # Iterate through pairs of columns (time series) in the DataFrame
    #     for column_1 in df_filtered.columns:
    #         for column_2 in df_filtered.columns:
    #             if column_1 == column_2:
    #                 continue
    #
    #             # Sort the pair based on mean prices (descending) and create the tuple
    #             pair = tuple(
    #                 sorted(
    #                     [column_1, column_2],
    #                     key=lambda x: df_filtered[x].mean(),
    #                     reverse=True,
    #                 )
    #             )
    #
    #             # If this pair has already been processed, skip it
    #             if pair in cointegration_pairs:
    #                 continue
    #
    #             # Perform the cointegration test
    #             coint_stat, p_value, crit_values = (
    #                 CointegrationCalculator.test_cointegration(df_filtered, pair)
    #             )
    #
    #             spread, hedge_ratio = None, None
    #             # If the p-value is below 0.05, calculate the spread and hedge ratio
    #             if p_value < 0.05:
    #                 spread, hedge_ratio = CointegrationCalculator.calculate_spread(
    #                     df_filtered, pair
    #                 )
    #
    #             # Store the cointegration results in the dictionary
    #             cointegration_pairs[pair] = CointegrationData(
    #                 pair=pair, p_value=p_value, spread=spread, hedge_ratio=hedge_ratio
    #             )
    #
    #     return cointegration_pairs

    @staticmethod
    def test_cointegration(df, pair):
        prices1 = df[pair[0]]
        prices2 = df[pair[1]]

        if prices1.isnull().any() or prices2.isnull().any():
            # print(f"{pair} One of the series contains NaN values. Returning None.")
            return None, None, None

        coint_result = coint(prices1, prices2)
        return coint_result

    @staticmethod
    def calculate_all_spreads(df, pairs):
        spreads = {}
        for pair in pairs:
            spread, hedge_ratio = CointegrationCalculator.calculate_spread(df, pair)
            spreads[pair] = {"spread": spread, "hedge_ratio": hedge_ratio}

        return spreads

    @staticmethod
    def calculate_spread(df, pair, hedge_ratio=None):
        prices1 = df[pair[0]]
        prices2 = df[pair[1]]

        if hedge_ratio is None:
            model = sm.OLS(prices1, sm.add_constant(prices2))
            result = model.fit()
            hedge_ratio = result.params.iloc[1]

        spread = prices1 - hedge_ratio * prices2
        spread = spread.dropna()

        return spread, hedge_ratio

    @staticmethod
    def identify_arbitrage_opportunities(spread, threshold=2):
        mean_spread = spread.mean()
        std_spread = spread.std()

        # Identify points where the spread deviates significantly from the mean
        spread_zscore = (spread - mean_spread) / std_spread
        arbitrage_signals = spread_zscore.abs() > threshold

        opportunities = pd.DataFrame(
            {"Spread": spread, "Z-Score": spread_zscore, "Arbitrage": arbitrage_signals}
        )

        return opportunities
