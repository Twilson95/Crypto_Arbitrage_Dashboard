import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint


class CointegrationCalculator:

    @staticmethod
    def find_cointegration_pairs(df, precalculated_pairs):
        # Start with the pre-existing cointegration pairs
        cointegration_pairs = precalculated_pairs.copy()

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

                # Skip the pair if it's already in the precalculated pairs
                if pair in cointegration_pairs:
                    continue

                coint_stat, p_value, crit_values = (
                    CointegrationCalculator.test_cointegration(df, pair[0], pair[1])
                )

                cointegration_pairs[pair] = p_value

        return cointegration_pairs

    @staticmethod
    def test_cointegration(df, coin1, coin2):
        prices1 = df[coin1]
        prices2 = df[coin2]

        coint_result = coint(prices1, prices2)
        return coint_result

    @staticmethod
    def calculate_all_spreads(df, pairs):
        spreads = {}
        for pair in pairs:
            spread, hedge_ratio = CointegrationCalculator.calculate_spread(
                df, pair[0], pair[1]
            )
            spreads[pair] = {"spread": spread, "hedge_ratio": hedge_ratio}

        return spreads

    @staticmethod
    def calculate_spread(df, coin1, coin2):
        prices1 = df[coin1]
        prices2 = df[coin2]

        # Use linear regression to find the hedge ratio
        model = sm.OLS(prices1, sm.add_constant(prices2))
        result = model.fit()
        hedge_ratio = result.params.iloc[1]

        # Calculate spread
        spread = prices1 - hedge_ratio * prices2
        spread = spread.dropna()

        return {"spread": spread, "hedge_ratio": hedge_ratio}

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
