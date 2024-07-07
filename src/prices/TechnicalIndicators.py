import pandas as pd


class TechnicalIndicators:
    def __init__(self):
        self.indicator_options = [
            {"label": "SMA", "value": "SMA"},
            {"label": "EMA", "value": "EMA"},
        ]

    @staticmethod
    def apply_indicators(prices, indicators):
        df = prices.to_dataframe()
        indicator_dict = {"datetime": df["datetime"]}

        if not indicators:
            return indicator_dict

        if "SMA" in indicators:
            indicator_dict["SMA"] = df["close"].rolling(window=20).mean()
        if "EMA" in indicators:
            indicator_dict["EMA"] = df["close"].ewm(span=20, adjust=False).mean()
        return indicator_dict

    def get_indicator_options(self):
        return self.indicator_options
