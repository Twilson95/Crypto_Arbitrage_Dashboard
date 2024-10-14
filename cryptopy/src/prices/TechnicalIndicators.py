import pandas as pd


class TechnicalIndicators:
    def __init__(self):
        self.indicator_options = [
            {"label": "SMA", "value": "SMA"},
            {"label": "EMA", "value": "EMA"},
            {"label": "Bollinger Bands", "value": "BB"},
            {"label": "RSI", "value": "RSI"},
            {"label": "MACD", "value": "MACD"},
            {"label": "ATR", "value": "ATR"},
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
        if "BB" in indicators:
            (
                indicator_dict["BB_upper"],
                indicator_dict["BB_middle"],
                indicator_dict["BB_lower"],
            ) = TechnicalIndicators.bollinger_bands(df["close"])
        if "RSI" in indicators:
            indicator_dict["RSI"] = TechnicalIndicators.rsi(df["close"])
        if "MACD" in indicators:
            (
                indicator_dict["MACD"],
                indicator_dict["MACD_signal"],
                indicator_dict["MACD_hist"],
            ) = TechnicalIndicators.macd(df["close"])
        if "ATR" in indicators:
            indicator_dict["ATR"] = TechnicalIndicators.atr(df)

        return indicator_dict

    @staticmethod
    def bollinger_bands(series, window=20, num_std_dev=2):
        middle_band = series.rolling(window=window).mean()
        std_dev = series.rolling(window=window).std()
        upper_band = middle_band + (std_dev * num_std_dev)
        lower_band = middle_band - (std_dev * num_std_dev)
        return upper_band, middle_band, lower_band

    @staticmethod
    def rsi(series, window=14):
        delta = series.diff(1)
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    @staticmethod
    def macd(series, fast_window=12, slow_window=26, signal_window=9):
        fast_ema = series.ewm(span=fast_window, adjust=False).mean()
        slow_ema = series.ewm(span=slow_window, adjust=False).mean()
        macd = fast_ema - slow_ema
        signal = macd.ewm(span=signal_window, adjust=False).mean()
        hist = macd - signal
        return macd, signal, hist

    @staticmethod
    def atr(df, window=14):
        high_low = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift()).abs()
        low_close = (df["low"] - df["close"].shift()).abs()
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        atr = true_range.rolling(window=window).mean()
        return atr

    def get_indicator_options(self):
        return self.indicator_options
