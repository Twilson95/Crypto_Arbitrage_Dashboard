import pandas as pd


class OHLCData:
    def __init__(self):
        self.datetime = []
        self.open = []
        self.high = []
        self.low = []
        self.close = []
        self.volume = []

    def to_dict(self):
        return {
            "datetime": self.datetime,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            # "volume": self.volume,
        }

    def update_from_dataframe(self, df):
        self.datetime.extend(df["datetime"].tolist())
        self.open.extend(df["open"].tolist())
        self.high.extend(df["high"].tolist())
        self.low.extend(df["low"].tolist())
        self.close.extend(df["close"].tolist())
        self.volume.extend(df["volume"].tolist())

    def to_dataframe(self):
        data_dict = self.to_dict()
        return pd.DataFrame(data_dict)

    def update_with_latest_value(self, new_price, new_datetime):
        # self.datetime = []
        # self.open = []
        if new_price > self.high[-1]:
            self.high[-1] = new_price
        if new_price < self.low[-1]:
            self.low[-1] = new_price
        self.close[-1] = new_price

    def __repr__(self):
        return (
            f"datetime: {self.datetime}\n"
            f"open: {self.open}\n"
            f"high: {self.high}\n"
            f"low: {self.low}\n"
            f"close: {self.close}\n"
            f"volume: {self.volume}\n"
        )
