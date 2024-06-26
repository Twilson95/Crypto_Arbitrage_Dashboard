from abc import abstractmethod
from bitmex import bitmex
from datetime import datetime, timedelta, timezone
from dateutil.tz import tzutc
import time
import pandas as pd
import ccxt
from src.OHLCData import OHLCData
import src.Warnings_to_ignore


bitmex_api_key = "Hi7WUgzxyzCRY_1BJ0e7meab"
bitmex_api_secret = "A64KwkbRqURgFAmarfF758ceAVtFoIJWZIXe5lpRIeld1FxD"

coinbase_API_key = "organizations/3a515644-f82d-412e-956e-38ee985c06d4/apiKeys/4b4bca01-ec6c-49b5-ad98-c427a6a80a78"
coinbase_API_secret = (
    "-----BEGIN EC PRIVATE KEY-----\nMHcCAQEEILioy4fqfPSC8p2Z2OhdxJo+rw3X5yNON+dutYXPNMgBoAoGCCqG"
    "SM49\nAwEHoUQDQgAEGFpwjI3Bs7ePdVjsOOsQs77dZZc1acrlgFkX69KJFslavy/6qlQM\nFRTUDrQeeyF1Xyjc1VhV"
    "LdOJUjlsosvXFA==\n-----END EC PRIVATE KEY-----\n"
)


class DataFetcher:
    def __init__(self):
        self.historical_data = {}
        self.live_data = {}
        self.fetch_all_initial_live_prices(count=10)
        self.update_all_historical_prices()

    def initialize_ohlc_data(self, currency):
        self.historical_data[currency] = OHLCData()
        self.live_data[currency] = OHLCData()

    def get_historical_prices(self, currency):
        return self.historical_data[currency]

    def get_live_prices(self, currency):
        return self.live_data[currency]

    def fetch_all_live_prices(self):
        for currency in self.currencies.keys():
            self.fetch_live_price(currency)
            time.sleep(0.1)

    @abstractmethod
    def fetch_live_price(self, currency):
        pass

    def fetch_all_initial_live_prices(self, count=10):
        for currency in self.currencies.keys():
            self.fetch_initial_live_prices(currency, count)
            time.sleep(0.1)

    @abstractmethod
    def fetch_initial_live_prices(self, currency, count):
        pass

    def update_all_historical_prices(self):
        for currency in self.currencies.keys():
            self.update_historical_prices(currency)

    @abstractmethod
    def update_historical_prices(self, currency):
        pass


class BitmexDataFetcher(DataFetcher):
    def __init__(self):
        # print("initialise")
        self.client = bitmex(
            test=False, api_key=bitmex_api_key, api_secret=bitmex_api_secret
        )
        self.currencies = {
            "bitcoin": "XBTUSD",
            "ethereum": "ETHUSD",
            "solana": "SOLUSD",
            "dogecoin": "DOGEUSD",
            "cardano": "ADAUSD",
            "ripple": "XRPUSD",
        }
        super().__init__()

    def fetch_initial_live_prices(self, currency, count):
        bitmex_symbol = self.currencies[currency]
        now = datetime.now(tzutc())
        start_time = now - timedelta(seconds=10 * count)

        historical_data = self.client.Trade.Trade_get(
            symbol=bitmex_symbol,
            startTime=start_time,
            count=1000,
        ).result()

        if currency not in self.live_data.keys():
            self.initialize_ohlc_data(currency)

        data = historical_data[0]
        data.sort(key=lambda x: x["timestamp"])

        interval_start = start_time
        interval_end = interval_start + timedelta(seconds=10)
        last_price = None

        datetimes = []
        price_data = []

        data_index = 0
        while interval_start < now:
            # Find the closest previous trade
            while (
                data_index < len(data) and data[data_index]["timestamp"] < interval_end
            ):
                last_price = data[data_index]["price"]
                data_index += 1

            # Use the last known price if no trade in this interval
            datetimes.append(interval_start)
            price_data.append(last_price)

            interval_start = interval_end
            interval_end = interval_start + timedelta(seconds=10)

        self.live_data[currency].datetime = datetimes[1:]
        self.live_data[currency].open = price_data[1:]
        self.live_data[currency].high = price_data[1:]
        self.live_data[currency].low = price_data[1:]
        self.live_data[currency].close = price_data[1:]

    def fetch_live_price(self, currency):
        bitmex_symbol = self.currencies[currency]
        instrument_data = self.client.Instrument.Instrument_get(
            symbol=bitmex_symbol,
            # reverse=True
        ).result()
        item = instrument_data[0][0]

        if currency not in self.live_data.keys():
            self.initialize_ohlc_data(currency)

        self.live_data[currency].datetime.append(item["timestamp"])
        self.live_data[currency].open.append(item["lastPrice"])
        self.live_data[currency].high.append(item["lastPrice"])
        self.live_data[currency].low.append(item["lastPrice"])
        self.live_data[currency].close.append(item["lastPrice"])
        self.live_data[currency].volume.append(item["volume"])

    def update_historical_prices(self, currency):
        bitmex_symbol = self.currencies[currency]
        historical_data = self.client.Trade.Trade_getBucketed(
            symbol=bitmex_symbol,
            binSize="1d",
            count=100,
            reverse=True,
            partial=True,
            columns="timestamp, symbol, open, high, low, close, volume",
        ).result()
        items = reversed(historical_data[0])
        # print("history prices: ", items)
        for item in items:  # Insert oldest first
            self.historical_data[currency].datetime.append(
                item["timestamp"] - timedelta(days=1)
            )
            self.historical_data[currency].open.append(item["open"])
            self.historical_data[currency].high.append(item["high"])
            self.historical_data[currency].low.append(item["low"])
            self.historical_data[currency].close.append(item["close"])
            self.historical_data[currency].volume.append(item["volume"])


class CoinbaseDataFetcher(DataFetcher):
    def __init__(self):
        self.client = ccxt.coinbase(
            {
                "apiKey": coinbase_API_key,
                "secret": coinbase_API_secret,
            }
        )

        self.currencies = {
            "bitcoin": "BTC/USD",
            "ethereum": "ETH/USD",
            "solana": "SOL/USD",
            "dogecoin": "DOGE/USD",
            "cardano": "ADA/USD",
            "ripple": "XRP/USD",
        }
        super().__init__()

    def fetch_initial_live_prices(self, currency, count):
        symbol = self.currencies[currency]
        since = int((datetime.now() - timedelta(seconds=1000)).timestamp() * 1000)
        trades_df = self.fetch_trades_within_timeframe(symbol, since)
        # Bucket trades into 10-second intervals and aggregate into OHLCV
        ohlcv_df = self.bucket_trades_into_intervals(
            trades_df, interval="10s"
        ).reset_index()

        if currency not in self.historical_data.keys():
            self.initialize_ohlc_data(currency)

        self.historical_data[currency].update_from_dataframe(ohlcv_df)

    def fetch_trades_within_timeframe(self, symbol, since):
        until = self.client.parse8601(datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"))
        trades = self.client.fetch_trades(symbol, since=since, params={"until": until})
        data = [
            {
                "timestamp": trade["timestamp"],
                "price": trade["price"],
                "amount": trade["amount"],
            }
            for trade in trades
        ]
        df = pd.DataFrame(data)
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        return df

    @staticmethod
    def bucket_trades_into_intervals(df, interval="10S"):

        # Resample trades into intervals and aggregate into OHLCV
        ohlcv = df.resample(interval, on="datetime").agg(
            {"price": ["first", "max", "min", "last"], "amount": "sum"}
        )
        # Rename columns
        ohlcv.columns = ["open", "high", "low", "close", "volume"]

        # Forward-fill missing data
        ohlcv["open"] = ohlcv["open"].ffill()
        ohlcv["high"] = ohlcv["high"].ffill()
        ohlcv["low"] = ohlcv["low"].ffill()
        ohlcv["close"] = ohlcv["close"].ffill()
        ohlcv["volume"] = ohlcv["volume"].fillna(0)

        return ohlcv

    def fetch_live_price(self, currency):
        symbol = self.currencies[currency]
        ticker = self.client.fetch_ticker(symbol)
        timestamp_ms = ticker["timestamp"]
        timestamp_s = timestamp_ms / 1000
        datetime_obj = datetime.fromtimestamp(timestamp_s)

        current_price = ticker["last"]
        self.live_data[currency].datetime.append(datetime_obj)
        self.live_data[currency].high.append(current_price)
        self.live_data[currency].low.append(current_price)
        self.live_data[currency].close.append(current_price)
        self.live_data[currency].open.append(current_price)

    def update_historical_prices(self, currency):
        symbol = self.currencies[currency]
        timeframe = "1d"  # Daily data
        since = self.client.parse8601(
            (datetime.today() - timedelta(days=100)).strftime("%Y-%m-%dT%H:%M:%SZ")
        )
        data = self.client.fetch_ohlcv(
            symbol,
            timeframe,
            since,  # params={"until": 1719014400 * 1000}
        )

        df = pd.DataFrame(
            data, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)

        if currency not in self.historical_data.keys():
            self.initialize_ohlc_data(currency)

        self.historical_data[currency].update_from_dataframe(df)
